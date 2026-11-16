"""Selection on practice success S_W in the replication runs, reported in Sec. 4.4 and App. G.

S_W(c) is the success rate of the practice episodes collected at gradient steps in (c - W, c]
(selectors.practice_success). The ranking test is the partial Spearman correlation between S_W and
scan success over all checkpoints of a run, controlling for the logarithm of the training step.
Success rates and margins are returned in percent or points, except in selection_series, which
returns fractions; correlations are unitless.
"""

import math
import statistics
from itertools import pairwise

from ckpt_selection import layout, regret, selectors, stats

MEDIAN_COLUMNS = ("r_last", "regret", "reduction", "partial_rho")


def scored_steps(run, window: int) -> dict[int, tuple[float, int]]:
    """Return S_W and its outcome count at every checkpoint; fail if a window is empty.

    The ranking test uses all checkpoints of a run, so every one needs a score, and an empty
    window is a data error. Selection alone (selectors.practice_pick) skips such checkpoints.
    """
    scores = selectors.practice_scores(run, window)
    empty = [step for step, (_, count) in scores.items() if count == 0]
    if empty:
        raise ValueError(
            f"run {run.tag}: no practice outcome in the {window}-step window at {empty}"
        )
    return scores


def partial_with_log_step(run, window: int) -> float:
    """Return the partial Spearman rho of S_W and scan success given the log training step."""
    scores = scored_steps(run, window)
    steps = sorted(scores)
    return stats.partial_spearman(
        [scores[step][0] for step in steps],
        [run.scan[step] for step in steps],
        [math.log(step) for step in steps],
    )


def replication_row(run, config: dict, window: int) -> dict:
    """Return selection on S_W at window W on one run, with the scores behind the margin.

    The margin is the selected checkpoint's S_W minus the highest S_W of any other checkpoint;
    its standard error is the binomial SE of the selected checkpoint's score.
    """
    tie = config["selectors"]["tie_rule"]
    scores = scored_steps(run, window)
    pick = selectors.practice_pick(run, window, tie)
    pick_regret = regret.regret(run, pick)
    if pick_regret is None:
        raise ValueError(f"run {run.tag}: the S_W pick {pick} has no confirm file")
    r_last = regret.r_last(run)
    partial = partial_with_log_step(run, window)
    final = selectors.final_step(run)
    pick_score, pick_count = scores[pick]
    final_score, final_count = scores[final]
    runner_up = max(score for step, (score, _) in scores.items() if step != pick)
    return {
        "run": run.tag,
        "r_last": r_last,
        "step": pick,
        "success": regret.points(run.confirm[pick]),
        "regret": pick_regret,
        "reduction": round(r_last - pick_regret, regret.POINTS_DECIMALS),
        "partial_rho": partial,
        "interval": stats.fisher_interval(partial, len(scores), controls=selectors.STEP_CONTROLS),
        "margin": regret.points(pick_score - runner_up),
        "pick_score": regret.points(pick_score),
        "pick_se": 100 * stats.binomial_se(pick_score, pick_count),
        "final_score": regret.points(final_score),
        "final_se": 100 * stats.binomial_se(final_score, final_count),
    }


def replication_table(runs, config: dict) -> dict:
    """Return selection on S_W at the primary window: one row per run, highest R_last first."""
    window = config["selectors"]["practice_window"]
    per_run = [replication_row(run, config, window) for run in runs]
    per_run.sort(key=lambda row: row["r_last"], reverse=True)
    medians = {key: statistics.median(row[key] for row in per_run) for key in MEDIAN_COLUMNS}
    standard_errors = [row["pick_se"] for row in per_run]
    return {
        "window": window,
        "runs": per_run,
        "medians": medians,
        "se_range": (min(standard_errors), max(standard_errors)),
        "all_margins_below_se": all(row["margin"] < row["pick_se"] for row in per_run),
    }


def uncontrolled_vs_step(runs, config: dict) -> dict:
    """Return rho(S_W, scan) and rho(training step, scan) per run, with medians and largest gap.

    Both correlations use all checkpoints at the primary window. The gap is the largest per-run
    absolute difference between the two.
    """
    window = config["selectors"]["practice_window"]
    per_run = {}
    for run in runs:
        scores = scored_steps(run, window)
        steps = sorted(scores)
        scan = [run.scan[step] for step in steps]
        score_rho = stats.spearman([scores[step][0] for step in steps], scan)
        step_rho = stats.spearman(steps, scan)
        per_run[run.tag] = {
            "score": score_rho,
            "step": step_rho,
            "difference": score_rho - step_rho,
        }
    return {
        "per_run": per_run,
        "median_score": statistics.median(row["score"] for row in per_run.values()),
        "median_step": statistics.median(row["step"] for row in per_run.values()),
        "max_abs_difference": max(abs(row["difference"]) for row in per_run.values()),
    }


def ranking_criteria(table: dict, uncontrolled: dict, config: dict) -> dict:
    """Return the ranking criterion of Sec. 4.4 and the stricter one, with the interval counts.

    table is replication_table's result and uncontrolled is uncontrolled_vs_step's. The criterion:
    a median partial rho of at least the configured minimum, and no selected checkpoint more than
    the configured regret below its run's best confirmed checkpoint. The stricter criterion: the
    median uncontrolled rho exceeds the training step's by the margin.
    """
    criteria = config["practice_criteria"]
    partials = [row["partial_rho"] for row in table["runs"]]
    median_partial = table["medians"]["partial_rho"]
    max_regret = max(row["regret"] for row in table["runs"])
    excess_over_step = uncontrolled["median_score"] - uncontrolled["median_step"]
    return {
        "max_selected_regret": max_regret,
        "criterion_met": median_partial >= criteria["min_median_partial_rho"]
        and max_regret <= criteria["max_selected_regret"],
        "positive_runs": sum(value > 0 for value in partials),
        "run_count": len(partials),
        "intervals_including_zero": sum(
            low <= 0 <= high for low, high in (row["interval"] for row in table["runs"])
        ),
        "excess_over_step": excess_over_step,
        "stricter_met": excess_over_step >= criteria["stricter_margin_over_step"],
    }


def moves_regret(row: dict, floor: float) -> bool:
    """Return whether selecting on S_W changes a run's regret by more than the floor (points).

    row is one of replication_table's per-run rows.
    """
    return abs(row["reduction"]) > floor


def selection_changes(table: dict, config: dict) -> dict:
    """Return the runs with large R_last and the runs on which selecting on S_W moves regret.

    table is replication_table's result. Large means at or above the regret floor; App. G's
    regret test applies only if enough runs have it. A run moves as moves_regret says.
    """
    floor = config["protocol"]["regret_floor"]
    min_large_runs = config["practice_criteria"]["min_runs_with_large_regret"]
    rows = table["runs"]
    large = [row["run"] for row in rows if row["r_last"] >= floor]
    moved = [
        {
            "run": row["run"],
            "step": row["step"],
            "r_last": row["r_last"],
            "regret": row["regret"],
            "reduction": row["reduction"],
            "share_removed": row["reduction"] / row["r_last"] if row["r_last"] > 0 else None,
        }
        for row in rows
        if moves_regret(row, floor)
    ]
    return {
        "large_r_last_runs": large,
        "large_r_last_count": len(large),
        "regret_test_applies": len(large) >= min_large_runs,
        "moved": moved,
        "moved_count": len(moved),
    }


def selection_series(run, window: int) -> dict[int, dict]:
    """Return S_W, its standard error, scan success and confirmed success at every checkpoint.

    Every value is a fraction. S_W and its SE are None where the window holds no practice outcome,
    and confirmed success is None where the checkpoint was not confirmed.
    """
    series = {}
    for step, (score, count) in selectors.practice_scores(run, window).items():
        series[step] = {
            "practice_success": score if count else None,
            "practice_se": stats.binomial_se(score, count) if count else None,
            "scan": run.scan[step],
            "confirmed": run.confirm.get(step),
        }
    return series


def expectation_check(table: dict, config: dict) -> dict:
    """Return the App. G expectation and the regret condition of the ranking criterion.

    The expectation: on runs with R_last below the floor, the selected checkpoint's regret stays
    within the configured margin of the final checkpoint's. Each failure lists S_W and its SE at
    the selected and the final checkpoint. The regret condition counts the runs whose selected
    checkpoint is at most practice_criteria.max_selected_regret points below the best.
    """
    floor = config["protocol"]["regret_floor"]
    criteria = config["practice_criteria"]
    small = [row for row in table["runs"] if row["r_last"] < floor]
    failures = [
        {
            key: row[key]
            for key in (
                "run",
                "step",
                "regret",
                "r_last",
                "pick_score",
                "pick_se",
                "final_score",
                "final_se",
            )
        }
        for row in small
        if row["regret"] - row["r_last"] > criteria["expected_regret_margin"]
    ]
    return {
        "small_r_last_count": len(small),
        "failure_count": len(failures),
        "failures": failures,
        "final_is_best_count": sum(row["r_last"] == 0 for row in table["runs"]),
        "condition_runs": sum(
            row["regret"] <= criteria["max_selected_regret"] for row in table["runs"]
        ),
        "run_count": len(table["runs"]),
    }


def window_sensitivity(runs, config: dict) -> dict:
    """Return, per sensitivity window, the median partial rho and the changed picks.

    A pick changes when it differs from the primary window's pick. The result also lists the
    changed picks that lie outside the run's confirm set.
    """
    tie = config["selectors"]["tie_rule"]
    primary = config["selectors"]["practice_window"]
    primary_picks = {run.tag: selectors.practice_pick(run, primary, tie) for run in runs}
    result = {}
    for window in config["selectors"]["practice_sensitivity_windows"]:
        picks = {run.tag: selectors.practice_pick(run, window, tie) for run in runs}
        changed = [run for run in runs if picks[run.tag] != primary_picks[run.tag]]
        result[layout.format_step(window)] = {
            "median_partial": statistics.median(partial_with_log_step(run, window) for run in runs),
            "picks": picks,
            "changed": [run.tag for run in changed],
            "changed_count": len(changed),
            "outside_confirm": [run.tag for run in changed if picks[run.tag] not in run.confirm],
            "outside_confirm_count": sum(picks[run.tag] not in run.confirm for run in changed),
        }
    return result


def window_choice(run, config: dict) -> dict:
    """Return the window choice of App. G on a run whose log records practice outcomes.

    The spacing is the median gap between logged practice outcomes. Each candidate window is a
    number of outcomes times that spacing; its score is rho(S_W, confirmed success) over the
    run's confirmed checkpoints.
    """
    grad_steps = sorted(step for step, _ in run.practice)
    spacing = statistics.median(later - earlier for earlier, later in pairwise(grad_steps))
    confirmed = sorted(step for step in run.confirm if step != layout.BASE_STEP)
    correlations = {}
    for outcomes in config["selectors"]["window_choice_outcomes"]:
        window = int(outcomes * spacing)
        scores = [selectors.practice_success(run.practice, step, window)[0] for step in confirmed]
        correlations[outcomes] = stats.spearman(scores, [run.confirm[step] for step in confirmed])
    best = max(correlations, key=correlations.get)
    return {
        "spacing": spacing,
        "confirmed_count": len(confirmed),
        "correlations": correlations,
        "best_outcomes": best,
        "best_window": int(best * spacing),
    }


def checkpoints_before_competence(run) -> int:
    """Return how many checkpoints come before the run's first competent checkpoint.

    Competent means scan success above the base policy's scan success. Sec. 3 compares success
    with the base policy's; this uses scan success because most checkpoints have no confirm
    episodes (regret.confirmed_drops, which needs confirmed checkpoints, uses confirmed success).
    practice.md reports the smallest count over the replication runs.
    """
    base = run.scan[layout.BASE_STEP]
    steps = selectors.candidate_steps(run)
    competent = [index for index, step in enumerate(steps) if run.scan[step] > base]
    return competent[0] if competent else len(steps)
