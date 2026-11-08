"""Regret of keeping the final checkpoint and of each selector, the oracle and its bias, collapses.

The paper reports these results in Sec. 4.1-4.3 and App. A-C. Success rates are fractions; every
regret and drop returned here is in points (percent).
"""

import statistics

import numpy as np

from ckpt_selection import config as config_lib
from ckpt_selection import layout, selectors, stats

# Confirm rates are counts over a fixed number of episodes, so differences in points are exact
# multiples of a small step. Rounding removes float residue in the last digits before any
# comparison with a threshold; it changes no printed value.
POINTS_DECIMALS = 9

TASK8_ROLES = ("earlier", "replication")  # the task-8 runs (Sec. 3)
TRAINED_ROLES = ("earlier", "replication", "control")  # runs trained for the full schedule
# Runs whose median R_last the sensitivity test recomputes with each run left out (App. B).
SENSITIVITY_ROLES = ("earlier", "control")

# Selectors compared with keeping the final checkpoint: (key, label, selector kind).
COMPARED_SELECTORS = (
    ("final", "Keeping the final checkpoint", "rollout_free"),
    ("spread_veto", "Ensemble-spread veto", "rollout_free"),
    ("lowest_entropy", "Lowest action entropy", "rollout_free"),
    ("highest_heldout_q", "Highest held-out Q", "rollout_free"),
    ("restricted_critic", "Restricted critic", "rollout_free"),
    ("highest_scan", "Highest scan success (reference)", "scan"),
)


def points(rate_difference: float) -> float:
    """Return a difference of two success rates in points."""
    return round(100 * rate_difference, POINTS_DECIMALS)


def oracle(run) -> float:
    """Return s*, the run's best confirmed success rate (a fraction)."""
    if not run.confirm:
        raise ValueError(f"run {run.tag} has no confirm files")
    return max(run.confirm.values())


def regret(run, step: int) -> float | None:
    """Return the regret of one checkpoint in points, or None if it was not confirmed."""
    if step not in run.confirm:
        return None
    return points(oracle(run) - run.confirm[step])


def r_last(run) -> float:
    """Return R_last, the regret of the final checkpoint in points."""
    final = selectors.final_step(run)
    if final not in run.confirm:
        raise ValueError(f"run {run.tag}: the final checkpoint {final} has no confirm file")
    return regret(run, final)


def scan_argmax_r_last(run, tie: str) -> float:
    """Return R_last under the scan-argmax oracle, in points.

    The value is the confirmed success of the highest-scanning checkpoint minus the final
    checkpoint's, so it is negative when the final checkpoint confirms higher.
    """
    step = selectors.scan_argmax(run, tie)
    final = selectors.final_step(run)
    for checkpoint in (step, final):
        if checkpoint not in run.confirm:
            raise ValueError(f"run {run.tag}: checkpoint {checkpoint} has no confirm file")
    return points(run.confirm[step] - run.confirm[final])


def summarize_regrets(per_run: dict) -> dict:
    """Return the mean, median and worst of per-run regrets."""
    values = list(per_run.values())
    return {
        "mean": float(np.mean(values)),
        "median": float(statistics.median(values)),
        "worst": float(max(values)),
    }


def selector_picks(run, config: dict) -> dict:
    """Return the checkpoint each compared selector picks on one run."""
    tie = config["selectors"]["tie_rule"]
    return {
        "final": selectors.final_step(run),
        "spread_veto": selectors.spread_veto(run, config["selectors"]["spread_veto_mads"]),
        "lowest_entropy": selectors.lowest_entropy(run, tie),
        "highest_heldout_q": selectors.highest_heldout_q(run, tie),
        "restricted_critic": selectors.restricted_critic(
            run, config["selectors"]["restricted_min_step"], tie
        ),
        "highest_scan": selectors.scan_argmax(run, tie),
    }


def scan_episodes_per_run(runs, config: dict) -> int:
    """Return the evaluation episodes a full scan of one run takes (checkpoints x scan episodes)."""
    counts = {len(config_lib.checkpoint_steps(config, run.tag)) for run in runs}
    if len(counts) != 1:
        raise ValueError(f"runs have different checkpoint counts: {sorted(counts)}")
    return counts.pop() * config["evaluation"]["scan"]["episodes"]


def selector_regret_table(runs, config: dict) -> dict:
    """Return, per compared selector, its pick and regret on each run and their summaries."""
    picks = {run.tag: selector_picks(run, config) for run in runs}
    scan_episodes = scan_episodes_per_run(runs, config)
    table = {}
    for key, label, kind in COMPARED_SELECTORS:
        per_run = {}
        for run in runs:
            step = picks[run.tag][key]
            value = regret(run, step)
            if value is None:
                raise ValueError(f"run {run.tag}: the {key} pick {step} has no confirm file")
            per_run[run.tag] = value
        table[key] = {
            "label": label,
            "kind": kind,
            "picks": {tag: picks[tag][key] for tag in picks},
            "per_run": per_run,
            **summarize_regrets(per_run),
            "episodes": scan_episodes if kind == "scan" else 0,
        }
    return table


def compare_to_final(per_run: dict, r_last_by_tag: dict, floor: float) -> dict:
    """Count the runs where a selector's regret is above R_last, by any amount and by the floor."""
    differences = {tag: per_run[tag] - r_last_by_tag[tag] for tag in per_run}
    return {
        "worse": sum(value > 0 for value in differences.values()),
        "worse_over_floor": sum(value > floor for value in differences.values()),
        "difference": differences,
    }


def restricted_at_cutoff(selector_table: dict, config: dict) -> dict:
    """Return the runs whose restricted-critic pick is the earliest allowed checkpoint.

    Also counts how many of the runs with the largest restricted-critic regret are among them.
    selector_table is selector_regret_table's result.
    """
    cutoff = config["selectors"]["restricted_min_step"]
    picks = selector_table["restricted_critic"]["picks"]
    per_run = selector_table["restricted_critic"]["per_run"]
    at_cutoff = [tag for tag, step in picks.items() if step == cutoff]
    by_regret = sorted(per_run, key=per_run.get, reverse=True)
    leading = 0
    for tag in by_regret:
        if tag not in at_cutoff:
            break
        leading += 1
    return {
        "cutoff_step": cutoff,
        "count": len(at_cutoff),
        "n_runs": len(picks),
        "tags": at_cutoff,
        "largest_regret_runs_at_cutoff": leading,
    }


def confirmed_drops(run, config: dict) -> list[dict]:
    """Return every confirmed drop of more than the collapse threshold that is followed by a rise.

    Each drop runs from one confirmed checkpoint to the next and is reported at the step it lands
    on; it counts as recovered when the next confirmed checkpoint is higher. `competent` says
    whether the drop starts above the run's base-policy scan success (Sec. 3).
    """
    threshold = config["protocol"]["collapse_points"]
    base_scan = run.scan[layout.BASE_STEP]
    steps = sorted(run.confirm)
    drops = []
    for before, step, after in zip(steps, steps[1:], steps[2:]):
        drop = points(run.confirm[before] - run.confirm[step])
        if drop > threshold and run.confirm[after] > run.confirm[step]:
            drops.append(
                {
                    "step": step,
                    "from_step": before,
                    "drop": drop,
                    "from_rate": run.confirm[before],
                    "to_rate": run.confirm[step],
                    "competent": run.confirm[before] > base_scan,
                }
            )
    return drops


def collapses(run, config: dict) -> list[dict]:
    """Return the run's collapses: competent confirmed drops followed by a rise."""
    return [drop for drop in confirmed_drops(run, config) if drop["competent"]]


def non_competent_drops(run, config: dict) -> list[dict]:
    """Return confirmed drops with recovery that start from a checkpoint that is not competent."""
    return [drop for drop in confirmed_drops(run, config) if not drop["competent"]]


def largest_scan_fall(run) -> dict:
    """Return the largest fall in scan success from one checkpoint to any later one.

    The result also says whether the fall starts above the base policy's scan success.
    """
    steps = sorted(step for step in run.scan if step != layout.BASE_STEP)
    best = {"drop": 0.0}
    for index, start in enumerate(steps):
        for end in steps[index + 1 :]:
            drop = points(run.scan[start] - run.scan[end])
            if drop > best["drop"]:
                best = {"drop": drop, "from_step": start, "to_step": end}
    if "from_step" in best:
        best["from_above_base"] = run.scan[best["from_step"]] > run.scan[layout.BASE_STEP]
        best["base_scan"] = run.scan[layout.BASE_STEP]
        best["from_rate"] = run.scan[best["from_step"]]
    return best


def final_drop(run) -> dict:
    """Return the fall from the checkpoint saved just before the final checkpoint to the final.

    The fall uses confirmed success if both checkpoints are confirmed, else scan success.
    """
    final = selectors.final_step(run)
    previous = max(step for step in run.scan if step < final and step != layout.BASE_STEP)
    if previous in run.confirm:
        if final not in run.confirm:
            raise ValueError(f"run {run.tag}: the final checkpoint {final} has no confirm file")
        return {
            "kind": "confirmed",
            "from_step": previous,
            "from_rate": run.confirm[previous],
            "final_rate": run.confirm[final],
            "drop": points(run.confirm[previous] - run.confirm[final]),
        }
    return {
        "kind": "scan",
        "from_step": previous,
        "from_rate": run.scan[previous],
        "final_rate": run.scan[final],
        "drop": points(run.scan[previous] - run.scan[final]),
    }


def terminal_drops(runs, config: dict) -> dict:
    """Return the fall into the final checkpoint of each run whose R_last exceeds the floor."""
    floor = config["protocol"]["regret_floor"]
    return {run.tag: final_drop(run) for run in runs if r_last(run) > floor}


def largest_confirmed_final_drop(runs) -> str:
    """Return the run whose final checkpoint falls most below the confirmed checkpoint before it."""
    drops = {run.tag: final_drop(run) for run in runs}
    confirmed = {tag: drop["drop"] for tag, drop in drops.items() if drop["kind"] == "confirmed"}
    if not confirmed:
        raise ValueError("no run has a confirmed checkpoint just before its final checkpoint")
    return max(confirmed, key=confirmed.get)


def final_regret_row(run, config: dict) -> dict:
    """Return one run's R_last, scan-argmax values, confirm-set size and collapses."""
    return {
        "tag": run.tag,
        "task": run.task,
        "role": run.role,
        "confirm_count": len(run.confirm),
        "r_last": r_last(run),
        **{f"scan_argmax_{tie}": scan_argmax_r_last(run, tie) for tie in selectors.TIE_RULES},
        "collapses": [
            {"step": drop["step"], "drop": drop["drop"]} for drop in collapses(run, config)
        ],
    }


def group_medians(rows: list[dict]) -> dict:
    """Return the medians of R_last and of both scan-argmax values over some final_regret_rows."""
    columns = ("r_last", *(f"scan_argmax_{tie}" for tie in selectors.TIE_RULES))
    return {column: float(statistics.median(row[column] for row in rows)) for column in columns}


def final_regret_table(runs, config: dict) -> dict:
    """Return R_last of every run trained for the full schedule, with group medians and counts.

    The runs are those with a role in TRAINED_ROLES. Rows are sorted by role, then by R_last,
    highest first.
    """
    floor = config["protocol"]["regret_floor"]
    rows = [final_regret_row(run, config) for run in runs]
    rows.sort(key=lambda row: (TRAINED_ROLES.index(row["role"]), -row["r_last"]))
    task8 = [row for row in rows if row["role"] in TASK8_ROLES]
    earlier = [row for row in rows if row["role"] == "earlier"]
    control = [row for row in rows if row["role"] == "control"]
    return {
        "rows": rows,
        "medians": {
            "earlier": group_medians(earlier),
            "task8": group_medians(task8),
            "earlier_and_control": group_medians(
                [row for row in rows if row["role"] in SENSITIVITY_ROLES]
            ),
        },
        "group_sizes": {
            "earlier": len(earlier),
            "task8": len(task8),
            "control": len(control),
        },
        "count_over_floor_task8": sum(row["r_last"] > floor for row in task8),
        "max_control": max(row["r_last"] for row in control),
        "scan_argmax": {
            tie: {
                "count_over_floor_task8": sum(row[f"scan_argmax_{tie}"] > floor for row in task8),
                "median_task8": float(
                    statistics.median(row[f"scan_argmax_{tie}"] for row in task8)
                ),
            }
            for tie in selectors.TIE_RULES
        },
        "collapse_runs": sum(bool(row["collapses"]) for row in task8),
        "collapse_events": sum(len(row["collapses"]) for row in task8),
    }


def median_sensitivity(r_last_by_tag: dict) -> dict:
    """Return the median R_last, the median with each run left out, and the runs that change it."""
    full = float(statistics.median(r_last_by_tag.values()))
    leave_one_out = {}
    for tag in r_last_by_tag:
        rest = [value for other, value in r_last_by_tag.items() if other != tag]
        leave_one_out[tag] = float(statistics.median(rest))
    changed = {tag: value for tag, value in leave_one_out.items() if value != full}
    return {"median": full, "leave_one_out": leave_one_out, "changed_by_leaving_out": changed}


def rerun_pairs(runs: dict) -> dict:
    """Return R_last and the fall into the final checkpoint of each rerun and of its original."""
    pairs = {}
    for run in runs.values():
        if run.rerun_of is None:
            continue
        original = runs[run.rerun_of]
        pairs[run.tag] = {
            "original": original.tag,
            "rerun_r_last": r_last(run),
            "original_r_last": r_last(original),
            "rerun_final_drop": final_drop(run),
            "original_final_drop": final_drop(original),
        }
    return pairs


def oracle_bias(runs, config: dict) -> dict:
    """Return the upward bias of s* under three models of the true rates (App. B "Oracle bias").

    For each run, the bias is E[max of independent binomial estimates] minus the largest true rate,
    with the confirm-pool size as the number of trials. True rates are: the observed confirmed
    rates; the observed rates with near-ties shrunk toward their mean; every rate tied at the best.
    """
    episodes = config["evaluation"]["confirm"]["episodes"]
    band = config["protocol"]["oracle_bias_shrink_band"]
    per_model = {"independent": {}, "shrunk": {}, "all_tied": {}}
    for run in runs:
        rates = np.array([run.confirm[step] for step in sorted(run.confirm)])
        truths = {
            "independent": rates,
            "shrunk": stats.shrink_near_ties(rates, band, episodes),
            "all_tied": np.full(len(rates), rates.max()),
        }
        for model, truth in truths.items():
            bias = stats.expected_max_binomial(truth, episodes) - truth.max()
            per_model[model][run.tag] = 100 * bias
    return {
        model: {
            "median": float(np.median(list(values.values()))),
            "max": float(max(values.values())),
            "min": float(min(values.values())),
            "per_run": values,
        }
        for model, values in per_model.items()
    }


def unconfirmed_above_oracle(runs) -> dict:
    """Return the unconfirmed checkpoints whose scan success is at or above their run's oracle."""
    found = []
    for run in runs:
        best = oracle(run)
        for step in sorted(run.scan):
            if step == layout.BASE_STEP or step in run.confirm:
                continue
            if run.scan[step] >= best:
                found.append({"tag": run.tag, "step": step, "scan": run.scan[step], "oracle": best})
    return {"count": len(found), "checkpoints": found}


def task8_r_last(runs) -> dict[str, float]:
    """Return R_last per run, highest first; equal values keep the order of runs."""
    values = {run.tag: r_last(run) for run in runs}
    return dict(sorted(values.items(), key=lambda item: -item[1]))


def scan_success_by_step(runs) -> dict:
    """Return scan success at every checkpoint of every run, and the median over runs per step."""
    per_run = {
        run.tag: {step: run.scan[step] for step in selectors.candidate_steps(run)} for run in runs
    }
    return {"per_run": per_run, "median": stats.median_by_step(list(per_run.values()))}


def best_confirmed_range(runs) -> dict:
    """Return the lowest and highest best confirmed success over some runs (fractions)."""
    best = {run.tag: oracle(run) for run in runs}
    return {"min": min(best.values()), "max": max(best.values()), "per_run": best}
