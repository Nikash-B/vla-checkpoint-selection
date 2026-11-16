"""Held-out statistics through training, and L2, KL and MMD as selectors (Sec. 4.2, App. C)."""

import itertools

import numpy as np

from ckpt_selection import load, regret, selectors, stats

# Held-out statistics summarized through training: (key in heldout_<step>.json, label).
HELDOUT_STATISTICS = (
    ("l2", "L2"),
    ("kl", "KL"),
    ("mmd", "MMD"),
    ("action_entropy", "Action entropy"),
)
SD_DDOF = 0  # the plateau SD is the standard deviation over checkpoints, without correction


def plateau_values(run, key: str, plateau_start: int) -> list[float]:
    """Return one statistic at every checkpoint from plateau_start on, in step order."""
    series = selectors.heldout_series(run, key)
    return [series[step] for step in sorted(series) if step >= plateau_start]


def heldout_summary_table(runs, base: dict, config: dict) -> dict:
    """Return a summary of each held-out statistic at the base policy and through training.

    Per statistic: the base value, the range over runs of the first checkpoint's value, and the
    range over runs of each run's mean and SD over the plateau checkpoints.
    """
    plateau_start = config["heldout"]["plateau_start"]
    first_step = min(config["training"]["checkpoint_steps"])
    table = {}
    for key, label in HELDOUT_STATISTICS:
        first = [selectors.heldout_series(run, key)[first_step] for run in runs]
        means = [float(np.mean(plateau_values(run, key, plateau_start))) for run in runs]
        sds = [float(np.std(plateau_values(run, key, plateau_start), ddof=SD_DDOF)) for run in runs]
        table[key] = {
            "label": label,
            "base": base[key],
            "first_checkpoint_step": first_step,
            "first_min": min(first),
            "first_max": max(first),
            "mean_min": min(means),
            "mean_max": max(means),
            "sd_min": min(sds),
            "sd_max": max(sds),
            "n_runs": len(runs),
        }
    return table


def statistics_by_step(runs) -> dict:
    """Return each held-out statistic at every checkpoint of every run, and the median per step."""
    by_statistic = {}
    for key, _ in HELDOUT_STATISTICS:
        per_run = {run.tag: selectors.heldout_series(run, key) for run in runs}
        by_statistic[key] = {
            "per_run": per_run,
            "median": stats.median_by_step(list(per_run.values())),
        }
    return by_statistic


def confirmed_by_step(runs) -> dict[str, dict[int, float]]:
    """Return confirmed success at every confirmed checkpoint of every run."""
    return {
        run.tag: {step: run.confirm[step] for step in selectors.confirmed_steps(run)}
        for run in runs
    }


def l2_relative_to_base(value: float, base: dict) -> float:
    """Return an L2 value as a percent above (positive) or below the base policy's L2."""
    return 100 * (value / base["l2"] - 1)


def l2_above_base(runs, base: dict, config: dict) -> dict:
    """Return the range over runs of the mean plateau L2 relative to the base policy (percent)."""
    plateau_start = config["heldout"]["plateau_start"]
    per_run = {
        run.tag: l2_relative_to_base(float(np.mean(plateau_values(run, "l2", plateau_start))), base)
        for run in runs
    }
    return {"min": min(per_run.values()), "max": max(per_run.values()), "per_run": per_run}


def l2_at_largest_final_drop(runs, base: dict) -> dict:
    """Return confirmed success and relative L2 around the largest final drop.

    The run is the one whose final checkpoint falls most below the confirmed checkpoint before
    it. L2 is given relative to the base policy at those two checkpoints.
    """
    tag = regret.largest_confirmed_final_drop(runs)
    run = next(run for run in runs if run.tag == tag)
    drop = regret.final_drop(run)
    series = selectors.heldout_series(run, "l2")
    steps = (drop["from_step"], selectors.final_step(run))
    return {
        "tag": tag,
        "steps": list(steps),
        "confirmed": [run.confirm[step] for step in steps],
        "drop": drop["drop"],
        "l2_above_base": [l2_relative_to_base(series[step], base) for step in steps],
    }


def control_l2_band(control_runs, all_runs: dict) -> dict:
    """Return, per control run, the largest deviation of L2 from its task's base policy.

    The deviation is in percent, over all checkpoints. It is None when no run of the task has a
    base-policy held-out file.
    """
    band = {}
    for run in control_runs:
        try:
            base = load.base_heldout(all_runs, run.task)
        except KeyError:
            band[run.tag] = None
            continue
        series = selectors.heldout_series(run, "l2")
        band[run.tag] = max(abs(l2_relative_to_base(value, base)) for value in series.values())
    return band


def spread_at_collapsed_final(runs) -> dict:
    """Return the ensemble spread at the collapsed final checkpoint and the run's median spread.

    The run is the one whose final checkpoint falls most. The median is over its checkpoints.
    """
    tag = regret.largest_confirmed_final_drop(runs)
    run = next(run for run in runs if run.tag == tag)
    series = selectors.heldout_series(run, "ensemble_spread")
    final_spread = series[selectors.final_step(run)]
    median_spread = float(np.median(list(series.values())))
    return {
        "tag": tag,
        "final_spread": final_spread,
        "median_spread": median_spread,
        "final_below_median": final_spread < median_spread,
    }


def score_rule(runs, picks: dict, floor: float) -> dict:
    """Score one set of picks against keeping the final checkpoint.

    Scan success is scored for every run, in points above the final's scan. Confirmed regret is
    scored where the pick was confirmed.
    """
    scan_delta = {}
    confirmed = {}
    for run in runs:
        step = picks[run.tag]
        final = selectors.final_step(run)
        scan_delta[run.tag] = regret.points(run.scan[step] - run.scan[final])
        value = regret.regret(run, step)
        if value is not None:
            confirmed[run.tag] = value - regret.r_last(run)
    deltas = list(scan_delta.values())
    return {
        "picks": picks,
        "scan_delta": scan_delta,
        "mean_scan_delta": float(np.mean(deltas)),
        "scan_worse_over_floor": sum(value < -floor for value in deltas),
        "scan_better_over_floor": sum(value > floor for value in deltas),
        "confirmed_regret_minus_final": confirmed,
    }


def distance_variants(runs, config: dict) -> dict:
    """Return L2, KL and MMD used as selectors, scored against keeping the final checkpoint.

    Each rule picks the highest or lowest value of one statistic, over all checkpoints or over
    those from a minimum step on (App. C). Two counts summarize the rules. n_mean_scan_worse
    counts the rules whose mean scan success is below the final checkpoint's.
    n_more_better_than_worse_over_floor counts the rules on which more runs gain than lose by more
    than the regret floor. heldout_stats.md reports both counts.
    """
    choices = config["selectors"]
    floor = config["protocol"]["regret_floor"]
    rules = {}
    for statistic, direction, min_step in itertools.product(
        choices["distance_statistics"],
        choices["distance_directions"],
        choices["distance_min_steps"],
    ):
        picks = {
            run.tag: selectors.distance_pick(
                run, statistic, direction, min_step, choices["tie_rule"]
            )
            for run in runs
        }
        rules[f"{statistic}_{direction}_from_{min_step}"] = score_rule(runs, picks, floor)
    pick_vectors = {tuple(rule["picks"][run.tag] for run in runs) for rule in rules.values()}
    return {
        "n_rules": len(rules),
        "n_distinct": len(pick_vectors),
        "n_mean_scan_worse": sum(rule["mean_scan_delta"] < 0 for rule in rules.values()),
        "n_more_better_than_worse_over_floor": sum(
            rule["scan_better_over_floor"] > rule["scan_worse_over_floor"]
            for rule in rules.values()
        ),
        "rules": rules,
    }


def entropy_vs_final(runs, selector_table: dict, config: dict) -> dict:
    """Compare the lowest-entropy pick with keeping the final checkpoint, run by run (Sec. 4.2).

    selector_table is regret.selector_regret_table's result. The comparison covers the run whose
    final checkpoint falls most, and the other runs where the difference is beyond the floor.
    """
    floor = config["protocol"]["regret_floor"]
    collapsed = regret.largest_confirmed_final_drop(runs)
    entropy = selector_table["lowest_entropy"]["per_run"]
    final = selector_table["final"]["per_run"]
    others = [tag for tag in entropy if tag != collapsed]
    return {
        "collapsed_tag": collapsed,
        "collapsed_r_last": final[collapsed],
        "collapsed_entropy_regret": entropy[collapsed],
        "others_worse_over_floor": sum(entropy[tag] - final[tag] > floor for tag in others),
        "others_better_over_floor": sum(final[tag] - entropy[tag] > floor for tag in others),
        "mean_higher": selector_table["lowest_entropy"]["mean"] > selector_table["final"]["mean"],
        "median_higher": (
            selector_table["lowest_entropy"]["median"] > selector_table["final"]["median"]
        ),
    }
