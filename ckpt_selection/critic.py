"""What the critic's value says about checkpoints: held-out Q and replay-batch Q.

The paper reports these results in Sec. 4.1 (the held-out Q peak, its step correlations, the
comparison with the base policy), App. C (held-out Q on the control runs) and App. E (replay-batch
Q from the training logs, the rules of Kumar et al., the critic's value against the return of
scan episodes). Success rates are fractions and regrets are in points (see regret.py). Critic
values and returns are in reward units: every decision before success earns a reward of -1.
"""

import math
import statistics
from itertools import pairwise

from ckpt_selection import layout, load, regret, selectors, stats


def heldout_q_peaks(runs, config: dict) -> dict:
    """Return each run's highest held-out Q checkpoint and its confirmed success (Sec. 4.1).

    Ties go as [selectors] tie_rule says. Keys: peak_runs (number of runs), peak_steps (distinct
    peak steps), peaks (per run: step and confirmed success), max_success_at_peak. Raises an
    error when a run's pick has no confirm file.
    """
    tie = config["selectors"]["tie_rule"]
    peaks = {}
    for run in runs:
        step = selectors.highest_heldout_q(run, tie)
        if step not in run.confirm:
            raise ValueError(f"run {run.tag}: the held-out Q pick {step} has no confirm file")
        peaks[run.tag] = {"step": step, "confirmed": run.confirm[step]}
    return {
        "peak_runs": len(peaks),
        "peak_steps": sorted({peak["step"] for peak in peaks.values()}),
        "peaks": peaks,
        "max_success_at_peak": max(peak["confirmed"] for peak in peaks.values()),
    }


def heldout_q_rho(run, min_step: int) -> float:
    """Return Spearman's rho of held-out Q with the training step over checkpoints >= min_step."""
    series = selectors.heldout_series(run, "heldout_q", min_step)
    return stats.spearman(list(series), list(series.values()))


def step_correlations(runs, min_step: int) -> dict:
    """Return rho(held-out Q, step) per run over all checkpoints and over those >= min_step.

    Also counts the runs on which the correlation over checkpoints >= min_step is negative.
    """
    rho_late = {run.tag: heldout_q_rho(run, min_step) for run in runs}
    return {
        "rho_all": {run.tag: heldout_q_rho(run, 0) for run in runs},
        "rho_late": rho_late,
        "late_negative_count": sum(rho < 0 for rho in rho_late.values()),
    }


def heldout_q_by_step(runs) -> dict[str, dict[int, float]]:
    """Return held-out Q at every checkpoint of every run, keyed by run tag and step."""
    return {run.tag: selectors.heldout_series(run, "heldout_q") for run in runs}


def peak_below_base(runs, config: dict) -> dict:
    """Compare the confirmed success of each run's held-out Q peak with the base's scan success.

    The standard error is that of the difference of the two rates, each measured on its own
    episodes. Returns the per-run gap in points and SE, the count below base, and the count below
    base by at least [protocol] below_base_ses standard errors.
    """
    tie = config["selectors"]["tie_rule"]
    min_ses = config["protocol"]["below_base_ses"]
    rows = {}
    for run in runs:
        step = selectors.highest_heldout_q(run, tie)
        if step not in run.confirm:
            raise ValueError(f"run {run.tag}: the held-out Q pick {step} has no confirm file")
        picked = run.confirm[step]
        base = run.scan[layout.BASE_STEP]
        se = math.hypot(
            stats.binomial_se(picked, len(run.confirm_rows[step])),
            stats.binomial_se(base, len(run.scan_rows[layout.BASE_STEP])),
        )
        rows[run.tag] = {
            "step": step,
            "confirmed": picked,
            "base_scan": base,
            "gap": regret.points(base - picked),
            "se": regret.points(se),
        }
    return {
        "below_base": rows,
        "below_base_count": sum(row["gap"] > 0 for row in rows.values()),
        "below_base_2se_count": sum(row["gap"] >= min_ses * row["se"] for row in rows.values()),
    }


def control_heldout_q(runs, config: dict) -> dict:
    """Return the highest held-out Q checkpoint on each control run (Sec. 4.1, App. C).

    Per run: the peak step, whether it lies before the restricted critic's cutoff, its confirmed
    success and regret (None without a confirm file), the base policy's scan success, and whether
    the pick confirms above that base (RL has learned the task by the peak).
    """
    tie = config["selectors"]["tie_rule"]
    cutoff = config["selectors"]["restricted_min_step"]
    peaks = {}
    for run in runs:
        step = selectors.highest_heldout_q(run, tie)
        confirmed = run.confirm.get(step)
        base = run.scan[layout.BASE_STEP]
        peaks[run.tag] = {
            "step": step,
            "early": step < cutoff,
            "confirmed": confirmed,
            "regret": regret.regret(run, step),
            "base_scan": base,
            "learned_by_peak": confirmed is not None and confirmed > base,
        }
    return {"runs": peaks, "early_count": sum(peak["early"] for peak in peaks.values())}


def latest_local_peak(values: dict[int, float]) -> int | None:
    """Return the latest step before the final one whose value exceeds both neighbours' values."""
    steps = sorted(values)
    for index in range(len(steps) - 2, 0, -1):
        step = steps[index]
        if values[steps[index - 1]] < values[step] > values[steps[index + 1]]:
            return step
    return None


def kumar_picks(q_at_checkpoints: dict, min_step: int, tie: str) -> dict:
    """Return the checkpoints that the guideline of Kumar et al. selects on one run (App. E).

    'before_decline' takes the highest value, the peak that the decline follows; 'highest_late'
    takes the highest value from min_step on; 'latest_local_peak' takes the latest local peak
    before the final checkpoint.
    """
    late = {step: value for step, value in q_at_checkpoints.items() if step >= min_step}
    return {
        "before_decline": selectors.pick(q_at_checkpoints, "max", tie),
        "highest_late": selectors.pick(late, "max", tie),
        "latest_local_peak": latest_local_peak(q_at_checkpoints),
    }


def replay_q_summary(runs, config: dict) -> dict:
    """Return replay-batch Q at the checkpoint steps of every run with a training log (App. E).

    Per run: the peak step and value, scan success at the peak, the lowest value after the peak
    (the trough) and the fall to it, rho with the step from the restricted cutoff on, the final
    value, and the picks of the Kumar et al. rules, which read [selectors] kumar_min_step and
    kumar_last_checkpoints. Then the counts and ranges over runs.
    """
    tie = config["selectors"]["tie_rule"]
    cutoff = config["selectors"]["restricted_min_step"]
    kumar_min_step = config["selectors"]["kumar_min_step"]
    last_count = config["selectors"]["kumar_last_checkpoints"]
    rows = {}
    for run in runs:
        replay_q = selectors.replay_q_series(run)
        peak = selectors.pick(replay_q, "max", tie)
        after_peak = {step: value for step, value in replay_q.items() if step > peak}
        trough = selectors.pick(after_peak, "min", tie)
        late = {step: value for step, value in replay_q.items() if step >= cutoff}
        final = selectors.final_step(run)
        rows[run.tag] = {
            "peak_step": peak,
            "peak_value": replay_q[peak],
            "scan_at_peak": run.scan[peak],
            "trough_step": trough,
            "trough_value": replay_q[trough],
            "fall": replay_q[peak] - replay_q[trough],
            "late_rho": stats.spearman(list(late), list(late.values())),
            "final_value": replay_q[final],
            "kumar": kumar_picks(replay_q, kumar_min_step, tie),
        }
    last_steps = {run.tag: selectors.candidate_steps(run)[-last_count:] for run in runs}
    picks_by_rule = {}
    for tag, row in rows.items():
        for rule, step in row["kumar"].items():
            picks_by_rule.setdefault(rule, {})[tag] = step
    kumar_counts = {}
    for rule, picks in picks_by_rule.items():
        kumar_counts[rule] = {
            "steps": sorted({step for step in picks.values() if step is not None}),
            "in_last_checkpoints": sum(picks[tag] in last_steps[tag] for tag in picks),
        }
    return {
        "replay_q": rows,
        "peak_steps": sorted({row["peak_step"] for row in rows.values()}),
        "scan_at_peak_range": [
            min(row["scan_at_peak"] for row in rows.values()),
            max(row["scan_at_peak"] for row in rows.values()),
        ],
        "fall_range": [
            min(row["fall"] for row in rows.values()),
            max(row["fall"] for row in rows.values()),
        ],
        "recovers": sum(row["final_value"] > row["trough_value"] for row in rows.values()),
        "late_positive": sum(row["late_rho"] > 0 for row in rows.values()),
        "below_peak_at_final": sum(row["final_value"] < row["peak_value"] for row in rows.values()),
        "kumar": {"last_checkpoints": last_count, **kumar_counts},
    }


def log_interval(runs) -> int:
    """Return the median over runs of each run's median gap between logged steps."""
    spacings = []
    for run in runs:
        steps = sorted(selectors.replay_q_series(run, checkpoints_only=False))
        spacings.append(statistics.median(later - earlier for earlier, later in pairwise(steps)))
    return int(statistics.median(spacings))


def replay_q_by_logged_step(runs) -> dict:
    """Return replay-batch Q at every logged step of every run, and the median over runs.

    The median covers the steps that every run logs.
    """
    per_run = {run.tag: selectors.replay_q_series(run, checkpoints_only=False) for run in runs}
    return {"per_run": per_run, "median": stats.median_by_step(list(per_run.values()))}


def full_log_peaks(runs, config: dict) -> dict:
    """Return the early replay-batch Q peak on every logged step, not only on checkpoint steps.

    The early peak is the highest logged value before the restricted cutoff. Returns it per run,
    the runs where it moves off the checkpoint peak, and every later logged value above it (step
    and excess).
    """
    tie = config["selectors"]["tie_rule"]
    cutoff = config["selectors"]["restricted_min_step"]
    peaks = {}
    moved = {}
    later_above = {}
    for run in runs:
        logged = selectors.replay_q_series(run, checkpoints_only=False)
        early = {step: value for step, value in logged.items() if step < cutoff}
        peak = selectors.pick(early, "max", tie)
        peaks[run.tag] = peak
        checkpoint_peak = selectors.pick(selectors.replay_q_series(run), "max", tie)
        if peak != checkpoint_peak:
            moved[run.tag] = {"checkpoint_peak": checkpoint_peak, "full_log_peak": peak}
        above = [
            {"step": step, "excess": value - logged[peak]}
            for step, value in logged.items()
            if step >= cutoff and value > logged[peak]
        ]
        if above:
            later_above[run.tag] = above
    return {
        "peaks": peaks,
        "moved": moved,
        "moved_count": len(moved),
        "later_above": later_above,
    }


def discounted_return(decisions: int, success: bool, gamma: float) -> float:
    """Return the discounted return of one episode from its first decision (App. E).

    Every decision earns -1, except the last decision of a success, which earns 0. A timeout is
    treated as terminal, which gives the highest return any treatment of the timeout allows.
    """
    paid = decisions - 1 if success else decisions
    return -(1 - gamma**paid) / (1 - gamma)


def scan_return(run, step: int, config: dict) -> float:
    """Return the mean discounted return of one checkpoint's scan episodes, per decision.

    Each decision earns one reward and takes one discount step.
    """
    query_freq = config["training"]["query_freq"]
    gamma = config["training"]["discount"] ** query_freq
    returns = [
        discounted_return(math.ceil(row["env_steps"] / query_freq), row["success"], gamma)
        for row in run.scan_rows[step]
    ]
    return statistics.fmean(returns)


def start_state_value(run, step: int) -> float:
    """Return the critic's mean value at the held-out demonstrations' start states (t = 0)."""
    values = [row["critic_value"] for row in run.heldout[step]["per_state"] if row["t"] == 0]
    return statistics.fmean(values)


def critic_vs_return(runs, config: dict) -> dict:
    """Compare the critic's value at demonstration start states with the scan episodes' return.

    The return is the mean discounted return of a checkpoint's scan episodes. The comparison
    starts at the earliest held-out Q peak step of these runs. Returns the count of checkpoints
    where the critic is above the return, the number of runs whose gap shrinks from the restricted
    cutoff to the final checkpoint, and, on the runs other than the approximate rerun, the median
    gap and median scan success at each held-out Q peak step and the median final gap.
    """
    cutoff = config["selectors"]["restricted_min_step"]
    peak_steps = heldout_q_peaks(runs, config)["peak_steps"]
    gaps = {}
    for run in runs:
        steps = [
            step for step in selectors.heldout_series(run, "heldout_q") if step >= peak_steps[0]
        ]
        gaps[run.tag] = {
            step: start_state_value(run, step) - scan_return(run, step, config) for step in steps
        }
    every_gap = [gap for run_gaps in gaps.values() for gap in run_gaps.values()]
    finals = {run.tag: selectors.final_step(run) for run in runs}
    shrinks = [tag for tag, run_gaps in gaps.items() if run_gaps[finals[tag]] < run_gaps[cutoff]]
    originals = [run for run in runs if run.rerun_of is None]
    return {
        "from_step": peak_steps[0],
        "checkpoints": len(every_gap),
        "above": sum(gap > 0 for gap in every_gap),
        "shrink_count": len(shrinks),
        "runs": len(gaps),
        "early_gap": {
            step: statistics.median(gaps[run.tag][step] for run in originals) for step in peak_steps
        },
        "early_scan": {
            step: statistics.median(run.scan[step] for run in originals) for step in peak_steps
        },
        "final_gap": statistics.median(gaps[run.tag][finals[run.tag]] for run in originals),
        "median_runs": len(originals),
    }


def overlap_with_heldout(runs: dict) -> dict:
    """Return the earlier runs that have both a training log and held-out statistics."""
    logged = {run.tag for run in load.with_stage(runs, "train_log")}
    heldout = {run.tag for run in load.with_stage(runs, "heldout", role="earlier")}
    tags = sorted(logged & heldout)
    return {"count": len(tags), "tags": tags}
