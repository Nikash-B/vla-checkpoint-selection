"""Bias of scan success against confirmed success on the same checkpoint, reported in App. F.

The scan and confirm pools draw different episodes, so on a confirmed checkpoint scan success minus
confirmed success measures how far the scan overstates the checkpoint. App. F groups the confirmed
checkpoints by why they entered the confirm set: the highest scan success (under either tie rule),
which is selected on the scan, and the final checkpoint, which enters whatever its scan success.
App. B cites the sign of the first group. Every value returned here is in points.
"""

import math

import numpy as np

from ckpt_selection import regret, selectors, stats


def scan_minus_confirm(run, step: int) -> float:
    """Return scan success minus confirmed success of one confirmed checkpoint, in points."""
    if step not in run.confirm:
        raise ValueError(f"run {run.tag}: checkpoint {step} has no confirm file")
    return regret.points(run.scan[step] - run.confirm[step])


def mean_and_se(per_run: dict) -> dict:
    """Return the mean and its standard error (sample SD over sqrt(n)) of per-run values."""
    values = np.array(list(per_run.values()), dtype=float)
    return {
        "mean": float(values.mean()),
        "se": float(values.std(ddof=1) / math.sqrt(len(values))),
        "n_runs": len(values),
        "per_run": per_run,
    }


def checkpoint_pair(run, step: int) -> dict:
    """Return the step, scan success and confirmed success (fractions) of one checkpoint."""
    return {"tag": run.tag, "step": step, "scan": run.scan[step], "confirm": run.confirm[step]}


def scan_bias_summary(runs, config: dict) -> dict:
    """Return the scan bias by group, with the runs where the final checkpoint scans highest.

    The runs where the final checkpoint ties for or alone holds the highest scan success are
    listed apart. The result also holds the standard error of a scan rate at the reference rate.
    """
    groups = {}
    for tie in selectors.TIE_RULES:
        groups[f"argmax_{tie}"] = {
            run.tag: scan_minus_confirm(run, selectors.scan_argmax(run, tie)) for run in runs
        }
    groups["final"] = {run.tag: scan_minus_confirm(run, selectors.final_step(run)) for run in runs}
    summary = {group: mean_and_se(per_run) for group, per_run in groups.items()}

    tie_runs = []
    unique_final_runs = []
    for run in runs:
        final = selectors.final_step(run)
        earliest = selectors.scan_argmax(run, "earliest")
        latest = selectors.scan_argmax(run, "latest")
        if latest == final and earliest != final:
            tie_runs.append(checkpoint_pair(run, final))
        elif earliest == final:
            unique_final_runs.append(checkpoint_pair(run, final))
    summary["final_tied_runs"] = tie_runs
    summary["final_unique_runs"] = unique_final_runs

    scan = config["evaluation"]["scan"]["episodes"]
    reference = config["scan_bias"]["reference_rate"]
    summary["scan_se_at_reference"] = 100 * stats.binomial_se(reference, scan)
    return summary
