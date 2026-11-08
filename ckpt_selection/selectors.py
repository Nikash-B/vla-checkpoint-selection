"""Rollout-free selectors and the per-checkpoint signals they read.

A selector chooses one checkpoint of a run without the scan or confirm episodes (Sec. 3). The five
rollout-free selectors of Sec. 3 read the held-out statistics (highest held-out Q, the restricted
critic, lowest action entropy, the ensemble-spread veto) or the practice outcomes (S_W). This
module also gives the signals that App. C, D and E rank checkpoints by: the L2, KL and MMD
distance rules, SoftOPC and its label-free control, replay-batch Q, and the training step itself.

Candidate checkpoints are the scanned checkpoints of a run (Algorithm 1 scans every checkpoint);
the base policy (step -1) is never a candidate. Every selector and signal takes a load.RunData;
pick, practice_success and negated work on plain series.
"""

import math

import numpy as np

from ckpt_selection import layout, stats, training_log

TIE_RULES = ("earliest", "latest")
MODES = ("max", "min")
STEP_SIGNAL = "training step"  # the signal that ranks checkpoints by their training step
# Every partial correlation controls for one variable, the log training step.
STEP_CONTROLS = 1


def pick(values: dict[int, float], mode: str, tie: str) -> int:
    """Return the step with the highest (mode 'max') or lowest ('min') value.

    Ties go to the earliest or the latest of the tied steps, as `tie` says.
    """
    if not values:
        raise ValueError("no checkpoint to pick from")
    if mode not in MODES or tie not in TIE_RULES:
        raise ValueError(f"mode must be one of {MODES} and tie one of {TIE_RULES}")
    best = max(values.values()) if mode == "max" else min(values.values())
    tied = [step for step, value in values.items() if value == best]
    return min(tied) if tie == "earliest" else max(tied)


def candidate_steps(run) -> list[int]:
    """Return the run's checkpoint steps in order: every scanned step except the base policy."""
    return sorted(step for step in run.scan if step != layout.BASE_STEP)


def confirmed_steps(run) -> list[int]:
    """Return the run's confirmed checkpoint steps in order, base policy excluded."""
    return sorted(step for step in run.confirm if step != layout.BASE_STEP)


def final_step(run) -> int:
    """Return the step of the run's final checkpoint."""
    return candidate_steps(run)[-1]


def scan_argmax(run, tie: str) -> int:
    """Return the checkpoint with the highest scan success, the reference that needs episodes."""
    return pick({step: run.scan[step] for step in candidate_steps(run)}, "max", tie)


def heldout_series(run, key: str, min_step: int = 0) -> dict[int, float]:
    """Return one held-out statistic per checkpoint at or after min_step, base policy excluded.

    key is a top-level field of heldout_<step>.json: l2, action_entropy, kl, mmd, heldout_q or
    ensemble_spread (App. C "Definitions").
    """
    if not run.heldout:
        raise ValueError(f"run {run.tag} has no held-out statistics")
    return {
        step: run.heldout[step][key]
        for step in sorted(run.heldout)
        if step != layout.BASE_STEP and step >= min_step
    }


def highest_heldout_q(run, tie: str) -> int:
    """Return the checkpoint with the highest held-out Q (Sec. 3, first selector)."""
    return pick(heldout_series(run, "heldout_q"), "max", tie)


def restricted_critic(run, min_step: int, tie: str) -> int:
    """Return the highest held-out Q among checkpoints at min_step or later (Sec. 3, second)."""
    return pick(heldout_series(run, "heldout_q", min_step), "max", tie)


def lowest_entropy(run, tie: str) -> int:
    """Return the checkpoint with the lowest action entropy (Sec. 3, third selector)."""
    return pick(heldout_series(run, "action_entropy"), "min", tie)


def spread_veto(run, mads: float) -> int:
    """Return the last checkpoint that the ensemble-spread veto keeps (Sec. 3, fourth selector).

    The veto removes every checkpoint whose spread exceeds the run's median spread by more than
    `mads` raw median absolute deviations.
    """
    return stats.spread_veto_pick(heldout_series(run, "ensemble_spread"), mads)


def distance_pick(run, statistic: str, direction: str, min_step: int, tie: str) -> int:
    """Return one L2, KL or MMD rule's pick: the highest or lowest value from min_step on.

    App. C applies each statistic in both directions from each cutoff in config [selectors]
    distance_min_steps.
    """
    return pick(heldout_series(run, statistic, min_step), direction, tie)


def practice_success(outcomes: list, step: int, window: int) -> tuple[float, int]:
    """Return S_W at one checkpoint and the number of outcomes it averages (App. G).

    S_W(c) is the mean outcome of the practice episodes collected at gradient steps in (c - W, c].
    The score is NaN when the window holds no outcome.
    """
    window_outcomes = [
        success for grad_step, success in outcomes if step - window < grad_step <= step
    ]
    if not window_outcomes:
        return math.nan, 0
    return float(np.mean(window_outcomes)), len(window_outcomes)


def practice_scores(run, window: int) -> dict[int, tuple[float, int]]:
    """Return S_W and its outcome count at every checkpoint of the run."""
    if not run.practice:
        raise ValueError(f"run {run.tag} has no practice outcomes")
    return {step: practice_success(run.practice, step, window) for step in candidate_steps(run)}


def empty_practice_windows(run, window: int) -> list[int]:
    """Return the checkpoints whose S_W window holds no practice outcome."""
    return [step for step, (_, count) in practice_scores(run, window).items() if count == 0]


def eligible_practice_scores(run, window: int) -> dict[int, float]:
    """Return S_W at every checkpoint whose window holds at least one practice outcome."""
    return {step: score for step, (score, count) in practice_scores(run, window).items() if count}


def practice_pick(run, window: int, tie: str) -> int:
    """Return the checkpoint with the highest S_W (Sec. 3, fifth selector).

    Every checkpoint whose window holds at least one practice outcome is eligible. When the
    training log holds one outcome per log interval, the windows of the earliest checkpoints can be
    empty, and those checkpoints cannot be picked; softopc.md lists them. softopc.signal_table
    fails if such a checkpoint is confirmed, so the correlations never skip a confirmed checkpoint.
    """
    return pick(eligible_practice_scores(run, window), "max", tie)


def replay_q_series(run, checkpoints_only: bool = True) -> dict[int, float]:
    """Return replay-batch Q (the training log's mean critic value) by gradient step (App. E).

    With checkpoints_only, keep the logged steps that are also checkpoint steps; otherwise keep
    every logged step.
    """
    if not run.train_log:
        raise ValueError(f"run {run.tag} has no training log")
    logged = {row["_step"]: row[training_log.CRITIC_KEY] for row in run.train_log}
    if not checkpoints_only:
        return dict(sorted(logged.items()))
    return {step: logged[step] for step in candidate_steps(run) if step in logged}


def softopc_series(run, key: str = "softopc") -> dict[int, float]:
    """Return SoftOPC ('softopc') or the label-free mean Q ('label_free_mean_q') by step.

    Both come from softopc/scores.json (App. D).
    """
    if not run.softopc_scores:
        raise ValueError(f"run {run.tag} has no SoftOPC scores")
    return {step: run.softopc_scores[step][key] for step in sorted(run.softopc_scores)}


def negated(series: dict[int, float]) -> dict[int, float]:
    """Return the series with every value negated, so that higher is better."""
    return {step: -value for step, value in series.items()}


def ranking_signals(run, window: int) -> dict[str, dict[int, float]]:
    """Return every ranking signal by checkpoint, oriented so that higher is better.

    The order is S_W, SoftOPC, label-free mean Q, held-out Q, and the negated ensemble spread,
    action entropy, L2, KL and MMD, then the training step, the control that ranks checkpoints as
    keeping the final checkpoint does. Steps without a value are left out.
    """
    return {
        f"practice success, S_{layout.format_step(window)}": eligible_practice_scores(run, window),
        "SoftOPC score": softopc_series(run, "softopc"),
        "label-free mean Q": softopc_series(run, "label_free_mean_q"),
        "held-out Q": heldout_series(run, "heldout_q"),
        "negative ensemble spread": negated(heldout_series(run, "ensemble_spread")),
        "negative action entropy": negated(heldout_series(run, "action_entropy")),
        "negative L2": negated(heldout_series(run, "l2")),
        "negative KL": negated(heldout_series(run, "kl")),
        "negative MMD": negated(heldout_series(run, "mmd")),
        STEP_SIGNAL: {step: float(step) for step in candidate_steps(run)},
    }


def rollout_free_picks(run, config: dict) -> dict[str, int]:
    """Return the pick of every rollout-free selector that applies to the run (Sec. 3).

    Keys are the reason names of confirm_set.json. The held-out selectors apply to runs whose
    config lists the held-out stage; S_W applies to runs with a practice source.
    """
    tie = config["selectors"]["tie_rule"]
    picks = {}
    if "heldout" in run.entry["stages"]:
        picks["heldout_q"] = highest_heldout_q(run, tie)
        picks["restricted_critic"] = restricted_critic(
            run, config["selectors"]["restricted_min_step"], tie
        )
        picks["entropy"] = lowest_entropy(run, tie)
        picks["spread_veto"] = spread_veto(run, config["selectors"]["spread_veto_mads"])
    if run.entry["practice_source"] is not None:
        picks["practice_success"] = practice_pick(run, config["selectors"]["practice_window"], tie)
    return picks
