"""SoftOPC, its label-free control, and every signal's correlation beyond the training step.

The paper reports these results in Sec. 4.1 and App. D. The SoftOPC score and the label-free mean
Q of each checkpoint come from softopc/scores.json; the other signals come from
selectors.ranking_signals. Correlations run over the confirmed checkpoints of one run, with 95%
Fisher intervals; the base policy (step -1) is never a checkpoint.
"""

import itertools
import math

from ckpt_selection import selectors, stats

# Keys of one signal row, in the order the results print them.
SIGNAL = "signal"
RHO = "rho_success"
RHO_INTERVAL = "rho_interval"
RHO_STEP = "rho_log_step"
PARTIAL = "partial_rho"
PARTIAL_INTERVAL = "partial_interval"


def pool_balance(run, config: dict) -> dict:
    """Return the pool's successes and failures, and whether both reach the minimum (App. D)."""
    minimum = config["softopc"]["min_per_class"]
    successes = run.pool_meta["successes"]
    failures = run.pool_meta["failures"]
    return {
        "episodes": successes + failures,
        "successes": successes,
        "failures": failures,
        "min_per_class": minimum,
        "eligible": successes >= minimum and failures >= minimum,
    }


def signal_row(name: str, values: list, success: list, log_step: list) -> dict:
    """Return the correlations of one signal with confirmed success and with the training step.

    The row holds ρ with success and its interval, ρ with log training step, and the partial ρ
    with success given log training step and its interval. The training step's own row has no
    partial correlation, since it controls for itself.
    """
    n_points = len(values)
    rho = stats.spearman(values, success)
    row = {
        SIGNAL: name,
        RHO: rho,
        RHO_INTERVAL: stats.fisher_interval(rho, n_points),
        RHO_STEP: stats.spearman(values, log_step),
        PARTIAL: None,
        PARTIAL_INTERVAL: None,
    }
    if name != selectors.STEP_SIGNAL:
        partial = stats.partial_spearman(values, success, log_step)
        row[PARTIAL] = partial
        row[PARTIAL_INTERVAL] = stats.fisher_interval(
            partial, n_points, controls=selectors.STEP_CONTROLS
        )
    return row


def signal_table(run, config: dict) -> list[dict]:
    """Return the correlations of every ranking signal over the run's confirmed checkpoints.

    Every signal is oriented so that higher is better (selectors.ranking_signals). A signal that
    lacks a value at a confirmed checkpoint raises an error rather than shrinking the table.
    """
    steps = selectors.confirmed_steps(run)
    success = [run.confirm[step] for step in steps]
    log_step = [math.log(step) for step in steps]
    signals = selectors.ranking_signals(run, config["selectors"]["practice_window"])
    rows = []
    for name, series in signals.items():
        missing = [step for step in steps if step not in series]
        if missing:
            raise KeyError(f"signal {name!r} has no value at confirmed steps {missing}")
        rows.append(signal_row(name, [series[step] for step in steps], success, log_step))
    return rows


def softopc_pick(run) -> dict:
    """Return where SoftOPC's highest score over all scored checkpoints falls (Sec. 4.1, App. D).

    Reports every step tied at the highest score, whether that is the final checkpoint alone,
    and whether the final checkpoint ties for the best confirmed success.
    """
    scores = selectors.softopc_series(run)
    highest = max(scores.values())
    top_steps = [step for step, score in scores.items() if score == highest]
    final = selectors.final_step(run)
    best_confirmed = max(run.confirm[step] for step in selectors.confirmed_steps(run))
    return {
        "top_steps": top_steps,
        "is_final": top_steps == [final],
        "final_step": final,
        "final_confirmed": run.confirm[final],
        "best_confirmed": best_confirmed,
        "final_ties_best": run.confirm[final] == best_confirmed,
        "n_scored": len(scores),
    }


def table_checks(rows: list[dict]) -> dict:
    """Return the App. D checks on the signal correlations.

    Checks: every partial interval contains zero; every ρ interval that excludes zero contains
    the training step's ρ. Also the signals with the highest ρ with success and the highest
    partial ρ, training step excluded.
    """
    step_row = next(row for row in rows if row[SIGNAL] == selectors.STEP_SIGNAL)
    step_rho = step_row[RHO]
    others = [row for row in rows if row[SIGNAL] != selectors.STEP_SIGNAL]
    excluding_zero = [
        row for row in others if not row[RHO_INTERVAL][0] <= 0 <= row[RHO_INTERVAL][1]
    ]
    return {
        "step_rho": step_rho,
        "all_partials_include_zero": all(
            row[PARTIAL_INTERVAL][0] <= 0 <= row[PARTIAL_INTERVAL][1] for row in others
        ),
        "signals_excluding_zero": [row[SIGNAL] for row in excluding_zero],
        "excluding_zero_contain_step_rho": all(
            row[RHO_INTERVAL][0] <= step_rho <= row[RHO_INTERVAL][1] for row in excluding_zero
        ),
        "highest_rho": max(others, key=lambda row: row[RHO])[SIGNAL],
        "highest_partial": max(others, key=lambda row: row[PARTIAL])[SIGNAL],
    }


def checkpoint_scores(run) -> dict[int, dict]:
    """Return SoftOPC, label-free mean Q and confirmed success (None if unconfirmed) by step."""
    scores = selectors.softopc_series(run)
    label_free = selectors.softopc_series(run, "label_free_mean_q")
    return {
        step: {
            "softopc": scores[step],
            "label_free_mean_q": label_free[step],
            "confirmed": run.confirm.get(step),
        }
        for step in scores
    }


def confirmed_falls_with_rising_score(run) -> list[int]:
    """Return the confirmed falls at which the SoftOPC score rises.

    Each entry is the later step of a pair of consecutive confirmed checkpoints where confirmed
    success falls but the SoftOPC score rises.
    """
    scores = selectors.softopc_series(run)
    steps = selectors.confirmed_steps(run)
    return [
        later
        for earlier, later in itertools.pairwise(steps)
        if run.confirm[later] < run.confirm[earlier] and scores[later] > scores[earlier]
    ]
