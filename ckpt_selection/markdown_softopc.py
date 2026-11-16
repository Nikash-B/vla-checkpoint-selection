"""Write the Markdown page of SoftOPC and the other ranking signals."""

from ckpt_selection.markdown import (
    DASH,
    Q_VALUE,
    RHO,
    SOFTOPC_SCORE,
    bullets,
    interval,
    number,
    page,
    percent,
    step,
    steps,
    table,
    yes_no,
)


def pools_section(results: dict, config: dict) -> tuple[str, str, list[str]]:
    """Return the section of the labeled pools."""
    minimum = config["softopc"]["min_per_class"]
    pool_rows = [
        [
            tag,
            str(pool["episodes"]),
            str(pool["successes"]),
            str(pool["failures"]),
            yes_no(pool["eligible"]),
        ]
        for tag, pool in results["pools"].items()
    ]
    return (
        "Labeled pools",
        f"SoftOPC needs a labeled pool with at least {minimum} successes and {minimum} "
        "failures. Rows follow the config's run order.",
        [table(["Run", "Episodes", "Successes", "Failures", "Eligible"], pool_rows)],
    )


def empty_practice_note(results: dict, config: dict) -> str:
    """Return the note on the checkpoints where S_W has no value, or an empty string."""
    window = step(config["selectors"]["practice_window"])
    empty = results["practice_empty_steps"]
    if not empty:
        return ""
    return (
        f" S_{window} has no value at these checkpoints because no practice outcome falls in "
        f"their window: {steps(empty)}. They are not confirmed, so every row uses every "
        "confirmed checkpoint."
    )


def signals_section(results: dict, config: dict) -> tuple[str, str, list[str]]:
    """Return the section of the signals against confirmed success."""
    signal_rows = [
        [
            row["signal"],
            number(row["rho_success"], RHO),
            interval(row["rho_interval"]),
            number(row["rho_log_step"], RHO),
            number(row["partial_rho"], RHO),
            interval(row["partial_interval"]),
        ]
        for row in results["signals"]
    ]
    checks = results["checks"]
    run = results["run"]
    return (
        f"Signals against confirmed success on {run}",
        f"Each signal ranks the {results['confirmed_count']} confirmed checkpoints of {run}, "
        "oriented so that higher is better. ρ is Spearman's correlation. The partial ρ "
        "controls for the log training step. Intervals are 95% Fisher intervals. Rows follow "
        "the signal order of selectors.ranking_signals, with the training step last."
        + empty_practice_note(results, config),
        [
            table(
                [
                    "Signal",
                    "ρ with success",
                    "95% interval",
                    "ρ with log step",
                    "Partial ρ",
                    "95% interval (partial)",
                ],
                signal_rows,
            ),
            bullets(
                [
                    f"ρ of the training step with success: {number(checks['step_rho'], RHO)}",
                    "Every partial interval contains zero: "
                    f"{yes_no(checks['all_partials_include_zero'])}",
                    "Signals whose ρ interval excludes zero: "
                    f"{'; '.join(checks['signals_excluding_zero']) or DASH}",
                    "Each of these intervals contains the training step's ρ: "
                    f"{yes_no(checks['excluding_zero_contain_step_rho'])}",
                    f"Highest ρ with success, training step excluded: {checks['highest_rho']}",
                    f"Highest partial ρ: {checks['highest_partial']}",
                ]
            ),
        ],
    )


def scores_section(results: dict) -> tuple[str, str, list[str]]:
    """Return the section of the SoftOPC score at every scored checkpoint."""
    score_rows = [
        [
            step(score_step),
            percent(score["confirmed"]),
            number(score["softopc"], SOFTOPC_SCORE),
            number(score["label_free_mean_q"], Q_VALUE),
        ]
        for score_step, score in results["scores"].items()
    ]
    pick = results["pick"]
    return (
        f"SoftOPC score at every scored checkpoint of {results['run']}",
        "Rows are sorted by step. A dash means that the checkpoint was not confirmed.",
        [
            table(
                [
                    "Step",
                    "Confirmed success (%)",
                    "SoftOPC score (reward units)",
                    "Label-free mean Q (reward units)",
                ],
                score_rows,
            ),
            bullets(
                [
                    f"Scored checkpoints: {pick['n_scored']}",
                    f"Highest SoftOPC score at: {steps(pick['top_steps'])}; the final "
                    f"checkpoint alone: {yes_no(pick['is_final'])}",
                    f"Confirmed success of the final checkpoint: "
                    f"{percent(pick['final_confirmed'])}%; ties the best confirmed success: "
                    f"{yes_no(pick['final_ties_best'])}",
                    "Confirmed falls between consecutive confirmed checkpoints where the "
                    f"SoftOPC score rises: {results['falls_with_rising_score_count']} (at "
                    f"{steps(results['falls_with_rising_score'])})",
                ]
            ),
        ],
    )


def softopc_page(results: dict, config: dict) -> str:
    """Return the page of SoftOPC and the other ranking signals."""
    sections = [
        pools_section(results, config),
        signals_section(results, config),
        scores_section(results),
    ]
    return page(
        "SoftOPC and the other ranking signals",
        "SoftOPC scores each checkpoint's critic on a labeled pool of successes and failures. We "
        "compare it, its label-free control and the other signals with confirmed success.",
        sections,
    )
