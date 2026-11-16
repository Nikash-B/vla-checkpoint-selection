"""Write the Markdown page of selection on practice success S_W."""

from ckpt_selection.markdown import (
    DASH,
    MARGIN,
    PERCENT,
    REGRET,
    RHO,
    SUCCESS_SE,
    bullets,
    interval,
    number,
    page,
    percent,
    step,
    step_table,
    table,
    tags,
    yes_no,
)


def score_name(results: dict) -> str:
    """Return the name of the practice-success score, S_ with the window."""
    return f"S_{step(results['replication']['window'])}"


def replication_section(results: dict, config: dict) -> tuple[str, str, list[str]]:
    """Return the section of selection on S_W in the replication runs."""
    replication = results["replication"]
    window = step(replication["window"])
    score = score_name(results)
    rows = [
        [
            row["run"],
            number(row["r_last"], REGRET),
            step(row["step"]),
            number(row["success"], PERCENT),
            number(row["regret"], REGRET),
            number(row["reduction"], REGRET),
            number(row["partial_rho"], RHO),
            interval(row["interval"]),
            number(row["margin"], MARGIN),
            number(row["pick_score"], PERCENT),
            number(row["pick_se"], SUCCESS_SE),
        ]
        for row in replication["runs"]
    ]
    medians = replication["medians"]
    rows.append(
        [
            "Median",
            number(medians["r_last"], REGRET),
            DASH,
            DASH,
            number(medians["regret"], REGRET),
            number(medians["reduction"], REGRET),
            number(medians["partial_rho"], RHO),
            DASH,
            DASH,
            DASH,
            DASH,
        ]
    )
    return (
        f"Selection on {score} in the replication runs",
        f"S_W at a checkpoint is the success rate of the practice episodes collected in the "
        f"W = {window} gradient steps up to it. We pick the checkpoint with the highest "
        f"{score}, ties to the {config['selectors']['tie_rule']} step. The partial ρ is "
        f"Spearman's ρ of {score} with scan success over all checkpoints, controlling for "
        "the log training step, with a 95% Fisher interval. The margin is the pick's "
        f"{score} minus the highest {score} of any other checkpoint; the SE is the binomial "
        f"standard error of the pick's {score}. Rows are sorted by R_last, highest first; "
        "the last row gives medians.",
        [
            table(
                [
                    "Run",
                    "R_last (points)",
                    "Selected step",
                    "Confirmed success (%)",
                    "Regret (points)",
                    "Reduction (points)",
                    "Partial ρ",
                    "95% interval",
                    "Margin (points)",
                    f"{score} at pick (%)",
                    "SE (points)",
                ],
                rows,
            ),
            bullets(
                [
                    f"Replication runs: {results['run_count']}",
                    f"SE of the pick's {score}: "
                    f"{number(replication['se_range'][0], SUCCESS_SE)}-"
                    f"{number(replication['se_range'][1], SUCCESS_SE)} points",
                    f"Every margin is below its SE: {yes_no(replication['all_margins_below_se'])}",
                ]
            ),
        ],
    )


def ranking_section(results: dict, config: dict) -> tuple[str, str, list[str]]:
    """Return the section of the ranking criterion."""
    score = score_name(results)
    criteria = config["practice_criteria"]
    medians = results["replication"]["medians"]
    ranking = results["ranking"]
    return (
        "Ranking criterion",
        f"The criterion asks for a median partial ρ of at least "
        f"{criteria['min_median_partial_rho']} and no selected checkpoint more than "
        f"{criteria['max_selected_regret']} points below its run's best confirmed checkpoint. "
        f"The stricter criterion asks the median uncontrolled ρ of {score} to exceed the "
        f"training step's by at least {criteria['stricter_margin_over_step']}.",
        [
            bullets(
                [
                    f"Median partial ρ: {number(medians['partial_rho'], RHO)}",
                    "Largest regret of a selected checkpoint: "
                    f"{number(ranking['max_selected_regret'], REGRET)} points",
                    f"Criterion met: {yes_no(ranking['criterion_met'])}",
                    f"Runs with a positive partial ρ: {ranking['positive_runs']} of "
                    f"{ranking['run_count']}",
                    "Runs whose partial interval contains zero: "
                    f"{ranking['intervals_including_zero']} of {ranking['run_count']}",
                    "Median uncontrolled ρ minus the training step's: "
                    f"{number(ranking['excess_over_step'], RHO)}",
                    f"Stricter criterion met: {yes_no(ranking['stricter_met'])}",
                ]
            )
        ],
    )


def moved_section(results: dict, config: dict) -> tuple[str, str, list[str]]:
    """Return the section of the runs on which selection moves regret."""
    floor = config["protocol"]["regret_floor"]
    criteria = config["practice_criteria"]
    selection = results["selection"]
    moved_rows = [
        [
            row["run"],
            number(row["r_last"], REGRET),
            number(row["regret"], REGRET),
            number(row["reduction"], REGRET),
            percent(row["share_removed"]),
        ]
        for row in selection["moved"]
    ]
    return (
        "Runs on which selection moves regret",
        "Selection moves regret on a run when it changes regret by more than the "
        f"{floor}-point floor. The regret test applies only if at least "
        f"{criteria['min_runs_with_large_regret']} runs have R_last of at least the floor. "
        "Rows are sorted by R_last, highest first.",
        [
            table(
                [
                    "Run",
                    "R_last (points)",
                    "Regret (points)",
                    "Reduction (points)",
                    "Share of R_last removed (%)",
                ],
                moved_rows,
            ),
            bullets(
                [
                    f"Runs on which selection moves regret: {selection['moved_count']}",
                    f"Runs with R_last of at least the floor: "
                    f"{selection['large_r_last_count']} "
                    f"({tags(selection['large_r_last_runs'])})",
                    f"Regret test applies: {yes_no(selection['regret_test_applies'])}",
                ]
            ),
        ],
    )


def moved_run_sections(results: dict) -> list[tuple[str, str, list[str]]]:
    """Return one section per run on which selection moves regret, with every checkpoint."""
    score = score_name(results)
    sections = []
    for row in results["selection"]["moved"]:
        series = results["moved_series"][row["run"]]
        quantities = (
            (f"{score} (%)", "practice_success"),
            (f"SE of {score} (points)", "practice_se"),
            ("Scan success (%)", "scan"),
            ("Confirmed success (%)", "confirmed"),
        )
        series_rows = [
            (
                name,
                {
                    checkpoint: values[key]
                    for checkpoint, values in series.items()
                    if values[key] is not None
                },
            )
            for name, key in quantities
        ]
        sections.append(
            (
                f"{score}, scan and confirmed success on {row['run']}",
                "Every checkpoint of a run on which selection moves regret. Rows are quantities; "
                "columns are checkpoints in step order. A dash means that the checkpoint has no "
                "value. These sections follow the row order of the table of runs on which "
                "selection moves regret.",
                [
                    step_table("Quantity", series_rows, percent),
                    bullets(
                        [
                            f"Selected step: {step(row['step'])}",
                            f"Final step: {step(max(series))}",
                            f"R_last: {number(row['r_last'], REGRET)} points; regret of the "
                            f"selected checkpoint: {number(row['regret'], REGRET)} points",
                        ]
                    ),
                ],
            )
        )
    return sections


def below_floor_section(results: dict, config: dict) -> tuple[str, str, list[str]]:
    """Return the section of the runs with R_last below the floor."""
    score = score_name(results)
    criteria = config["practice_criteria"]
    expectation = results["expectation"]
    failure_rows = [
        [
            row["run"],
            step(row["step"]),
            number(row["regret"], REGRET),
            number(row["r_last"], REGRET),
            number(row["pick_score"], PERCENT),
            number(row["pick_se"], SUCCESS_SE),
            number(row["final_score"], PERCENT),
            number(row["final_se"], SUCCESS_SE),
        ]
        for row in expectation["failures"]
    ]
    return (
        "Runs with R_last below the floor",
        "On these runs we expect the selected checkpoint's regret to stay within "
        f"{criteria['expected_regret_margin']} points of the final checkpoint's. The table "
        f"lists the runs where it does not, with {score} and its SE at the selected and the "
        "final checkpoint. Rows are sorted by R_last, highest first.",
        [
            table(
                [
                    "Run",
                    "Selected step",
                    "Regret (points)",
                    "R_last (points)",
                    f"{score} at pick (%)",
                    "SE at pick (points)",
                    f"{score} at final (%)",
                    "SE at final (points)",
                ],
                failure_rows,
            ),
            bullets(
                [
                    f"Runs with R_last below the floor: {expectation['small_r_last_count']}; "
                    f"outside the expectation: {expectation['failure_count']}",
                    "Runs whose final checkpoint is the best confirmed: "
                    f"{expectation['final_is_best_count']} of {expectation['run_count']}",
                    f"Runs whose selected checkpoint is at most "
                    f"{criteria['max_selected_regret']} points below the best: "
                    f"{expectation['condition_runs']} of {expectation['run_count']}",
                ]
            ),
        ],
    )


def uncontrolled_section(results: dict) -> tuple[str, str, list[str]]:
    """Return the section of the correlations without the step control."""
    score = score_name(results)
    uncontrolled = results["uncontrolled"]
    uncontrolled_rows = [
        [tag, number(row["score"], RHO), number(row["step"], RHO), number(row["difference"], RHO)]
        for tag, row in uncontrolled["per_run"].items()
    ]
    return (
        "Correlations without the step control",
        f"Spearman's ρ of {score} and of the training step with scan success, over all "
        "checkpoints. Rows follow the config's run order.",
        [
            table(
                ["Run", f"ρ({score}, scan)", "ρ(step, scan)", "Difference"],
                uncontrolled_rows,
            ),
            bullets(
                [
                    f"Median ρ({score}, scan): {number(uncontrolled['median_score'], RHO)}",
                    f"Median ρ(step, scan): {number(uncontrolled['median_step'], RHO)}",
                    "Largest absolute difference on a run: "
                    f"{number(uncontrolled['max_abs_difference'], RHO)}",
                ]
            ),
        ],
    )


def windows_section(results: dict) -> tuple[str, str, list[str]]:
    """Return the section of selection with other windows."""
    replication = results["replication"]
    window = step(replication["window"])
    windows = results["windows"]
    window_rows = [
        [
            name,
            number(entry["median_partial"], RHO),
            f"{entry['changed_count']} ({tags(entry['changed'])})",
            f"{entry['outside_confirm_count']} ({tags(entry['outside_confirm'])})",
        ]
        for name, entry in windows.items()
    ]
    window_pick_rows = [
        [
            row["run"],
            step(row["step"]),
            *(step(windows[name]["picks"][row["run"]]) for name in windows),
        ]
        for row in replication["runs"]
    ]
    return (
        "Other windows",
        f"The same selection with other windows. A pick changes when it differs from the "
        f"{window} pick. Rows of the first table follow the config's sensitivity windows. "
        "The second table gives every pick, with rows sorted by R_last as above.",
        [
            table(
                [
                    "Window",
                    "Median partial ρ",
                    "Changed picks",
                    "Changed picks outside the confirm set",
                ],
                window_rows,
            ),
            table(["Run", window, *windows], window_pick_rows),
        ],
    )


def window_choice_sections(results: dict) -> list[tuple[str, str, list[str]]]:
    """Return one section per run of the window choice."""
    sections = []
    for tag, choice in results["window_choice"].items():
        sections.append(
            (
                f"Window choice on {tag}",
                "The training log of this run records one practice outcome per log interval. "
                "Each candidate window is a number of outcomes times the median interval between "
                f"outcomes, {step(choice['spacing'])}. We score each window by Spearman's ρ of "
                f"S_W with confirmed success over the run's {choice['confirmed_count']} confirmed "
                "checkpoints. Rows are sorted by window.",
                [
                    table(
                        ["Outcomes", "Window", "ρ with confirmed success"],
                        [
                            [
                                str(outcomes),
                                step(int(outcomes * choice["spacing"])),
                                number(rho, RHO),
                            ]
                            for outcomes, rho in sorted(choice["correlations"].items())
                        ],
                    ),
                    bullets(
                        [
                            f"Best window: {step(choice['best_window'])} "
                            f"({choice['best_outcomes']} outcomes)"
                        ]
                    ),
                ],
            )
        )
    return sections


def before_competence_section(results: dict) -> tuple[str, str, list[str]]:
    """Return the section of the checkpoints before competence."""
    return (
        "Checkpoints before competence",
        "A checkpoint is competent when its scan success is above the base policy's. The "
        "table counts the checkpoints before the first competent one. On every run, at least "
        f"the first {results['before_competence_min']} checkpoints come before competence. "
        "Rows follow the config's run order.",
        [
            table(
                ["Run", "Checkpoints before competence"],
                [[tag, str(count)] for tag, count in results["before_competence"].items()],
            )
        ],
    )


def practice_page(results: dict, config: dict) -> str:
    """Return the page of selection on practice success S_W."""
    sections = [
        replication_section(results, config),
        ranking_section(results, config),
        moved_section(results, config),
        *moved_run_sections(results),
        below_floor_section(results, config),
        uncontrolled_section(results),
        windows_section(results),
        *window_choice_sections(results),
        before_competence_section(results),
    ]
    return page(
        "Selection on practice success",
        "The practice-success selector reads the outcomes of the episodes that training collects "
        "anyway, so it needs no extra evaluation episodes.",
        sections,
    )
