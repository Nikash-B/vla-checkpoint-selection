"""Write the Markdown page of the base-policy task screen and the drop screen."""

from ckpt_selection.markdown import (
    DASH,
    SUCCESS_CHANGE,
    bullets,
    config_percent,
    number,
    page,
    percent,
    step,
    table,
    yes_no,
)

SUITE_NAMES = {"libero_10": "LIBERO-Long", "libero_90": "LIBERO-90"}


def tasks_section(results: dict) -> tuple[str, str, list[str]]:
    """Return the section of the tasks with a run."""
    rows = []
    for row in results["tasks"]:
        low = percent(row["success_min"])
        high = percent(row["success_max"])
        rows.append(
            [
                SUITE_NAMES[row["suite"]],
                row["tasks"],
                low if low == high else f"{low}-{high}",
                yes_no(row["screened_for_drops"]),
                row["role"] or DASH,
            ]
        )
    return (
        "Tasks with a run",
        "Rows follow the run order of config/experiments.toml. The last row of a suite whose "
        "screen covers all its tasks gives the range over its other tasks.",
        [
            table(
                ["Suite", "Task", "Base-policy success (%)", "Screened for drops", "Role"],
                rows,
            ),
            bullets(
                [
                    f"Task IDs inside the band, {SUITE_NAMES[suite]}: "
                    + (", ".join(str(task) for task in found) or DASH)
                    for suite, found in results["qualifying"].items()
                ]
            ),
        ],
    )


def two_stage_sections(results: dict) -> list[tuple[str, str, list[str]]]:
    """Return one section per suite with a two-stage screen."""
    sections = []
    for suite, recheck in results["two_stage_recheck"].items():
        stay = recheck["stage2_stay_in_band"]
        sections.append(
            (
                f"Two-stage screen of {SUITE_NAMES[suite]}",
                f"Stage 1 ran {recheck['stage1_tasks']} tasks on "
                f"{recheck['stage1_episodes']} episodes each. Stage 2 reran the tasks inside the "
                f"band on {recheck['stage2_episodes']} new episodes. The table lists the tasks "
                "that stay inside the band at stage 2, sorted by stage-2 success, lowest first.",
                [
                    table(
                        ["Task", "Stage-2 success (%)"],
                        [[str(task), percent(rate)] for task, rate in stay.items()],
                    ),
                    bullets(
                        [
                            f"Tasks inside the band at stage 1: {recheck['stage1_in_band_count']} "
                            f"({', '.join(str(task) for task in recheck['stage1_in_band'])})",
                            "Tasks that stay inside the band at stage 2: "
                            f"{recheck['stage2_stay_in_band_count']}",
                        ]
                    ),
                ],
            )
        )
    return sections


def drop_screen_section(results: dict, config: dict) -> tuple[str, str, list[str]]:
    """Return the section of the drop screen."""
    drop = results["drop_screen"]
    return (
        "Drop screen",
        "Each drop-screen run trains for the shorter schedule in the config and scans every "
        "checkpoint. A drop is a fall in scan success between consecutive checkpoints that "
        "the next checkpoint rises from again. A task is flagged when its largest drop is "
        f"more than {config['protocol']['drop_points']} points. Rows follow the config's run "
        "order.",
        [
            table(
                ["Run", "Task", "Largest drop (points)", "From step", "To step", "Flagged"],
                [
                    [
                        tag,
                        str(row["task"]),
                        number(row["drop"], SUCCESS_CHANGE),
                        step(row.get("from_step")),
                        step(row.get("to_step")),
                        yes_no(row["flagged"]),
                    ]
                    for tag, row in drop["per_run"].items()
                ],
            ),
            bullets(
                [
                    f"Drop-screen runs: {drop['run_count']}",
                    "Flagged tasks: "
                    + (", ".join(str(task) for task in drop["flagged_tasks"]) or DASH),
                ]
            ),
        ],
    )


def screen_page(results: dict, config: dict) -> str:
    """Return the page of the base-policy task screen and the drop screen."""
    band = results["band"]
    band_text = f"{config_percent(band[0])}-{config_percent(band[1])}%"
    sections = [
        tasks_section(results),
        *two_stage_sections(results),
        drop_screen_section(results, config),
    ]
    return page(
        "Base-policy task screen",
        "We ran the base policy on every task of each screened suite and marked the tasks whose "
        f"success lies in the {band_text} band.",
        sections,
    )
