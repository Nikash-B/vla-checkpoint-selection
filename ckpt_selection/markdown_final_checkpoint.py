"""Write the sections of the regret page on keeping the final checkpoint."""

from ckpt_selection import selectors
from ckpt_selection.markdown import (
    DASH,
    ORACLE_BIAS,
    REGRET,
    SUCCESS_CHANGE,
    bullets,
    config_percent,
    number,
    percent,
    step,
    step_table,
    table,
    tags,
    yes_no,
)

MEDIAN_ROWS = {
    "earlier": "Median, earlier runs",
    "task8": "Median, task-8 runs",
    "earlier_and_control": "Median, earlier and control runs",
}
BIAS_MODELS = {
    "independent": "Observed rates",
    "shrunk": "Near-ties shrunk",
    "all_tied": "All tied",
}


def trained_runs_rows(final: dict) -> list[list[str]]:
    """Return the rows of the table of trained runs, one per run, then the median rows."""
    rows = []
    for row in final["rows"]:
        collapse_text = ", ".join(
            f"{step(event['step'])} ({number(event['drop'], SUCCESS_CHANGE)})"
            for event in row["collapses"]
        )
        rows.append(
            [
                row["role"],
                row["tag"],
                str(row["task"]),
                str(row["confirm_count"]),
                number(row["r_last"], REGRET),
                number(row["scan_argmax_earliest"], REGRET),
                number(row["scan_argmax_latest"], REGRET),
                collapse_text or "none",
            ]
        )
    for group, label in MEDIAN_ROWS.items():
        medians = final["medians"][group]
        rows.append(
            [
                label,
                DASH,
                DASH,
                DASH,
                number(medians["r_last"], REGRET),
                number(medians["scan_argmax_earliest"], REGRET),
                number(medians["scan_argmax_latest"], REGRET),
                DASH,
            ]
        )
    return rows


def trained_runs_section(results: dict, config: dict) -> tuple[str, str, list[str]]:
    """Return the section of keeping the final checkpoint on every trained run."""
    floor = config["protocol"]["regret_floor"]
    final = results["final_checkpoint"]
    task8_count = final["group_sizes"]["task8"]
    scan_argmax = final["scan_argmax"]
    counted = results["collapses_if_counted"]
    counts = results["run_counts"]
    return (
        "Keeping the final checkpoint on every trained run",
        "R_last is the regret of keeping the final checkpoint. One row per run trained for "
        "the full schedule. Rows are sorted by run group (earlier, replication, control), "
        "then by R_last, highest first; equal values keep the config's run order. The "
        "scan-argmax columns give the confirmed success of the highest-scanning checkpoint "
        "minus the final checkpoint's, with ties broken toward the earliest or the latest "
        "step. A collapse is a confirmed drop of more than "
        f"{config['protocol']['collapse_points']} points from a competent checkpoint that the "
        "next confirmed checkpoint rises from; the last column gives its step and drop in "
        "points.",
        [
            table(
                [
                    "Group",
                    "Run",
                    "Task",
                    "Confirmed checkpoints",
                    "R_last (points)",
                    "Scan argmax, earliest tie (points)",
                    "Scan argmax, latest tie (points)",
                    "Collapses",
                ],
                trained_runs_rows(final),
            ),
            bullets(
                [
                    f"Runs: {counts['earlier']} earlier, {counts['replication']} replication "
                    f"({counts['task8']} on task 8), {counts['control']} control",
                    f"Task-8 runs with R_last above the {floor}-point floor: "
                    f"{final['count_over_floor_task8']} of {task8_count}",
                    f"Highest R_last on a control run: {number(final['max_control'], REGRET)}",
                    *(
                        f"Scan argmax with the {tie} tie on the task-8 runs: above the floor "
                        f"on {scan_argmax[tie]['count_over_floor_task8']} of {task8_count}, "
                        f"median {number(scan_argmax[tie]['median_task8'], REGRET)}"
                        for tie in selectors.TIE_RULES
                    ),
                    f"Task-8 runs with a collapse: {final['collapse_runs']}, with "
                    f"{final['collapse_events']} collapses",
                    f"Collapses on the control runs: {results['control_collapse_count']}",
                    "Collapses if drops from checkpoints that are not competent also count: "
                    f"{counted['events']} on {counted['runs']} runs",
                ]
            ),
        ],
    )


def task8_section(results: dict, config: dict) -> tuple[str, str, list[str]]:
    """Return the section of R_last of the task-8 runs."""
    floor = config["protocol"]["regret_floor"]
    final = results["final_checkpoint"]
    roles = {row["tag"]: row["role"] for row in final["rows"]}
    task8_rows = [
        [tag, roles[tag], number(value, REGRET)] for tag, value in results["task8_r_last"].items()
    ]
    task8_rows.append(["Median", DASH, number(final["medians"]["task8"]["r_last"], REGRET)])
    return (
        "R_last of the task-8 runs",
        "R_last of every earlier and replication run, highest first; equal values keep the "
        f"config's run order. The last row gives the median. The regret floor is {floor} "
        "points.",
        [table(["Run", "Group", "R_last (points)"], task8_rows)],
    )


def scan_success_section(results: dict) -> tuple[str, str, list[str]]:
    """Return the section of scan success at every checkpoint."""
    scan_success = results["scan_success_by_step"]
    scan_rows = [*scan_success["per_run"].items(), ("Median", scan_success["median"])]
    return (
        "Scan success at every checkpoint",
        "Scan success of every earlier and replication run, in percent, at every checkpoint. "
        "Rows follow the config's run order; the last row gives the median over runs at each "
        "checkpoint. Columns are checkpoints in step order.",
        [step_table("Run", scan_rows, percent)],
    )


def terminal_drop_section(results: dict) -> tuple[str, str, list[str]]:
    """Return the section of the fall into the final checkpoint."""
    terminal_rows = [
        [
            tag,
            drop["kind"],
            step(drop["from_step"]),
            percent(drop["from_rate"]),
            percent(drop["final_rate"]),
            number(drop["drop"], SUCCESS_CHANGE),
        ]
        for tag, drop in results["terminal_drops"].items()
    ]
    return (
        "Fall into the final checkpoint",
        "For each task-8 run whose R_last is above the floor, the fall from the checkpoint "
        "saved before the final one. The fall uses confirmed success when both checkpoints "
        "are confirmed, else scan success. Rows follow the config's run order.",
        [
            table(
                [
                    "Run",
                    "Measured on",
                    "From step",
                    "From success (%)",
                    "Final success (%)",
                    "Drop (points)",
                ],
                terminal_rows,
            )
        ],
    )


def control_fall_section(results: dict) -> tuple[str, str, list[str]]:
    """Return the section of the largest scan fall on the control runs."""
    fall_rows = [
        [
            tag,
            number(fall["drop"], SUCCESS_CHANGE),
            step(fall.get("from_step")),
            step(fall.get("to_step")),
            percent(fall.get("from_rate")),
            percent(fall.get("base_scan")),
            yes_no(fall.get("from_above_base", False)),
        ]
        for tag, fall in results["control_scan_falls"].items()
    ]
    best = results["control_best_confirmed"]
    return (
        "Largest scan fall on the control runs",
        "The largest fall in scan success from one checkpoint to any later one. Rows follow "
        "the config's run order.",
        [
            table(
                [
                    "Run",
                    "Fall (points)",
                    "From step",
                    "To step",
                    "From success (%)",
                    "Base scan success (%)",
                    "Starts above base",
                ],
                fall_rows,
            ),
            bullets(
                [
                    "Best confirmed success on the control runs: "
                    f"{percent(best['min'])}-{percent(best['max'])}%"
                ]
            ),
        ],
    )


def oracle_bias_section(results: dict, config: dict) -> tuple[str, str, list[str]]:
    """Return the section of the bias of the best confirmed success."""
    bias = results["oracle_bias"]
    bias_rows = [
        [
            BIAS_MODELS[model],
            number(values["median"], ORACLE_BIAS),
            number(values["min"], ORACLE_BIAS),
            number(values["max"], ORACLE_BIAS),
        ]
        for model, values in bias.items()
    ]
    bias_run_rows = [
        [tag, *(number(bias[model]["per_run"][tag], ORACLE_BIAS) for model in BIAS_MODELS)]
        for tag in bias["independent"]["per_run"]
    ]
    unconfirmed = results["unconfirmed_above_oracle"]
    unconfirmed_rows = [
        [item["tag"], step(item["step"]), percent(item["scan"]), percent(item["oracle"])]
        for item in unconfirmed["checkpoints"]
    ]
    return (
        "Bias of the best confirmed success",
        "The best confirmed success of a run overstates its best true rate. The bias is the "
        "expected maximum of independent binomial estimates on "
        f"{config['evaluation']['confirm']['episodes']} episodes minus the largest true rate, "
        "in points, over the task-8 runs. The true rates are the observed confirmed rates, "
        "the observed rates with near-ties within "
        f"{config_percent(config['protocol']['oracle_bias_shrink_band'])} points shrunk toward "
        "their mean, or every rate tied at the best. Rows of the first table list the models "
        "in this order. Rows of the second table follow the config's run order. Rows of the "
        "last table follow the config's run order, then the step.",
        [
            table(["Model", "Median (points)", "Lowest (points)", "Highest (points)"], bias_rows),
            table(["Run", *BIAS_MODELS.values()], bias_run_rows),
            bullets(
                [
                    "Unconfirmed checkpoints whose scan success is at or above their run's "
                    f"best confirmed success: {unconfirmed['count']}"
                ]
            ),
            table(
                ["Run", "Step", "Scan success (%)", "Best confirmed success (%)"],
                unconfirmed_rows,
            ),
        ],
    )


def median_sensitivity_section(results: dict) -> tuple[str, str, list[str]]:
    """Return the section of the sensitivity of the median R_last."""
    sensitivity = results["median_sensitivity"]
    sensitivity_rows = [
        [tag, number(value, REGRET)] for tag, value in sensitivity["leave_one_out"].items()
    ]
    return (
        "Sensitivity of the median R_last",
        "The median R_last over the earlier and control runs, with each run left out in "
        "turn. Rows follow the order of the table of trained runs above.",
        [
            table(["Run left out", "Median R_last (points)"], sensitivity_rows),
            bullets(
                [
                    f"Median R_last with every run: {number(sensitivity['median'], REGRET)}",
                    "Runs whose removal changes it: "
                    f"{len(sensitivity['changed_by_leaving_out'])} "
                    f"({tags(sensitivity['changed_by_leaving_out'])})",
                ]
            ),
        ],
    )


def rerun_section(results: dict) -> tuple[str, str, list[str]]:
    """Return the section of the approximate rerun and the run it repeats."""
    rerun_rows = []
    for tag, pair in results["rerun_pairs"].items():
        for name, r_last, drop in (
            (tag, pair["rerun_r_last"], pair["rerun_final_drop"]),
            (pair["original"], pair["original_r_last"], pair["original_final_drop"]),
        ):
            rerun_rows.append(
                [
                    name,
                    number(r_last, REGRET),
                    drop["kind"],
                    step(drop["from_step"]),
                    percent(drop["from_rate"]),
                    percent(drop["final_rate"]),
                    number(drop["drop"], SUCCESS_CHANGE),
                ]
            )
    return (
        "Approximate rerun",
        "R_last and the fall into the final checkpoint of the approximate rerun and of the "
        "run it repeats with the same configuration. Each rerun's row comes before the row "
        "of the run it repeats.",
        [
            table(
                [
                    "Run",
                    "R_last (points)",
                    "Fall measured on",
                    "From step",
                    "From success (%)",
                    "Final success (%)",
                    "Fall (points)",
                ],
                rerun_rows,
            )
        ],
    )


def noncompetent_drop_section(results: dict) -> tuple[str, str, list[str]]:
    """Return the section of the drops from checkpoints that are not competent."""
    noncompetent_rows = [
        [
            tag,
            step(drop["step"]),
            step(drop["from_step"]),
            percent(drop["from_rate"]),
            percent(drop["to_rate"]),
            number(drop["drop"], SUCCESS_CHANGE),
        ]
        for tag, drops in results["noncompetent_drops"].items()
        for drop in drops
    ]
    return (
        "Drops from checkpoints that are not competent",
        "Confirmed drops of more than the collapse threshold, followed by a rise, that start "
        "from a checkpoint whose confirmed success is not above the base policy's scan "
        "success. Rows follow the config's run order, then the step.",
        [
            table(
                [
                    "Run",
                    "Step",
                    "From step",
                    "From success (%)",
                    "To success (%)",
                    "Drop (points)",
                ],
                noncompetent_rows,
            )
        ],
    )


def final_checkpoint_sections(results: dict, config: dict) -> list[tuple[str, str, list[str]]]:
    """Return the regret page's sections on keeping the final checkpoint and its falls."""
    return [
        trained_runs_section(results, config),
        task8_section(results, config),
        scan_success_section(results),
        terminal_drop_section(results),
        control_fall_section(results),
        oracle_bias_section(results, config),
        median_sensitivity_section(results),
        rerun_section(results),
        noncompetent_drop_section(results),
    ]
