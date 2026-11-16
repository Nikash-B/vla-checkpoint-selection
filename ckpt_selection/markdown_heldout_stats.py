"""Write the Markdown page of the held-out statistics through training."""

import itertools

from ckpt_selection.markdown import (
    Q_VALUE,
    RELATIVE_L2,
    SUCCESS_CHANGE,
    bullets,
    number,
    page,
    percent,
    step,
    step_table,
    table,
    table_label,
    yes_no,
)

HELDOUT_DECIMALS = {"l2": 3, "kl": 1, "mmd": 2, "action_entropy": 2}  # per held-out statistic
HELDOUT_LABELS = {
    "l2": "L2",
    "kl": "KL (nats)",
    "mmd": "Squared MMD",
    "action_entropy": "Action entropy",
}


def summary_section(results: dict, config: dict) -> tuple[str, str, list[str]]:
    """Return the section of the summary over the earlier runs."""
    summary = results["summary"]
    columns = list(summary)
    first = step(summary[columns[0]]["first_checkpoint_step"])
    quantities = (
        ("Base policy", "base"),
        (f"First checkpoint ({first}), lowest", "first_min"),
        (f"First checkpoint ({first}), highest", "first_max"),
        ("Plateau mean, lowest", "mean_min"),
        ("Plateau mean, highest", "mean_max"),
        ("Plateau SD, lowest", "sd_min"),
        ("Plateau SD, highest", "sd_max"),
    )
    summary_rows = [
        [name, *(number(summary[key][field], HELDOUT_DECIMALS[key]) for key in columns)]
        for name, field in quantities
    ]
    return (
        "Summary over the earlier runs",
        f"Each statistic is measured at every checkpoint of the {summary['l2']['n_runs']} "
        "earlier runs with held-out statistics. Plateau values use the checkpoints from "
        f"{step(config['heldout']['plateau_start'])} on, and the SD is over a run's plateau "
        "checkpoints. Lowest and highest are over runs. Rows are quantities; columns are "
        "statistics.",
        [table(["Quantity", *(HELDOUT_LABELS[key] for key in columns)], summary_rows)],
    )


def by_step_section(results: dict) -> tuple[str, str, list[str]]:
    """Return the section of the held-out statistics at every checkpoint."""
    summary = results["summary"]
    by_step = results["by_step"]
    statistic_blocks = []
    for key in summary:
        rows = [*by_step[key]["per_run"].items(), ("Median", by_step[key]["median"])]
        statistic_blocks.append(table_label(HELDOUT_LABELS[key]))
        statistic_blocks.append(
            step_table("Run", rows, lambda value, key=key: number(value, HELDOUT_DECIMALS[key]))
        )
    return (
        "Held-out statistics at every checkpoint",
        "Each statistic of each earlier run with held-out statistics, at every checkpoint. "
        "Rows follow the config's run order; the last row gives the median over runs at each "
        "checkpoint. Columns are checkpoints in step order.",
        [
            *statistic_blocks,
            bullets([f"Base-policy L2: {number(summary['l2']['base'], HELDOUT_DECIMALS['l2'])}"]),
        ],
    )


def confirmed_section(results: dict) -> tuple[str, str, list[str]]:
    """Return the section of confirmed success at the confirmed checkpoints."""
    return (
        "Confirmed success at the confirmed checkpoints",
        "Confirmed success of each earlier run with held-out statistics, in percent. Rows "
        "follow the config's run order; columns are checkpoints in step order. A dash means "
        "that the run did not confirm the checkpoint.",
        [step_table("Run", list(results["confirmed_by_step"].items()), percent)],
    )


def l2_above_base_section(results: dict) -> tuple[str, str, list[str]]:
    """Return the section of L2 above the base policy."""
    l2 = results["l2_above_base"]
    return (
        "L2 above the base policy",
        "Each run's mean plateau L2 relative to the base policy's L2. Rows follow the "
        "config's run order.",
        [
            table(
                ["Run", "Mean plateau L2 above base (%)"],
                [[tag, number(value, RELATIVE_L2)] for tag, value in l2["per_run"].items()],
            ),
            bullets(
                [
                    f"Range over runs: {number(l2['min'], RELATIVE_L2)}-"
                    f"{number(l2['max'], RELATIVE_L2)}%"
                ]
            ),
        ],
    )


def collapse_section(results: dict) -> tuple[str, str, list[str]]:
    """Return the section of the final checkpoint that falls most."""
    collapse = results["collapse_l2"]
    spread = results["spread_at_collapse"]
    return (
        "The final checkpoint that falls most",
        "The run whose final checkpoint falls most below the confirmed checkpoint saved before it.",
        [
            bullets(
                [
                    f"Run: {collapse['tag']}",
                    f"Confirmed success at {step(collapse['steps'][0])} and "
                    f"{step(collapse['steps'][1])}: {percent(collapse['confirmed'][0])}% and "
                    f"{percent(collapse['confirmed'][1])}% (drop "
                    f"{number(collapse['drop'], SUCCESS_CHANGE)} points)",
                    f"L2 above base at these checkpoints: "
                    f"{number(collapse['l2_above_base'][0], RELATIVE_L2)}% and "
                    f"{number(collapse['l2_above_base'][1], RELATIVE_L2)}%",
                    f"Ensemble spread at the final checkpoint: "
                    f"{number(spread['final_spread'], Q_VALUE)} reward units; the run's "
                    f"median over checkpoints: {number(spread['median_spread'], Q_VALUE)}; "
                    f"final below median: {yes_no(spread['final_below_median'])}",
                ]
            )
        ],
    )


def control_l2_section(results: dict) -> tuple[str, str, list[str]]:
    """Return the section of L2 on the control runs."""
    return (
        "L2 on the control runs",
        "The largest deviation of L2 from the base policy of the run's task, over all "
        "checkpoints. A dash means that no run of the task has a base-policy held-out file. "
        "Rows follow the config's run order.",
        [
            table(
                ["Run", "Largest L2 deviation from base (%)"],
                [
                    [tag, number(value, RELATIVE_L2)]
                    for tag, value in results["control_l2_band"].items()
                ],
            )
        ],
    )


def distance_rules_section(results: dict, config: dict) -> tuple[str, str, list[str]]:
    """Return the section of L2, KL and MMD as selectors."""
    rules = results["distance_rules"]
    choices = config["selectors"]
    rule_rows = []
    for statistic, direction, min_step in itertools.product(
        choices["distance_statistics"],
        choices["distance_directions"],
        choices["distance_min_steps"],
    ):
        rule = rules["rules"][f"{statistic}_{direction}_from_{min_step}"]
        rule_rows.append(
            [
                statistic.upper(),
                "highest" if direction == "max" else "lowest",
                step(min_step) if min_step else "all",
                number(rule["mean_scan_delta"], SUCCESS_CHANGE),
                str(rule["scan_worse_over_floor"]),
                str(rule["scan_better_over_floor"]),
                str(len(rule["confirmed_regret_minus_final"])),
            ]
        )
    floor = config["protocol"]["regret_floor"]
    return (
        "L2, KL and MMD as selectors",
        "Each rule picks the checkpoint with the highest or lowest value of one statistic, "
        "over all checkpoints or over those from a minimum step on. We score each rule by its "
        "pick's scan success minus the final checkpoint's, in points, on every run. Rows go "
        "by statistic, then direction, then minimum step. A rule counts as worse when its "
        "mean scan change is below zero. The last line counts the rules on which more runs "
        f"gain than lose by more than the {floor}-point floor.",
        [
            table(
                [
                    "Statistic",
                    "Direction",
                    "From step",
                    "Mean scan change (points)",
                    f"Runs worse by more than {floor}",
                    f"Runs better by more than {floor}",
                    "Runs whose pick is confirmed",
                ],
                rule_rows,
            ),
            bullets(
                [
                    f"Rules: {rules['n_rules']}, with {rules['n_distinct']} distinct sets of picks",
                    f"Rules with a mean scan change below zero: {rules['n_mean_scan_worse']}",
                    "Rules on which more runs gain than lose by more than the floor: "
                    f"{rules['n_more_better_than_worse_over_floor']}",
                ]
            ),
        ],
    )


def heldout_stats_page(results: dict, config: dict) -> str:
    """Return the page of the held-out statistics through training."""
    sections = [
        summary_section(results, config),
        by_step_section(results),
        confirmed_section(results),
        l2_above_base_section(results),
        collapse_section(results),
        control_l2_section(results),
        distance_rules_section(results, config),
    ]
    return page(
        "Held-out statistics through training",
        "We measured the policy's actions against held-out demonstrations at every checkpoint: "
        "L2 distance, KL divergence, squared MMD and action entropy.",
        sections,
    )
