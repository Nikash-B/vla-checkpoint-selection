"""Write the Markdown page of scan success against confirmed success."""

from ckpt_selection.markdown import (
    DASH,
    SCAN_BIAS,
    SUCCESS_SE,
    bullets,
    config_percent,
    number,
    page,
    percent,
    step,
    table,
)

SCAN_BIAS_GROUPS = {
    "argmax_earliest": "Highest scan success, earliest tie",
    "argmax_latest": "Highest scan success, latest tie",
    "final": "Final checkpoint",
}


def checkpoint_pairs(items: list[dict]) -> str:
    """Return checkpoints as run, step, scan success and confirmed success."""
    text = "; ".join(
        f"{item['tag']} at {step(item['step'])}: scan {percent(item['scan'])}%, "
        f"confirmed {percent(item['confirm'])}%"
        for item in items
    )
    return text or DASH


def scan_bias_page(results: dict, config: dict) -> str:
    """Return the page of scan success against confirmed success."""
    group_rows = [
        [
            name,
            number(results[group]["mean"], SCAN_BIAS),
            number(results[group]["se"], SCAN_BIAS),
            str(results[group]["n_runs"]),
        ]
        for group, name in SCAN_BIAS_GROUPS.items()
    ]
    run_rows = [
        [tag, *(number(results[group]["per_run"][tag], SCAN_BIAS) for group in SCAN_BIAS_GROUPS)]
        for tag in results["final"]["per_run"]
    ]
    scan = config["evaluation"]["scan"]["episodes"]
    reference = config["scan_bias"]["reference_rate"]
    sections = [
        (
            "Bias by group",
            "Mean over runs of scan success minus confirmed success, with its standard error. "
            "Rows list the groups in a fixed order: the highest scan success with the earliest "
            "tie, then with the latest tie, then the final checkpoint.",
            [table(["Group", "Mean (points)", "Standard error (points)", "Runs"], group_rows)],
        ),
        (
            "Bias on each run",
            "Scan success minus confirmed success in points. Rows follow the config's run order.",
            [
                table(["Run", *SCAN_BIAS_GROUPS.values()], run_rows),
                bullets(
                    [
                        "Runs where the final checkpoint ties for the highest scan success: "
                        + checkpoint_pairs(results["final_tied_runs"]),
                        "Runs where the final checkpoint alone has the highest scan success: "
                        + checkpoint_pairs(results["final_unique_runs"]),
                        f"Standard error of a scan success near {config_percent(reference)}% on "
                        f"{scan} episodes: {number(results['scan_se_at_reference'], SUCCESS_SE)} "
                        "points",
                    ]
                ),
            ],
        ),
    ]
    return page(
        "Scan success against confirmed success",
        "On a confirmed checkpoint, scan success minus confirmed success measures how far the "
        f"scan overstates the checkpoint. We use the {results['n_runs']} runs whose confirm rule "
        f"is {config['scan_bias']['confirm_rule']}. The groups are the highest-scanning "
        "checkpoint, with ties broken toward the earliest or the latest step, and the final "
        "checkpoint.",
        sections,
    )
