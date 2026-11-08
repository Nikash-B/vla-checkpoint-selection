"""Write the Markdown page of regret of the selectors and of keeping the final checkpoint."""

from ckpt_selection import markdown_final_checkpoint, regret
from ckpt_selection.markdown import REGRET, bullets, number, page, step, table, table_label


def selector_label(key: str, row: dict, config: dict) -> str:
    """Return a selector's label; the restricted critic's names its earliest allowed step."""
    if key == "restricted_critic":
        return f"{row['label']} ({step(config['selectors']['restricted_min_step'])} or later)"
    return row["label"]


def selector_labels(results: dict, config: dict) -> dict[str, str]:
    """Return the label of each compared selector, in the order of regret.COMPARED_SELECTORS."""
    selector_table = results["selectors"]
    return {
        key: selector_label(key, selector_table[key], config)
        for key, _, _ in regret.COMPARED_SELECTORS
    }


def run_tags(results: dict) -> list[str]:
    """Return the tags of the runs that the selectors are compared on, in the config's order."""
    return list(results["selectors"]["final"]["per_run"])


def summary_section(results: dict, config: dict) -> tuple[str, str, list[str]]:
    """Return the section of the selectors against keeping the final checkpoint."""
    selector_table = results["selectors"]
    labels = selector_labels(results, config)
    summary_rows = [
        [
            label,
            number(selector_table[key]["mean"], REGRET),
            number(selector_table[key]["median"], REGRET),
            number(selector_table[key]["worst"], REGRET),
            str(selector_table[key]["episodes"]),
        ]
        for key, label in labels.items()
    ]
    lowest = results["lowest_regret_selectors"]
    return (
        "Selectors against keeping the final checkpoint",
        f"The table covers the {len(run_tags(results))} earlier runs with held-out statistics. "
        "Rows give keeping the final checkpoint, then the rollout-free selectors, then the "
        "highest scan success as a reference that needs evaluation episodes. Ties between "
        f"checkpoints go to the {config['selectors']['tie_rule']} step.",
        [
            table(
                [
                    "Selector",
                    "Mean regret (points)",
                    "Median regret (points)",
                    "Worst regret (points)",
                    "Evaluation episodes per run",
                ],
                summary_rows,
            ),
            bullets(
                [
                    f"Lowest {column} regret among the rollout-free selectors: "
                    + ", ".join(labels[key] for key in lowest[column])
                    for column in ("mean", "median", "worst")
                ]
            ),
        ],
    )


def per_run_section(results: dict, config: dict) -> tuple[str, str, list[str]]:
    """Return the section of the regret and pick of each selector on each run."""
    selector_table = results["selectors"]
    labels = selector_labels(results, config)
    per_run_rows = [
        [tag, *(number(selector_table[key]["per_run"][tag], REGRET) for key in labels)]
        for tag in run_tags(results)
    ]
    pick_rows = [
        [tag, *(step(selector_table[key]["picks"][tag]) for key in labels)]
        for tag in run_tags(results)
    ]
    return (
        "Regret and pick of each selector on each run",
        "The first table gives each selector's regret in points, the second its selected "
        "step. Rows of both follow the config's run order.",
        [
            table_label("Regret (points)"),
            table(["Run", *labels.values()], per_run_rows),
            table_label("Selected step"),
            table(["Run", *labels.values()], pick_rows),
        ],
    )


def restricted_section(results: dict, config: dict) -> tuple[str, str, list[str]]:
    """Return the section of the restricted critic and the other selectors against R_last."""
    floor = config["protocol"]["regret_floor"]
    restricted = results["restricted_critic"]
    cutoff = restricted["at_cutoff"]
    difference_rows = [
        [tag, number(value, REGRET)]
        for tag, value in sorted(restricted["difference"].items(), key=lambda item: -item[1])
    ]
    entropy = results["lowest_entropy_vs_final"]
    n_runs = len(run_tags(results))
    return (
        "Restricted critic and the other selectors against keeping the final checkpoint",
        "The table gives the restricted critic's regret minus R_last, sorted from largest to "
        "smallest.",
        [
            table(["Run", "Restricted critic minus R_last (points)"], difference_rows),
            bullets(
                [
                    f"Runs where the restricted critic is worse than keeping the final "
                    f"checkpoint: {restricted['worse']} of {n_runs}",
                    f"Runs where it is worse by more than the {floor}-point floor: "
                    f"{restricted['worse_over_floor']} of {n_runs}",
                    f"Runs where it picks the earliest allowed checkpoint, "
                    f"{step(cutoff['cutoff_step'])}: {cutoff['count']} of {cutoff['n_runs']}",
                    f"The {cutoff['largest_regret_runs_at_cutoff']} runs with the largest "
                    "restricted-critic regret all pick the earliest allowed checkpoint",
                    "Runs where the ensemble-spread veto keeps the final checkpoint: "
                    f"{results['spread_veto_final_count']} of {n_runs}",
                    f"Lowest action entropy on {entropy['collapsed_tag']}, the run whose "
                    f"final checkpoint falls most: regret "
                    f"{number(entropy['collapsed_entropy_regret'], REGRET)} against R_last "
                    f"{number(entropy['collapsed_r_last'], REGRET)}",
                    "Lowest action entropy on the other runs: worse than keeping the final "
                    f"checkpoint by more than the floor on {entropy['others_worse_over_floor']}"
                    f", better by more than the floor on {entropy['others_better_over_floor']}",
                ]
            ),
        ],
    )


def regret_page(results: dict, config: dict) -> str:
    """Return the page of regret of the selectors and of keeping the final checkpoint."""
    floor = config["protocol"]["regret_floor"]
    sections = [
        summary_section(results, config),
        per_run_section(results, config),
        restricted_section(results, config),
        *markdown_final_checkpoint.final_checkpoint_sections(results, config),
    ]
    return page(
        "Regret of checkpoint selectors",
        "Regret is a run's best confirmed success minus the confirmed success of the selected "
        f"checkpoint, in points. The regret floor is {floor} points.",
        sections,
    )
