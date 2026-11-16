"""Write the Markdown page of the confirm sets rebuilt from their rules."""

from ckpt_selection.markdown import bullets, page, steps, table


def confirm_sets_page(results: dict, config: dict) -> str:
    """Return the page of the confirm sets rebuilt from their rules."""
    rows = [
        [
            tag,
            check["role"],
            check["rule"],
            str(check["confirmed_count"]),
            steps(check["rule_only"]),
            steps(check["confirmed_only"]),
        ]
        for tag, check in results["runs"].items()
    ]
    return page(
        "Confirm sets",
        "Each run confirms the checkpoints that its confirm rule in config/experiments.toml "
        "chooses. We rebuild every confirm set from the run's scan and held-out files and "
        "compare it with the checkpoints the run confirmed. The analysis reads the confirm files "
        "that exist, so a difference changes no other result. Rows follow the config's run "
        "order.",
        [
            (
                "Rebuilt against confirmed",
                "",
                [
                    table(
                        [
                            "Run",
                            "Group",
                            "Rule",
                            "Confirmed checkpoints",
                            "Rule adds, not confirmed",
                            "Confirmed, rule does not add",
                        ],
                        rows,
                    ),
                    bullets(
                        [
                            f"Runs whose rebuilt set matches: {results['matching_count']} of "
                            f"{results['run_count']}",
                            "Confirmed checkpoints per task-8 run: "
                            f"{results['task8_confirmed_min']} to {results['task8_confirmed_max']}",
                        ]
                    ),
                ],
            )
        ],
    )
