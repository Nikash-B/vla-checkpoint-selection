"""Write the Markdown page of held-out Q and replay-batch Q as checkpoint selectors."""

from ckpt_selection.markdown import (
    Q_VALUE,
    REGRET,
    RHO,
    SUCCESS_CHANGE,
    SUCCESS_SE,
    all_steps,
    bullets,
    number,
    page,
    percent,
    step,
    step_table,
    steps,
    table,
    tags,
    yes_no,
)


def heldout_peaks_section(results: dict, config: dict, cutoff: str) -> tuple[str, str, list[str]]:
    """Return the section of held-out Q on the earlier runs."""
    states = config["heldout"]["demos"] * config["heldout"]["states_per_demo"]
    heldout_q = results["heldout_q"]
    n_runs = heldout_q["peak_runs"]
    peak_rows = []
    for tag, peak in heldout_q["peaks"].items():
        gap = heldout_q["below_base"][tag]
        peak_rows.append(
            [
                tag,
                step(peak["step"]),
                percent(peak["confirmed"]),
                percent(gap["base_scan"]),
                number(gap["gap"], SUCCESS_CHANGE),
                number(gap["se"], SUCCESS_SE),
                number(heldout_q["rho_all"][tag], RHO),
                number(heldout_q["rho_late"][tag], RHO),
            ]
        )
    return (
        "Held-out Q on the earlier runs",
        f"Held-out Q is the critic's mean value at {states} states of held-out "
        "demonstrations. For each run, the table gives the checkpoint with the highest "
        "held-out Q, its confirmed success, and its gap below the base policy's scan success "
        "with the standard error of that gap. The last two columns give Spearman's ρ of "
        "held-out Q with the training step over all checkpoints and over the checkpoints "
        f"from {cutoff} on. Rows follow the config's run order.",
        [
            table(
                [
                    "Run",
                    "Peak step",
                    "Confirmed success at peak (%)",
                    "Base scan success (%)",
                    "Gap below base (points)",
                    "SE of gap (points)",
                    "ρ with step, all checkpoints",
                    f"ρ with step from {cutoff}",
                ],
                peak_rows,
            ),
            bullets(
                [
                    f"Runs: {n_runs}",
                    f"Distinct peak steps: {steps(heldout_q['peak_steps'])}",
                    "Highest confirmed success at a peak: "
                    f"{percent(heldout_q['max_success_at_peak'])}%",
                    f"Runs with ρ below zero from {cutoff} on: "
                    f"{heldout_q['late_negative_count']} of {n_runs}",
                    "Runs whose peak confirms below the base policy's scan success: "
                    f"{heldout_q['below_base_count']} of {n_runs}",
                    f"Runs below it by at least {config['protocol']['below_base_ses']} "
                    f"standard errors: {heldout_q['below_base_2se_count']} of {n_runs}",
                ]
            ),
        ],
    )


def heldout_by_step_section(results: dict) -> tuple[str, str, list[str]]:
    """Return the section of held-out Q at every checkpoint."""
    heldout_q_rows = list(results["heldout_q_by_step"].items())
    return (
        "Held-out Q at every checkpoint",
        "Held-out Q of each earlier run with held-out statistics, in reward units, at every "
        "checkpoint. Rows follow the config's run order; columns are checkpoints in step "
        "order.",
        [step_table("Run", heldout_q_rows, lambda value: number(value, Q_VALUE))],
    )


def control_heldout_section(results: dict, cutoff: str) -> tuple[str, str, list[str]]:
    """Return the section of held-out Q on the control runs."""
    control = results["control_heldout_q"]
    control_rows = [
        [
            tag,
            step(peak["step"]),
            yes_no(peak["early"]),
            percent(peak["confirmed"]),
            number(peak["regret"], REGRET),
            percent(peak["base_scan"]),
            yes_no(peak["learned_by_peak"]),
        ]
        for tag, peak in control["runs"].items()
    ]
    return (
        "Held-out Q on the control runs",
        "For each control run, the checkpoint with the highest held-out Q, whether it comes "
        f"before {cutoff}, its confirmed success and regret, and whether it confirms above "
        "the base policy's scan success. Rows follow the config's run order.",
        [
            table(
                [
                    "Run",
                    "Peak step",
                    f"Before {cutoff}",
                    "Confirmed success (%)",
                    "Regret (points)",
                    "Base scan success (%)",
                    "Confirms above base",
                ],
                control_rows,
            ),
            bullets(
                [
                    f"Runs whose peak comes before {cutoff}: {control['early_count']} of "
                    f"{len(control['runs'])}"
                ]
            ),
        ],
    )


def replay_section(results: dict, cutoff: str) -> tuple[str, str, list[str]]:
    """Return the section of replay-batch Q from the training logs."""
    replay = results["replay_q"]
    replay_rows = [
        [
            tag,
            step(row["peak_step"]),
            number(row["peak_value"], Q_VALUE),
            percent(row["scan_at_peak"]),
            step(row["trough_step"]),
            number(row["trough_value"], Q_VALUE),
            number(row["fall"], Q_VALUE),
            number(row["late_rho"], RHO),
            number(row["final_value"], Q_VALUE),
        ]
        for tag, row in replay["per_run"].items()
    ]
    replay_count = len(replay["per_run"])
    return (
        "Replay-batch Q from the training logs",
        "Replay-batch Q is the training log's mean critic value on replay batches, in reward "
        "units. The table reads it at the checkpoint steps. The trough is the lowest value "
        "after the peak, and the fall is the peak minus the trough. ρ is Spearman's ρ with "
        f"the step from {cutoff} on. Rows follow the config's run order.",
        [
            table(
                [
                    "Run",
                    "Peak step",
                    "Peak Q (reward units)",
                    "Scan success at peak (%)",
                    "Trough step",
                    "Trough Q (reward units)",
                    "Fall (reward units)",
                    f"ρ with step from {cutoff}",
                    "Final Q (reward units)",
                ],
                replay_rows,
            ),
            bullets(
                [
                    f"Runs with a training log: {replay_count}",
                    f"Median interval between logged steps: {step(replay['log_interval'])}",
                    f"Distinct peak steps: {steps(replay['peak_steps'])}",
                    "Scan success at the peak: "
                    f"{percent(replay['scan_at_peak_range'][0])}-"
                    f"{percent(replay['scan_at_peak_range'][1])}%",
                    f"Fall from peak to trough: {number(replay['fall_range'][0], Q_VALUE)} to "
                    f"{number(replay['fall_range'][1], Q_VALUE)} reward units",
                    f"Runs whose final value is above the trough: {replay['recovers']} of "
                    f"{replay_count}",
                    f"Runs with ρ above zero from {cutoff} on: {replay['late_positive']} of "
                    f"{replay_count}",
                    f"Runs whose final value is below the peak: "
                    f"{replay['below_peak_at_final']} of {replay_count}",
                ]
            ),
        ],
    )


def replay_by_logged_step_section(results: dict) -> tuple[str, str, list[str]]:
    """Return the section of replay-batch Q at every logged step."""
    logged = results["replay_q"]["by_logged_step"]
    logged_tags = list(logged["per_run"])
    logged_rows = [
        [
            step(logged_step),
            *(number(logged["per_run"][tag].get(logged_step), Q_VALUE) for tag in logged_tags),
            number(logged["median"].get(logged_step), Q_VALUE),
        ]
        for logged_step in all_steps(list(logged["per_run"].values()))
    ]
    return (
        "Replay-batch Q at every logged step",
        "Replay-batch Q of each run with a training log, in reward units, at every logged "
        "step. Rows are logged steps in step order; columns are runs in the config's run "
        "order, then the median over runs at the steps that every run logs.",
        [table(["Step", *logged_tags, "Median"], logged_rows)],
    )


def kumar_section(results: dict, config: dict) -> tuple[str, str, list[str]]:
    """Return the section of the rules of Kumar et al. on replay-batch Q."""
    replay = results["replay_q"]
    kumar = replay["kumar"]
    kumar_names = {
        "before_decline": "Highest value, the peak before the decline",
        "highest_late": f"Highest value from {step(config['selectors']['kumar_min_step'])} on",
        "latest_local_peak": "Latest local peak before the final checkpoint",
    }
    kumar_rows = [
        [name, steps(kumar[rule]["steps"]), str(kumar[rule]["in_last_checkpoints"])]
        for rule, name in kumar_names.items()
    ]
    kumar_run_rows = [
        [tag, *(step(row["kumar"][rule]) for rule in kumar_names)]
        for tag, row in replay["per_run"].items()
    ]
    return (
        "Rules of Kumar et al. on replay-batch Q",
        "Each rule picks one checkpoint per run from replay-batch Q at the checkpoint steps. "
        "Rows of the first table list the rules in a fixed order. Its last column counts the "
        f"runs whose pick is one of their last {kumar['last_checkpoints']} checkpoints. The "
        "second table gives each run's picks, in the config's run order.",
        [
            table(
                [
                    "Rule",
                    "Steps picked",
                    f"Runs with a pick in the last {kumar['last_checkpoints']}",
                ],
                kumar_rows,
            ),
            table(["Run", *kumar_names.values()], kumar_run_rows),
        ],
    )


def full_log_section(results: dict, cutoff: str) -> tuple[str, str, list[str]]:
    """Return the section of the early peak on every logged step."""
    replay = results["replay_q"]
    full_log = results["full_log"]
    full_rows = [
        [
            tag,
            step(replay["per_run"][tag]["peak_step"]),
            step(peak),
            str(len(full_log["later_above"].get(tag, []))),
        ]
        for tag, peak in full_log["peaks"].items()
    ]
    return (
        "Early peak on every logged step",
        "This repeats the peak on every logged step, not only the checkpoint steps. The early "
        f"peak is the highest logged value before {cutoff}. The last column counts the logged "
        f"values from {cutoff} on that lie above the early peak. Rows follow the config's run "
        "order.",
        [
            table(
                [
                    "Run",
                    "Peak step, checkpoints only",
                    "Early peak step, every logged step",
                    f"Logged values from {cutoff} on above the early peak",
                ],
                full_rows,
            ),
            bullets(
                [
                    "Runs whose early peak moves off the checkpoint peak: "
                    f"{full_log['moved_count']} of {len(full_log['peaks'])}"
                ]
            ),
        ],
    )


def critic_vs_return_section(results: dict, cutoff: str) -> tuple[str, str, list[str]]:
    """Return the section of the critic value against the return."""
    versus = results["critic_vs_return"]
    return_rows = [
        [step(peak_step), number(gap, Q_VALUE), percent(versus["early_scan"][peak_step])]
        for peak_step, gap in versus["early_gap"].items()
    ]
    overlap = results["overlap_with_heldout"]
    return (
        "Critic value against the return",
        "We compare the critic's mean value at the demonstration start states with the mean "
        "discounted return of the scan episodes, at every checkpoint from "
        f"{step(versus['from_step'])} on. The gap is the critic value minus the return, in "
        f"reward units. The table gives medians over the {versus['median_runs']} runs other "
        "than the approximate rerun, at each held-out Q peak step. Rows are sorted by step.",
        [
            table(["Step", "Median gap (reward units)", "Median scan success (%)"], return_rows),
            bullets(
                [
                    f"Checkpoints where the critic value is above the return: "
                    f"{versus['above']} of {versus['checkpoints']}",
                    f"Runs whose gap shrinks from {cutoff} to the final checkpoint: "
                    f"{versus['shrink_count']} of {versus['runs']}",
                    "Median gap at the final checkpoint: "
                    f"{number(versus['final_gap'], Q_VALUE)} reward units",
                    "Earlier runs with both a training log and held-out statistics: "
                    f"{overlap['count']} ({tags(overlap['tags'])})",
                ]
            ),
        ],
    )


def critic_page(results: dict, config: dict) -> str:
    """Return the page of held-out Q and replay-batch Q."""
    cutoff = step(config["selectors"]["restricted_min_step"])
    sections = [
        heldout_peaks_section(results, config, cutoff),
        heldout_by_step_section(results),
        control_heldout_section(results, cutoff),
        replay_section(results, cutoff),
        replay_by_logged_step_section(results),
        kumar_section(results, config),
        full_log_section(results, cutoff),
        critic_vs_return_section(results, cutoff),
    ]
    return page(
        "Critic values as checkpoint selectors",
        "We read the critic's value on held-out demonstration states and on replay batches, and "
        "compare it with the success of the checkpoints. Critic values are in reward units: "
        "each decision before success earns a reward of -1.",
        sections,
    )
