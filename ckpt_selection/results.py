"""Compute the results of every experiment from the run outputs; write and print them.

Run from the repo root (CPU only):

    python -m ckpt_selection.results [--outputs DIR]

Writes results/<experiment>.md (Markdown tables) for each experiment and results/results.json
(the same numbers at full precision, keyed by experiment and quantity) under the outputs folder,
then prints each Markdown file. Per-checkpoint and per-step series are keyed by run tag, then by
gradient step.
"""

import argparse
from pathlib import Path

from ckpt_selection import config as config_lib
from ckpt_selection import (
    confirm_set,
    critic,
    heldout,
    layout,
    load,
    markdown_confirm_sets,
    markdown_critic,
    markdown_heldout_stats,
    markdown_practice,
    markdown_regret,
    markdown_scan_bias,
    markdown_screen,
    markdown_softopc,
    practice,
    regret,
    scan_bias,
    screening,
    selectors,
)
from ckpt_selection import softopc as softopc_analysis

RESULTS_NAME = "results.json"


def merge_into(section: dict, values: dict, source: str) -> None:
    """Add an analysis result's keys to a section; fail if a key is already there."""
    clash = sorted(set(section) & set(values))
    if clash:
        raise KeyError(f"{source} returns keys already in the section: {clash}")
    section.update(values)


def screen_results(config: dict, outputs: Path, runs: dict) -> dict:
    """Return the base-policy task screen, the two-stage recheck and the drop screen."""
    drop_runs = [run for run in runs.values() if run.role == "drop_screen"]
    tasks = screening.screening_table(config, outputs)
    return {
        "band": tasks["band"],
        "tasks": tasks["rows"],
        "qualifying": tasks["qualifying"],
        "two_stage_recheck": screening.two_stage_recheck(config, outputs),
        "drop_screen": {
            "run_count": len(config_lib.run_tags(config, ["drop_screen"])),
            **screening.drop_screen(drop_runs, config),
        },
    }


def critic_results(config: dict, runs: dict) -> dict:
    """Return held-out Q on the earlier and control runs and replay-batch Q from the logs.

    The per-checkpoint held-out Q and the replay-batch Q at every logged step are included.
    """
    heldout_runs = load.with_stage(runs, "heldout", role="earlier")
    control_runs = load.with_stage(runs, "heldout", role="control")
    log_runs = load.with_stage(runs, "train_log")
    min_step = config["selectors"]["restricted_min_step"]
    heldout_q = {}
    merge_into(heldout_q, critic.heldout_q_peaks(heldout_runs, config), "critic.heldout_q_peaks")
    merge_into(
        heldout_q, critic.step_correlations(heldout_runs, min_step), "critic.step_correlations"
    )
    merge_into(heldout_q, critic.peak_below_base(heldout_runs, config), "critic.peak_below_base")
    summary = critic.replay_q_summary(log_runs, config)
    replay_q = {
        "log_interval": critic.log_interval(log_runs),
        "per_run": summary.pop("replay_q"),
        "by_logged_step": critic.replay_q_by_logged_step(log_runs),
    }
    merge_into(replay_q, summary, "critic.replay_q_summary")
    return {
        "run_counts": {
            "heldout_earlier": len(heldout_runs),
            "heldout_control": len(control_runs),
            "train_log": len(log_runs),
        },
        "heldout_q": heldout_q,
        "heldout_q_by_step": critic.heldout_q_by_step(heldout_runs),
        "control_heldout_q": critic.control_heldout_q(control_runs, config),
        "replay_q": replay_q,
        "full_log": critic.full_log_peaks(log_runs, config),
        "critic_vs_return": critic.critic_vs_return(heldout_runs, config),
        "overlap_with_heldout": critic.overlap_with_heldout(runs),
    }


def lowest_regret_selectors(selector_table: dict) -> dict:
    """Return, per regret summary, the rollout-free selectors with the lowest value."""
    candidates = {key: row for key, row in selector_table.items() if row["kind"] == "rollout_free"}
    lowest = {}
    for column in ("mean", "median", "worst"):
        best = min(row[column] for row in candidates.values())
        lowest[column] = [key for key, row in candidates.items() if row[column] == best]
    return lowest


def regret_results(config: dict, runs: dict) -> dict:
    """Return the regret of each selector and of keeping the final checkpoint on every run."""
    heldout_runs = load.with_stage(runs, "heldout", role="earlier")
    floor = config["protocol"]["regret_floor"]
    selector_table = regret.selector_regret_table(heldout_runs, config)
    r_last_by_tag = selector_table["final"]["per_run"]
    final_picks = selector_table["final"]["picks"]
    veto_picks = selector_table["spread_veto"]["picks"]
    restricted = regret.compare_to_final(
        selector_table["restricted_critic"]["per_run"], r_last_by_tag, floor
    )

    trained = [run for run in runs.values() if run.role in regret.TRAINED_ROLES]
    task8 = [run for run in trained if run.role in regret.TASK8_ROLES]
    controls = [run for run in trained if run.role == "control"]
    final_table = regret.final_regret_table(trained, config)
    noncompetent = {run.tag: regret.non_competent_drops(run, config) for run in task8}
    noncompetent = {tag: drops for tag, drops in noncompetent.items() if drops}
    sensitivity_runs = {
        row["tag"]: row["r_last"]
        for row in final_table["rows"]
        if row["role"] in regret.SENSITIVITY_ROLES
    }
    collapse_tags = {row["tag"] for row in final_table["rows"] if row["collapses"]}
    run_counts = {role: len(config_lib.run_tags(config, [role])) for role in regret.TRAINED_ROLES}
    run_counts["task8"] = len(config_lib.run_tags(config, list(regret.TASK8_ROLES)))
    return {
        "run_counts": run_counts,
        "selectors": selector_table,
        "lowest_regret_selectors": lowest_regret_selectors(selector_table),
        "restricted_critic": {
            "worse": restricted["worse"],
            "worse_over_floor": restricted["worse_over_floor"],
            "difference": restricted["difference"],
            "at_cutoff": regret.restricted_at_cutoff(selector_table, config),
        },
        "spread_veto_final_count": sum(veto_picks[tag] == final_picks[tag] for tag in final_picks),
        "lowest_entropy_vs_final": heldout.entropy_vs_final(heldout_runs, selector_table, config),
        "final_checkpoint": final_table,
        "task8_r_last": regret.task8_r_last(task8),
        "scan_success_by_step": regret.scan_success_by_step(task8),
        "terminal_drops": regret.terminal_drops(task8, config),
        "control_collapse_count": sum(len(regret.collapses(run, config)) for run in controls),
        "control_scan_falls": {run.tag: regret.largest_scan_fall(run) for run in controls},
        "control_best_confirmed": regret.best_confirmed_range(controls),
        "oracle_bias": regret.oracle_bias(task8, config),
        "unconfirmed_above_oracle": regret.unconfirmed_above_oracle(task8),
        "median_sensitivity": regret.median_sensitivity(sensitivity_runs),
        "rerun_pairs": regret.rerun_pairs(runs),
        "noncompetent_drops": noncompetent,
        "collapses_if_counted": {
            "events": final_table["collapse_events"]
            + sum(len(drops) for drops in noncompetent.values()),
            "runs": len(collapse_tags | set(noncompetent)),
        },
    }


def heldout_stats_results(config: dict, runs: dict) -> dict:
    """Return the held-out statistics through training and L2, KL and MMD as selectors."""
    heldout_runs = load.with_stage(runs, "heldout", role="earlier")
    control_runs = load.with_stage(runs, "heldout", role="control")
    base = load.base_heldout(runs, heldout_runs[0].task)
    return {
        "summary": heldout.heldout_summary_table(heldout_runs, base, config),
        "l2_above_base": heldout.l2_above_base(heldout_runs, base, config),
        "spread_at_collapse": heldout.spread_at_collapsed_final(heldout_runs),
        "collapse_l2": heldout.l2_at_largest_final_drop(heldout_runs, base),
        "control_l2_band": heldout.control_l2_band(control_runs, runs),
        "distance_rules": heldout.distance_variants(heldout_runs, config),
        "by_step": heldout.statistics_by_step(heldout_runs),
        "confirmed_by_step": heldout.confirmed_by_step(heldout_runs),
    }


def softopc_results(config: dict, runs: dict) -> dict:
    """Return the pool balance of every pooled run, and SoftOPC and the other signals on one run.

    That run is the one pooled run whose pool holds enough successes and failures.
    """
    pooled = load.with_stage(runs, "softopc")
    pools = {run.tag: softopc_analysis.pool_balance(run, config) for run in pooled}
    eligible = [run for run in pooled if pools[run.tag]["eligible"]]
    if len(eligible) != 1:
        raise ValueError(f"SoftOPC needs one run with an eligible pool, found {len(eligible)}")
    run = eligible[0]
    rows = softopc_analysis.signal_table(run, config)
    falls = softopc_analysis.confirmed_falls_with_rising_score(run)
    window = config["selectors"]["practice_window"]
    return {
        "pool_count": len(pooled),
        "pools": pools,
        "run": run.tag,
        "confirmed_count": len(run.confirm),
        "practice_empty_steps": selectors.empty_practice_windows(run, window),
        "signals": rows,
        "checks": softopc_analysis.table_checks(rows),
        "pick": softopc_analysis.softopc_pick(run),
        "falls_with_rising_score": falls,
        "falls_with_rising_score_count": len(falls),
        "scores": softopc_analysis.checkpoint_scores(run),
    }


def scan_bias_results(config: dict, runs: dict) -> dict:
    """Return scan minus confirmed success on the runs whose confirm rule is [scan_bias]'s."""
    rule = config["scan_bias"]["confirm_rule"]
    rule_runs = [run for run in runs.values() if run.entry["confirm_rule"] == rule]
    results = {"n_runs": len(rule_runs)}
    merge_into(results, scan_bias.scan_bias_summary(rule_runs, config), "scan_bias_summary")
    return results


def practice_results(config: dict, runs: dict) -> dict:
    """Return selection on S_W in the replication runs and the window choice of S_W."""
    replication = [run for run in runs.values() if run.role == "replication"]
    choice_runs = [
        run
        for run in runs.values()
        if run.role != "replication" and run.entry["practice_source"] is not None
    ]
    table = practice.replication_table(replication, config)
    uncontrolled = practice.uncontrolled_vs_step(replication, config)
    before_competence = {
        run.tag: practice.checkpoints_before_competence(run) for run in replication
    }
    selection = practice.selection_changes(table, config)
    window = config["selectors"]["practice_window"]
    return {
        "run_count": len(config_lib.run_tags(config, ["replication"])),
        "replication": table,
        "ranking": practice.ranking_criteria(table, uncontrolled, config),
        "selection": selection,
        "moved_series": {
            row["run"]: practice.selection_series(runs[row["run"]], window)
            for row in selection["moved"]
        },
        "expectation": practice.expectation_check(table, config),
        "uncontrolled": uncontrolled,
        "windows": practice.window_sensitivity(replication, config),
        "window_choice": {run.tag: practice.window_choice(run, config) for run in choice_runs},
        "before_competence": before_competence,
        "before_competence_min": min(before_competence.values()),
    }


def confirm_sets_results(config: dict, runs: dict) -> dict:
    """Return, per confirmed run, its confirm set rebuilt by its rule against what it confirmed."""
    confirmed_runs = [run for run in runs.values() if run.entry["confirm_rule"] is not None]
    checks = {
        run.tag: {"role": run.role, **confirm_set.rebuild_check(run, config)}
        for run in confirmed_runs
    }
    task8_counts = [len(run.confirm) for run in confirmed_runs if run.role in regret.TASK8_ROLES]
    return {
        "runs": checks,
        "run_count": len(checks),
        "matching_count": sum(check["matches"] for check in checks.values()),
        "task8_confirmed_min": min(task8_counts),
        "task8_confirmed_max": max(task8_counts),
    }


def all_results(config: dict, outputs: Path) -> dict:
    """Run every analysis and return the results, keyed by experiment."""
    runs = load.load_runs(config, outputs)
    return {
        "screen": screen_results(config, outputs, runs),
        "critic": critic_results(config, runs),
        "regret": regret_results(config, runs),
        "heldout_stats": heldout_stats_results(config, runs),
        "softopc": softopc_results(config, runs),
        "scan_bias": scan_bias_results(config, runs),
        "practice": practice_results(config, runs),
        "confirm_sets": confirm_sets_results(config, runs),
    }


def all_pages(results: dict, config: dict) -> dict[str, str]:
    """Return the Markdown page of every experiment, keyed by experiment name."""
    writers = {
        "screen": markdown_screen.screen_page,
        "critic": markdown_critic.critic_page,
        "regret": markdown_regret.regret_page,
        "heldout_stats": markdown_heldout_stats.heldout_stats_page,
        "softopc": markdown_softopc.softopc_page,
        "scan_bias": markdown_scan_bias.scan_bias_page,
        "practice": markdown_practice.practice_page,
        "confirm_sets": markdown_confirm_sets.confirm_sets_page,
    }
    return {name: writer(results[name], config) for name, writer in writers.items()}


def main() -> None:
    """Compute every result, write the Markdown and JSON files, and print the Markdown files."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outputs", default=None, help="outputs folder (default from config)")
    args = parser.parse_args()
    config = config_lib.load_config()
    outputs = config_lib.outputs_root(config, args.outputs)

    results = all_results(config, outputs)
    folder = layout.results_dir(outputs)
    layout.write_json(folder / RESULTS_NAME, results)
    for experiment, page in all_pages(results, config).items():
        path = folder / f"{experiment}.md"
        path.write_text(page)
        print(page)


if __name__ == "__main__":
    main()
