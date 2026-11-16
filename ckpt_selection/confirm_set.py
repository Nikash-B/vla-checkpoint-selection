"""Choose the checkpoints of one run to confirm (Algorithm 1, line 4; App. B "Confirm sets").

Every confirm set holds the final checkpoint and the highest-scanning checkpoint. The rest depends
on the run's rule, a key of [confirm_rules] in the config:

    around_drops  the rollout-free picks, the three checkpoints around every drop of more than
                  [protocol] drop_points between consecutive scans, and the run's top_scan
                  highest-scanning checkpoints; no cap
    late_drops    the spread veto, entropy and held-out Q picks, then checkpoints from start_step
                  on that scan more than drop_points below the final checkpoint, capped at cap;
                  then the restricted-critic pick
    practice      the S_W pick, then the late drops as above, capped at cap
    control       the held-out selectors' picks; no drops and no cap
    listed        the run's confirm_steps from the config, chosen by hand

Each run names its rule in config/experiments.toml. rebuild_check compares the rebuilt set with
the checkpoints a run confirmed and lists every difference; confirm_sets.md reports them per run.
The analysis reads the confirm files that exist, so a difference changes no other result.

Usage: python -m ckpt_selection.confirm_set --run TAG [--outputs DIR]
"""

import argparse

from ckpt_selection import config as config_lib
from ckpt_selection import layout, load, selectors

# Order in which each rule adds the rollout-free picks; under a cap, earlier entries win.
# The around_drops and control rules have no cap and add every pick that applies to the run.
UNCAPPED_PICKS = ("heldout_q", "entropy", "spread_veto", "restricted_critic", "practice_success")
LATE_DROPS_PICKS = ("spread_veto", "entropy", "heldout_q")
PRACTICE_PICKS = ("practice_success",)


def drop_in_points(higher: float, lower: float) -> float:
    """Return the fall from one success rate to another in points, free of float residue."""
    return round(100 * (higher - lower), 9)


def around_drops(scan: dict, steps: list[int], drop_points: float) -> list[int]:
    """Return the checkpoints around every scan drop of more than drop_points.

    For consecutive checkpoints a and b whose scan success falls by more than drop_points, add
    a, b and the checkpoint after b (when there is one).
    """
    added = []
    for index in range(len(steps) - 1):
        if drop_in_points(scan[steps[index]], scan[steps[index + 1]]) > drop_points:
            added.extend(steps[index : index + 3])
    return added


def late_drops(scan: dict, steps: list[int], start_step: int, drop_points: float) -> list[int]:
    """Return the checkpoints from start_step on that scan more than drop_points below the final."""
    final_rate = scan[steps[-1]]
    return [
        step
        for step in steps
        if step >= start_step and drop_in_points(final_rate, scan[step]) > drop_points
    ]


def top_scanning(scan: dict, steps: list[int], count: int, tie: str) -> list[int]:
    """Return the count highest-scanning checkpoints; ties go to the earliest or latest step."""
    direction = 1 if tie == "earliest" else -1
    return sorted(steps, key=lambda step: (-scan[step], direction * step))[:count]


def add_candidates(chosen: dict, candidates: list[tuple[int, str]], cap: int | None) -> None:
    """Add (step, reason) candidates in order to chosen {step: [reasons]}.

    A step already chosen gains the reason. Once chosen holds cap steps, new steps are skipped.
    """
    for step, reason in candidates:
        if step in chosen:
            if reason not in chosen[step]:
                chosen[step].append(reason)
        elif cap is None or len(chosen) < cap:
            chosen[step] = [reason]


def build_confirm_set(run, config: dict) -> dict[int, list[str]]:
    """Return the run's confirm set by its rule, as {step: reasons} in step order."""
    rule_name = run.entry["confirm_rule"]
    rule = config["confirm_rules"][rule_name]
    drop_points = config["protocol"]["drop_points"]
    steps = selectors.candidate_steps(run)
    tie = config["selectors"]["tie_rule"]
    picks = selectors.rollout_free_picks(run, config)
    candidates = [
        (selectors.final_step(run), "final"),
        (selectors.scan_argmax(run, tie), "scan_argmax"),
    ]
    chosen = {}
    if rule_name == "around_drops":
        candidates += [(picks[name], name) for name in UNCAPPED_PICKS if name in picks]
        candidates += [(step, "drop") for step in around_drops(run.scan, steps, drop_points)]
        top = top_scanning(run.scan, steps, run.entry["top_scan"], tie)
        candidates += [(step, "top_scan") for step in top]
        add_candidates(chosen, candidates, None)
    elif rule_name in ("late_drops", "practice"):
        names = LATE_DROPS_PICKS if rule_name == "late_drops" else PRACTICE_PICKS
        candidates += [(picks[name], name) for name in names]
        drops = late_drops(run.scan, steps, rule["start_step"], drop_points)
        candidates += [(step, "drop") for step in drops]
        add_candidates(chosen, candidates, rule["cap"])
        if rule_name == "late_drops":
            add_candidates(chosen, [(picks["restricted_critic"], "restricted_critic")], None)
    elif rule_name == "control":
        candidates += [(picks[name], name) for name in UNCAPPED_PICKS if name in picks]
        add_candidates(chosen, candidates, None)
    elif rule_name == "listed":
        candidates += [(step, "listed") for step in run.entry["confirm_steps"]]
        add_candidates(chosen, candidates, None)
    else:
        raise ValueError(f"run {run.tag}: unknown confirm rule {rule_name!r}")
    return {step: chosen[step] for step in sorted(chosen)}


def rebuild_check(run, config: dict) -> dict:
    """Compare the confirm set that the run's rule gives with the checkpoints it confirmed."""
    rebuilt = set(build_confirm_set(run, config))
    confirmed = set(run.confirm)
    return {
        "rule": run.entry["confirm_rule"],
        "confirmed_count": len(confirmed),
        "rule_only": sorted(rebuilt - confirmed),
        "confirmed_only": sorted(confirmed - rebuilt),
        "matches": rebuilt == confirmed,
    }


def main() -> None:
    """Write confirm_set.json for one run from its scan and, where it has them, held-out files."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="run tag from config/experiments.toml")
    parser.add_argument("--outputs", default=None, help="outputs folder (default from the config)")
    args = parser.parse_args()

    config = config_lib.load_config()
    outputs = config_lib.outputs_root(config, args.outputs)
    run = load.load_run(config, outputs, args.run)
    reasons = build_confirm_set(run, config)
    layout.write_json(
        layout.confirm_set_path(outputs, args.run),
        {
            "run_tag": args.run,
            "rule": run.entry["confirm_rule"],
            "steps": list(reasons),
            "reasons": {str(step): step_reasons for step, step_reasons in reasons.items()},
        },
    )
    for step, step_reasons in reasons.items():
        print(f"{args.run} confirm {step}: {', '.join(step_reasons)}")


if __name__ == "__main__":
    main()
