"""Base-policy task screen of the LIBERO suites, in stages (App. A "Task selection").

Stage 1 runs pi0 with i.i.d. noise on every task of a suite. Each later stage reruns, on new
seeds, the tasks whose success rate at the stage before lies inside the band. The suites, seed
pools, episode counts and the band come from [screen] in config/experiments.toml.

The file screen/<suite>_stage<N>.json is rewritten after every task, and tasks already in it are
skipped, so an interrupted screen resumes where it stopped.

    python -m ckpt_selection.screen [--suite SUITE] [--stage N] [--outputs DIR]

Without --suite, every suite in [screen] runs. Without --stage, every stage of a suite runs, in
order. --stage N alone runs stage N of each suite that has one.
"""

import argparse
from pathlib import Path

from ckpt_selection import agent as agent_lib
from ckpt_selection import config as config_lib
from ckpt_selection import layout
from ckpt_selection.evaluate import run_episodes
from ckpt_selection.screening import in_band, screen_stages


def stage_settings(config: dict, suite: str, stage: int) -> dict:
    """Return the seed base and episode count of one stage (stages count from 1)."""
    return config["screen"][suite]["stages"][stage - 1]


def stage_tasks(config: dict, outputs: Path, suite: str, stage: int) -> list[int]:
    """Return a stage's tasks: all tasks at stage 1, else the stage before's tasks in band."""
    if stage == 1:
        return list(range(agent_lib.suite_size(suite)))
    previous = layout.screen_path(outputs, suite, stage - 1)
    if not previous.exists():
        raise FileNotFoundError(f"{previous} is missing; run stage {stage - 1} first")
    band = config["screen"]["band"]
    rows = layout.read_json(previous)["tasks"]
    return [row["task"] for row in rows if in_band(row["success_rate"], band)]


def load_stage_results(
    config: dict, outputs: Path, suite: str, stage: int, task_ids: list[int]
) -> dict:
    """Return a stage's results, holding the tasks that an earlier, interrupted screen finished."""
    settings = stage_settings(config, suite, stage)
    results = {
        "suite": suite,
        "stage": stage,
        "seed_base": settings["seed_base"],
        "episodes": settings["episodes"],
        "episode_limit": config["evaluation"]["episode_limit"],
        "tasks": [],
    }
    path = layout.screen_path(outputs, suite, stage)
    if path.exists():
        previous = layout.read_json(path)
        if any(previous[key] != results[key] for key in results if key != "tasks"):
            raise ValueError(f"{path} was made with other settings; move it away to rescreen")
        results["tasks"] = [row for row in previous["tasks"] if row["task"] in task_ids]
    return results


def screen_tasks(pi0, config: dict, outputs: Path, results: dict, task_ids: list[int]) -> None:
    """Run the base policy on each task not yet in the results and write the stage file.

    The file is written after every task, and once more at the end so that a stage with no task
    left to run (or no task inside the band) still gets its file.
    """
    path = layout.screen_path(outputs, results["suite"], results["stage"])
    done = {row["task"] for row in results["tasks"]}
    for task_id in task_ids:
        if task_id in done:
            continue
        env, task = agent_lib.make_libero_env(
            results["suite"], task_id, config["evaluation"]["render_resolution"]
        )
        variant = agent_lib.base_variant(config, str(task.language))
        print(f"{results['suite']} task {task_id}: {task.language}", flush=True)
        try:
            rows = run_episodes(
                pi0,
                None,
                env,
                variant,
                results["seed_base"],
                results["episodes"],
                results["episode_limit"],
            )
        finally:
            env.close()
        rate = sum(row["success"] for row in rows) / len(rows)
        results["tasks"].append(
            {"task": task_id, "language": str(task.language), "success_rate": rate}
        )
        results["tasks"].sort(key=lambda row: row["task"])
        layout.write_json(path, results)
    layout.write_json(path, results)


def main() -> None:
    """Screen the requested stages, loading pi0 only if some task still needs episodes."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", default=None, help="one suite with a [screen.<suite>] table")
    parser.add_argument("--stage", default=None, type=int, help="one stage number, from 1")
    parser.add_argument("--outputs", default=None, help="outputs folder (default: from config)")
    args = parser.parse_args()

    config = config_lib.load_config()
    outputs = config_lib.outputs_root(config, args.outputs)
    stage_counts = screen_stages(config)
    if args.suite is not None:
        if args.suite not in stage_counts:
            parser.error(f"no [screen.{args.suite}] table in the config")
        stage_counts = {args.suite: stage_counts[args.suite]}
    if args.stage is not None:
        stage_counts = {
            suite: count for suite, count in stage_counts.items() if args.stage <= count
        }
        if args.stage < 1 or not stage_counts:
            parser.error(f"no requested suite has a screening stage {args.stage}")
    pi0 = None
    for suite, count in stage_counts.items():
        stages = [args.stage] if args.stage is not None else range(1, count + 1)
        for stage in stages:
            task_ids = stage_tasks(config, outputs, suite, stage)
            results = load_stage_results(config, outputs, suite, stage, task_ids)
            finished = len(results["tasks"])
            print(f"{suite} stage {stage}: {finished} of {len(task_ids)} tasks done", flush=True)
            if finished < len(task_ids) and pi0 is None:
                pi0 = agent_lib.load_pi0(config)
            screen_tasks(pi0, config, outputs, results, task_ids)


if __name__ == "__main__":
    main()
