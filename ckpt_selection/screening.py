"""Base-policy task screen and screening for drops, reported in App. A "Task selection".

Reads the screen files under outputs/screen/ and the scans of the runs screened for drops. A
drop is a fall in scan success between consecutive saved checkpoints, and it counts when the
next checkpoint rises again, as for the confirmed drops of regret.confirmed_drops.
"""

from pathlib import Path

from ckpt_selection import layout, load, regret

MAIN_TASK_LABEL = "main task"
CONTROL_TASK_LABEL = "control task"
EXCLUDED_LABEL = "excluded"  # role of a task group whose rates all lie outside the band


def screen_stages(config: dict) -> dict:
    """Return {suite: number of stages} for every suite in [screen]."""
    return {
        suite: len(entry["stages"])
        for suite, entry in config["screen"].items()
        if isinstance(entry, dict) and "stages" in entry
    }


def read_screen(outputs: Path, suite: str, stage: int) -> dict:
    """Return {task: success_rate} of one screen stage."""
    data = layout.read_json(layout.screen_path(outputs, suite, stage))
    return {row["task"]: row["success_rate"] for row in data["tasks"]}


def screen_episodes(outputs: Path, suite: str, stage: int) -> int:
    """Return the number of episodes per task that one screen stage ran."""
    return layout.read_json(layout.screen_path(outputs, suite, stage))["episodes"]


def in_band(rate: float, band: list) -> bool:
    """Return whether a base-policy success rate lies inside the eligibility band (inclusive)."""
    return band[0] <= rate <= band[1]


def task_role(config: dict, suite: str, task: int) -> str | None:
    """Return the role of a screened task: 'main task', 'control task' or None."""
    roles = {
        entry["role"]
        for entry in config["runs"]
        if entry["suite"] == suite and entry["task"] == task
    }
    if roles & set(regret.TASK8_ROLES):
        return MAIN_TASK_LABEL
    if "control" in roles:
        return CONTROL_TASK_LABEL
    return None


def screening_table(config: dict, outputs: Path) -> dict:
    """Return base-policy success per task with a run, the tasks inside the band, and the band.

    There is one row per task that has a run, with its base-policy success at its suite's last
    screen stage. A suite whose last stage covers all its tasks also gets a range row for its
    other tasks and a list of its tasks inside the band.
    """
    band = config["screen"]["band"]
    rows = []
    qualifying = {}
    for suite, stages in screen_stages(config).items():
        final = read_screen(outputs, suite, stages)
        # Later stages rerun only the tasks in band, so only a one-stage screen covers the suite.
        covers_suite = final.keys() == read_screen(outputs, suite, 1).keys()
        suite_runs = [entry for entry in config["runs"] if entry["suite"] == suite]
        run_tasks = list(dict.fromkeys(entry["task"] for entry in suite_runs))
        screened_for_drops = {
            entry["task"] for entry in suite_runs if entry["role"] == "drop_screen"
        }
        for task in run_tasks:
            rows.append(
                {
                    "suite": suite,
                    "tasks": str(task),
                    "success_min": final[task],
                    "success_max": final[task],
                    "screened_for_drops": task in screened_for_drops,
                    "role": task_role(config, suite, task),
                    "in_band": in_band(final[task], band),
                }
            )
        others = [task for task in final if task not in run_tasks]
        if covers_suite and others:
            rates = [final[task] for task in others]
            any_in_band = any(in_band(rate, band) for rate in rates)
            rows.append(
                {
                    "suite": suite,
                    "tasks": f"other {len(others)}",
                    "success_min": min(rates),
                    "success_max": max(rates),
                    "screened_for_drops": False,
                    "role": None if any_in_band else EXCLUDED_LABEL,
                    "in_band": any_in_band,
                }
            )
        if covers_suite:
            qualifying[suite] = sorted(task for task, rate in final.items() if in_band(rate, band))
    return {"rows": rows, "qualifying": qualifying, "band": band}


def two_stage_recheck(config: dict, outputs: Path) -> dict:
    """Return, per two-stage suite, the tasks in band at stage 1 and those that stay in at stage 2.

    Stage 2 reruns the stage-1 tasks inside the band on new episodes. The episode counts come
    from the screen files.
    """
    band = config["screen"]["band"]
    recheck = {}
    for suite, stages in screen_stages(config).items():
        if stages != 2:
            continue
        first = read_screen(outputs, suite, 1)
        second = read_screen(outputs, suite, 2)
        first_in_band = sorted(task for task, rate in first.items() if in_band(rate, band))
        stay = {task: second[task] for task in first_in_band if in_band(second[task], band)}
        recheck[suite] = {
            "stage1_tasks": len(first),
            "stage1_episodes": screen_episodes(outputs, suite, 1),
            "stage2_episodes": screen_episodes(outputs, suite, 2),
            "stage1_in_band": first_in_band,
            "stage1_in_band_count": len(first_in_band),
            "stage2_stay_in_band": dict(sorted(stay.items(), key=lambda item: item[1])),
            "stage2_stay_in_band_count": len(stay),
        }
    return recheck


def drop_screen(runs: list[load.RunData], config: dict) -> dict:
    """Return the largest scan drop of each run screened for drops, and the flagged tasks.

    A drop is a fall in scan success between consecutive saved checkpoints that the next
    checkpoint rises from again. A task is flagged if its largest drop exceeds the threshold.
    """
    threshold = config["protocol"]["drop_points"]
    per_run = {}
    for run in runs:
        steps = sorted(step for step in run.scan if step != layout.BASE_STEP)
        largest = {"drop": 0.0}
        for before, step, after in zip(steps, steps[1:], steps[2:]):
            drop = regret.points(run.scan[before] - run.scan[step])
            if run.scan[after] > run.scan[step] and drop > largest["drop"]:
                largest = {"drop": drop, "from_step": before, "to_step": step}
        per_run[run.tag] = {"task": run.task, **largest, "flagged": largest["drop"] > threshold}
    flagged = sorted({entry["task"] for entry in per_run.values() if entry["flagged"]})
    return {"per_run": per_run, "flagged_tasks": flagged}
