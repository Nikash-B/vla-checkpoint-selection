"""Read the outputs of each run into one RunData object for the CPU analyses."""

from dataclasses import dataclass, field
from pathlib import Path

from ckpt_selection import config as config_lib
from ckpt_selection import layout, training_log

# Which RunData field must be non-empty when a run's config lists the stage. The confirm_set
# stage writes no RunData field, so with_stage does not accept it.
STAGE_FIELDS = {
    "scan": "scan",
    "heldout": "heldout",
    "confirm": "confirm",
    "softopc": "pool_meta",
    "train_log": "train_log",
}


@dataclass
class RunData:
    """All outputs of one run.

    The per-step dicts (scan, confirm, heldout, softopc_scores) are keyed by gradient step;
    step -1 is the base policy.
    """

    tag: str
    task: int
    role: str
    rerun_of: str | None
    entry: dict
    scan: dict = field(default_factory=dict)
    scan_rows: dict = field(default_factory=dict)
    confirm: dict = field(default_factory=dict)
    confirm_rows: dict = field(default_factory=dict)
    heldout: dict = field(default_factory=dict)
    practice: list = field(default_factory=list)
    train_log: list = field(default_factory=list)
    softopc_scores: dict = field(default_factory=dict)
    pool_meta: dict = field(default_factory=dict)


def read_pool(outputs: Path, tag: str, pool: str) -> tuple[dict, dict]:
    """Return {step: success_rate} and {step: rows} of every episode file in one pool."""
    rates = {}
    rows = {}
    for step in layout.evaluation_steps(outputs, tag, pool):
        data = layout.read_json(layout.evaluation_path(outputs, tag, pool, step))
        rates[step] = data["success_rate"]
        rows[step] = data["rows"]
    return rates, rows


def read_practice(outputs: Path, tag: str, source: str | None, train_log: list) -> list:
    """Return the run's practice outcomes as (grad_step, success) pairs in log order."""
    if source == "train_log":
        step_key = training_log.STEP_KEY
        practice_key = training_log.PRACTICE_KEY
        return [(row[step_key], int(row[practice_key])) for row in train_log]
    if source == "episodes":
        path = layout.practice_log_path(outputs, tag)
        if not path.exists():
            raise FileNotFoundError(f"run {tag} has practice_source 'episodes' but no {path}")
        return [(row["grad_step"], int(row["success"])) for row in layout.read_jsonl(path)]
    if source is None:
        return []
    raise ValueError(f"run {tag}: unknown practice_source {source!r}")


def load_run(config: dict, outputs: Path, tag: str) -> RunData:
    """Read every output file that exists for one run; missing parts stay empty."""
    entry = config_lib.run_entry(config, tag)
    run = RunData(
        tag=tag,
        task=entry["task"],
        role=entry["role"],
        rerun_of=entry["rerun_of"],
        entry=entry,
    )
    run.scan, run.scan_rows = read_pool(outputs, tag, "scan")
    run.confirm, run.confirm_rows = read_pool(outputs, tag, "confirm")
    for step in layout.heldout_steps(outputs, tag):
        run.heldout[step] = layout.read_json(layout.heldout_path(outputs, tag, step))
    if layout.train_log_path(outputs, tag).exists():
        run.train_log = layout.read_json(layout.train_log_path(outputs, tag))
    run.practice = read_practice(outputs, tag, entry["practice_source"], run.train_log)
    if layout.softopc_scores_path(outputs, tag).exists():
        scores = layout.read_json(layout.softopc_scores_path(outputs, tag))["scores"]
        run.softopc_scores = {int(step): values for step, values in scores.items()}
    if layout.pool_meta_path(outputs, tag).exists():
        run.pool_meta = layout.read_json(layout.pool_meta_path(outputs, tag))
    return run


def load_runs(config: dict, outputs: Path) -> dict[str, RunData]:
    """Read every run in config order."""
    return {tag: load_run(config, outputs, tag) for tag in config_lib.run_tags(config)}


def base_heldout(runs: dict[str, RunData], task: int) -> dict:
    """Return the base-policy held-out row of a task from the first run of that task that has it."""
    for run in runs.values():
        if run.task == task and layout.BASE_STEP in run.heldout:
            return run.heldout[layout.BASE_STEP]
    raise KeyError(f"no run of task {task} has a base-policy held-out file")


def with_stage(
    runs: dict[str, RunData],
    stage: str,
    role: str | None = None,
) -> list[RunData]:
    """Return the runs whose config lists `stage`, optionally only one role.

    The run set comes from the config, not from which files exist. A run that lists the stage but
    has no output for it raises an error, so a missing file never shrinks a run set silently.
    """
    if stage not in STAGE_FIELDS:
        raise ValueError(f"stage must be one of {sorted(STAGE_FIELDS)}, not {stage!r}")
    selected = []
    for run in runs.values():
        if stage not in run.entry["stages"]:
            continue
        if role is not None and run.role != role:
            continue
        if not getattr(run, STAGE_FIELDS[stage]):
            raise FileNotFoundError(f"run {run.tag} lists stage {stage!r} but has no output for it")
        selected.append(run)
    return selected
