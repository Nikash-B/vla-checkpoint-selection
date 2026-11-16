"""Read config/experiments.toml, the one file that holds every run, schedule and threshold.

Python modules import the functions below. Shell scripts call the small command-line interface:

    python -m ckpt_selection.config tags [--role ROLE ...]
    python -m ckpt_selection.config get SECTION.KEY
    python -m ckpt_selection.config train-flags TAG [--outputs DIR]
    python -m ckpt_selection.config stages TAG

Every command prints one value per line, so paths with spaces survive. The scripts read the lines
into a bash array with a `while IFS= read -r` loop (see scripts/train.sh).
"""

import argparse
import itertools
import tomllib
from pathlib import Path

from ckpt_selection import layout

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "experiments.toml"

REQUIRED_SECTIONS = (
    "paths",
    "upstream",
    "training",
    "evaluation",
    "screen",
    "heldout",
    "selectors",
    "protocol",
    "practice_criteria",
    "scan_bias",
    "softopc",
    "confirm_rules",
    "runs",
)
REQUIRED_RUN_KEYS = ("tag", "suite", "task", "seed", "role", "stages")
ROLES = ("earlier", "replication", "control", "drop_screen")
STAGES = ("scan", "heldout", "confirm_set", "confirm", "softopc", "train_log")
PRACTICE_SOURCES = ("train_log", "episodes")
RUN_DEFAULTS = {
    "confirm_rule": None,
    "top_scan": 0,
    "confirm_steps": None,
    "rerun_of": None,
    "max_steps": None,
    "practice_source": None,
}

# [training] keys that train_flags() sets per run instead of passing them through: the checkpoint
# schedule is cut at the run's length, and a run may override max_steps.
RUN_SPECIFIC_TRAINING_KEYS = ("checkpoint_steps", "max_steps")


def check_run(entry: dict, config: dict, tags: set) -> None:
    """Raise ValueError if one [[runs]] table is incomplete or inconsistent."""
    tag = entry.get("tag", "<no tag>")
    missing = [key for key in REQUIRED_RUN_KEYS if key not in entry]
    if missing:
        raise ValueError(f"run {tag}: missing keys {missing}")
    unknown = set(entry) - set(REQUIRED_RUN_KEYS) - set(RUN_DEFAULTS)
    if unknown:
        raise ValueError(f"run {tag}: unknown keys {sorted(unknown)}")
    if entry["role"] not in ROLES:
        raise ValueError(f"run {tag}: role {entry['role']!r} is not one of {ROLES}")
    if not entry["stages"]:
        raise ValueError(f"run {tag}: the stage list is empty")
    bad_stages = [stage for stage in entry["stages"] if stage not in STAGES]
    if bad_stages:
        raise ValueError(f"run {tag}: unknown stages {bad_stages}")
    order = [STAGES.index(stage) for stage in entry["stages"]]
    if order != sorted(order):
        raise ValueError(f"run {tag}: stages must follow the order {STAGES}")
    rule = entry.get("confirm_rule")
    if "confirm_set" in entry["stages"] and rule not in config["confirm_rules"]:
        raise ValueError(f"run {tag}: confirm_rule {rule!r} is not a key of [confirm_rules]")
    if rule is not None and "confirm_set" not in entry["stages"]:
        raise ValueError(f"run {tag}: confirm_rule needs the confirm_set stage")
    if entry.get("top_scan", 0) > 0 and rule != "around_drops":
        raise ValueError(f"run {tag}: top_scan applies only to confirm_rule 'around_drops'")
    listed = entry.get("confirm_steps")
    if (listed is not None) != (rule == "listed"):
        raise ValueError(
            f"run {tag}: confirm_steps goes with confirm_rule 'listed', and only there"
        )
    if listed is not None and not set(listed) <= set(config["training"]["checkpoint_steps"]):
        raise ValueError(f"run {tag}: confirm_steps must be checkpoint steps of [training]")
    if entry.get("rerun_of") is not None and entry["rerun_of"] not in tags:
        raise ValueError(f"run {tag}: rerun_of {entry['rerun_of']!r} is not a run tag")
    source = entry.get("practice_source")
    if source is not None and source not in PRACTICE_SOURCES:
        raise ValueError(f"run {tag}: practice_source {source!r} is not one of {PRACTICE_SOURCES}")
    if source == "train_log" and "train_log" not in entry["stages"]:
        raise ValueError(f"run {tag}: practice_source 'train_log' needs the train_log stage")


def check_seed_pools(config: dict) -> None:
    """Raise ValueError if two episode pools that must be disjoint share an episode seed.

    The scan, confirm and SoftOPC pools are pairwise disjoint (Sec. 3, App. D), and each stage of
    a suite's task screen runs new episodes (App. A). A pool covers the seeds from its seed_base
    on, one per episode. The SoftOPC pool may share seeds with a screen stage, which only picks
    tasks.
    """
    evaluation = config["evaluation"]
    softopc = config["softopc"]
    groups = [
        {
            "scan pool": (evaluation["scan"]["seed_base"], evaluation["scan"]["episodes"]),
            "confirm pool": (evaluation["confirm"]["seed_base"], evaluation["confirm"]["episodes"]),
            "SoftOPC pool": (softopc["pool_seed_base"], softopc["episodes_per_source"]),
        }
    ]
    for suite, entry in config["screen"].items():
        if isinstance(entry, dict) and "stages" in entry:
            groups.append(
                {
                    f"{suite} screen stage {number}": (stage["seed_base"], stage["episodes"])
                    for number, stage in enumerate(entry["stages"], start=1)
                }
            )
    for pools in groups:
        for (name_a, (base_a, count_a)), (name_b, (base_b, count_b)) in itertools.combinations(
            pools.items(), 2
        ):
            if max(base_a, base_b) < min(base_a + count_a, base_b + count_b):
                raise ValueError(f"the {name_a} and the {name_b} share episode seeds")


def load_config() -> dict:
    """Read the TOML file and check its sections and every [[runs]] table."""
    with open(CONFIG_PATH, "rb") as handle:
        config = tomllib.load(handle)
    missing = [section for section in REQUIRED_SECTIONS if section not in config]
    if missing:
        raise ValueError(f"{CONFIG_PATH}: missing sections {missing}")
    tags = [entry.get("tag") for entry in config["runs"]]
    duplicates = sorted({tag for tag in tags if tags.count(tag) > 1})
    if duplicates:
        raise ValueError(f"{CONFIG_PATH}: duplicate run tags {duplicates}")
    for entry in config["runs"]:
        check_run(entry, config, set(tags))
    check_seed_pools(config)
    return config


def run_entry(config: dict, tag: str) -> dict:
    """Return one [[runs]] table with the optional keys filled in."""
    for entry in config["runs"]:
        if entry["tag"] == tag:
            filled = dict(RUN_DEFAULTS)
            filled.update(entry)
            if filled["max_steps"] is None:
                filled["max_steps"] = config["training"]["max_steps"]
            return filled
    raise KeyError(f"no run with tag {tag!r} in the config")


def run_tags(config: dict, roles: list[str] | None = None) -> list[str]:
    """Return run tags in file order, optionally only those whose role is in roles."""
    return [entry["tag"] for entry in config["runs"] if roles is None or entry["role"] in roles]


def checkpoint_steps(config: dict, tag: str) -> list[int]:
    """Return the run's checkpoint steps: the schedule cut at the run's training length."""
    max_steps = run_entry(config, tag)["max_steps"]
    return [step for step in config["training"]["checkpoint_steps"] if step <= max_steps]


def outputs_root(config: dict, override: str | Path | None = None) -> Path:
    """Return the outputs directory: override if given, else [paths] outputs under the repo root."""
    if override is not None:
        return Path(override)
    return REPO_ROOT / config["paths"]["outputs"]


def flag_values(value) -> list[str]:
    """Format one config value as command-line words (a list becomes several words)."""
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def train_flags(config: dict, tag: str, outputs: Path) -> list[str]:
    """Return the command-line flags of dsrl_pi0/examples/launch_train_sim.py for one run."""
    entry = run_entry(config, tag)
    max_steps = entry["max_steps"]
    flags = [
        "--task_suite",
        entry["suite"],
        "--task_id",
        str(entry["task"]),
        "--seed",
        str(entry["seed"]),
        "--prefix",
        tag,
        "--output_dir",
        str(layout.train_dir(outputs, tag)),
        "--checkpoint_steps",
        ",".join(str(step) for step in checkpoint_steps(config, tag)),
        "--max_steps",
        str(max_steps),
    ]
    for key, value in config["training"].items():
        if key not in RUN_SPECIFIC_TRAINING_KEYS:
            flags.append(f"--{key}")
            flags.extend(flag_values(value))
    return flags


def lookup(config: dict, dotted_key: str):
    """Return the value at a dotted key such as 'upstream.openpi_commit'."""
    value = config
    for part in dotted_key.split("."):
        value = value[part]
    return value


def main() -> None:
    """Print config values for the shell scripts."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    tags_parser = commands.add_parser("tags", help="run tags, one per line")
    tags_parser.add_argument("--role", action="append", choices=ROLES)
    get_parser = commands.add_parser("get", help="one value; lists print one item per line")
    get_parser.add_argument("key")
    flags_parser = commands.add_parser(
        "train-flags", help="training flags of one run, one per line"
    )
    flags_parser.add_argument("tag")
    flags_parser.add_argument("--outputs", default=None)
    stages_parser = commands.add_parser("stages", help="evaluation stages of one run")
    stages_parser.add_argument("tag")
    args = parser.parse_args()

    config = load_config()
    if args.command == "tags":
        print("\n".join(run_tags(config, args.role)))
    elif args.command == "get":
        value = lookup(config, args.key)
        if isinstance(value, dict):
            parser.error(f"{args.key} is a table; ask for one of its keys")
        print("\n".join(flag_values(value)))
    elif args.command == "train-flags":
        outputs = outputs_root(config, args.outputs)
        print("\n".join(train_flags(config, args.tag, outputs)))
    elif args.command == "stages":
        print("\n".join(run_entry(config, args.tag)["stages"]))


if __name__ == "__main__":
    main()
