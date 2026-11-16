"""Paths of every run output, the JSON formats they hold, and the short name of a step.

No other module in ckpt_selection builds these paths, except agent.restore, which takes a
training folder and adds the checkpoint<step> name itself.

Layout under the outputs directory (default `outputs/`):

    screen/<suite>_stage<N>.json         base-policy task screen, one file per suite and stage
    <tag>/train/                         training run (dsrl_pi0 launch_train_sim.py --output_dir)
        variant.json                     full run configuration, read to rebuild the agent
        checkpoint<step>/                one flax checkpoint per scheduled step
        practice_episodes.jsonl          one line per practice episode
        <run name>/wandb/offline-run-*/run-*.wandb   offline W&B log of the training scalars
    <tag>/train.log                      stdout and stderr of training
    <tag>/train_log.json                 training scalars at every log interval
    <tag>/scan/scan_<step>.json          scan episodes; step -1 is the base policy
    <tag>/heldout/heldout_<step>.json    held-out statistics; step -1 is the base policy
    <tag>/heldout/base_prefixes.npz      sampled base-policy action prefixes (input to MMD)
    <tag>/confirm_set.json               checkpoints chosen for confirmation
    <tag>/confirm/confirm_<step>.json    confirm episodes
    <tag>/softopc/pool.npz               labeled SoftOPC pool rows
    <tag>/softopc/pool_meta.json         SoftOPC pool sources and counts
    <tag>/softopc/scores.json            SoftOPC and label-free mean Q of every checkpoint
    results/<experiment>.md              the results of one experiment as Markdown tables
    results/results.json                 the same numbers, keyed by experiment and quantity

JSON formats (rates are fractions in [0, 1]; steps are gradient steps):

    screen/<suite>_stage<N>.json: suite, stage, seed_base, episodes, episode_limit,
        tasks[{task, language, success_rate}]
    scan_<step>.json, confirm_<step>.json: run_tag, suite, task, step, pool, seed_base, episodes,
        episode_limit, agent_seed, success_rate, rows[{episode, seed, env_steps, success}]
    heldout_<step>.json: run_tag, step, n_states, samples_per_state, l2, action_entropy, kl, mmd,
        heldout_q, ensemble_spread, per_state[{demo, t, l2, entropy, kl, mmd, critic_value,
        ensemble_spread}]. The base row (step -1) has null critic fields
        and a null mmd.
    practice_episodes.jsonl: one {grad_step, episode, success} object per line; grad_step is the
        step of the policy that collected the episode.
    train_log.json: list of {"_step", "is_success (exploration)", "training/q"} (upstream names).
    confirm_set.json: run_tag, rule, steps[], reasons{step: [reason, ...]}.
    softopc/pool_meta.json: run_tag, task, sources, episodes_per_source, seed_base,
        per_source{source: {episodes, successes}}, successes, failures, n_rows, base_latent.
    softopc/scores.json: run_tag, n_rows, scores{step: {softopc, label_free_mean_q}}.

JSON object keys are strings, so steps used as keys are written as strings.
"""

import json
from pathlib import Path

BASE_STEP = -1  # the base policy's step in scan and held-out files
POOLS = ("scan", "confirm")
PRACTICE_LOG_NAME = "practice_episodes.jsonl"  # also written by dsrl_pi0/ (see there)
VARIANT_NAME = "variant.json"  # also written by dsrl_pi0/


def format_step(step: int) -> str:
    """Return a gradient step in thousands: 100000 -> '100k', 2500 -> '2.5k'."""
    return f"{step / 1000:g}k"


def run_dir(outputs: Path, tag: str) -> Path:
    """Return the folder of one run."""
    return Path(outputs) / tag


def train_dir(outputs: Path, tag: str) -> Path:
    """Return the training folder of one run (checkpoints, variant, practice log, W&B log)."""
    return run_dir(outputs, tag) / "train"


def checkpoint_dir(outputs: Path, tag: str, step: int) -> Path:
    """Return the flax checkpoint folder of one step."""
    return train_dir(outputs, tag) / f"checkpoint{step}"


def practice_log_path(outputs: Path, tag: str) -> Path:
    """Return the practice-episode log that the training code writes."""
    return train_dir(outputs, tag) / PRACTICE_LOG_NAME


def wandb_file(outputs: Path, tag: str) -> Path:
    """Return the run's one offline W&B log file; fail if there is none or more than one."""
    found = sorted(train_dir(outputs, tag).glob("**/offline-run-*/*.wandb"))
    if len(found) != 1:
        raise FileNotFoundError(
            f"expected one offline W&B file under {train_dir(outputs, tag)}, found {len(found)}"
        )
    return found[0]


def train_log_path(outputs: Path, tag: str) -> Path:
    """Return the parsed training log."""
    return run_dir(outputs, tag) / "train_log.json"


def evaluation_dir(outputs: Path, tag: str, pool: str) -> Path:
    """Return the folder of one pool's episode files ('scan' or 'confirm')."""
    if pool not in POOLS:
        raise ValueError(f"pool must be one of {POOLS}, not {pool!r}")
    return run_dir(outputs, tag) / pool


def evaluation_path(outputs: Path, tag: str, pool: str, step: int) -> Path:
    """Return the episode file of one checkpoint in one pool ('scan' or 'confirm')."""
    return evaluation_dir(outputs, tag, pool) / f"{pool}_{step}.json"


def heldout_dir(outputs: Path, tag: str) -> Path:
    """Return the folder of the held-out statistics and the base-policy prefix cache."""
    return run_dir(outputs, tag) / "heldout"


def heldout_path(outputs: Path, tag: str, step: int) -> Path:
    """Return the held-out statistics file of one checkpoint."""
    return heldout_dir(outputs, tag) / f"heldout_{step}.json"


def base_prefixes_path(outputs: Path, tag: str) -> Path:
    """Return the cache of sampled base-policy action prefixes."""
    return heldout_dir(outputs, tag) / "base_prefixes.npz"


def steps_with_files(folder: Path, prefix: str) -> list[int]:
    """Return the sorted steps of the files <prefix>_<step>.json in a folder.

    Raise an error that names the file if a matching file name does not end in an integer step.
    """
    steps = []
    for path in Path(folder).glob(f"{prefix}_*.json"):
        suffix = path.stem[len(prefix) + 1 :]
        try:
            steps.append(int(suffix))
        except ValueError:
            raise ValueError(f"{path} does not match {prefix}_<step>.json") from None
    return sorted(steps)


def evaluation_steps(outputs: Path, tag: str, pool: str) -> list[int]:
    """Return the steps that have an episode file in one pool."""
    return steps_with_files(evaluation_dir(outputs, tag, pool), pool)


def heldout_steps(outputs: Path, tag: str) -> list[int]:
    """Return the steps that have a held-out statistics file."""
    return steps_with_files(heldout_dir(outputs, tag), "heldout")


def confirm_set_path(outputs: Path, tag: str) -> Path:
    """Return the confirm-set file of one run."""
    return run_dir(outputs, tag) / "confirm_set.json"


def softopc_dir(outputs: Path, tag: str) -> Path:
    """Return the SoftOPC folder of one run."""
    return run_dir(outputs, tag) / "softopc"


def pool_path(outputs: Path, tag: str) -> Path:
    """Return the labeled SoftOPC pool rows."""
    return softopc_dir(outputs, tag) / "pool.npz"


def pool_meta_path(outputs: Path, tag: str) -> Path:
    """Return the SoftOPC pool description."""
    return softopc_dir(outputs, tag) / "pool_meta.json"


def softopc_scores_path(outputs: Path, tag: str) -> Path:
    """Return the SoftOPC scores of every checkpoint."""
    return softopc_dir(outputs, tag) / "scores.json"


def screen_path(outputs: Path, suite: str, stage: int) -> Path:
    """Return the task-screen file of one suite and stage (stages count from 1)."""
    return Path(outputs) / "screen" / f"{suite}_stage{stage}.json"


def results_dir(outputs: Path) -> Path:
    """Return the folder of the experiment results."""
    return Path(outputs) / "results"


def read_json(path: Path):
    """Read one JSON file."""
    with open(path) as handle:
        return json.load(handle)


def write_json(path: Path, data) -> None:
    """Write one JSON file, creating its folder."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(data, handle, indent=1)
        handle.write("\n")


def read_jsonl(path: Path) -> list[dict]:
    """Read a file with one JSON object per line."""
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]
