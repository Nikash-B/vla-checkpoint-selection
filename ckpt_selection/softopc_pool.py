"""Collect a success-labeled episode pool and score every checkpoint's critic on it (GPU; App. D).

collect  Roll out each pool source (the base policy, step -1, and the checkpoints listed in
         [softopc] pool_sources) on the same episodes_per_source episodes, seeds pool_seed_base
         onward. At every decision, store the actor's observation (64x64 image and 8-d state),
         the 32-d noise vector the policy used, the episode and source, and, once the episode
         ends, its success. A checkpoint's noise vector is its actor sample. The base policy
         draws every pi0 noise entry independently, so its rows store the first noise row.
score    If the pool holds at least min_per_class successful and failed episodes, read every
         checkpoint's critic on all rows. SoftOPC is the mean Q (mean over critic heads) on rows
         of successful episodes minus the mean Q on all rows; the label-free mean Q is the mean
         Q on all rows. Otherwise print the counts and write no scores.

Usage: python -m ckpt_selection.softopc_pool {collect,score} --run TAG [--outputs DIR]
"""

import argparse

import numpy as np

from ckpt_selection import agent as agent_lib
from ckpt_selection import config as config_lib
from ckpt_selection import layout

BASE_LATENT = "first_row_of_iid_noise"  # what a base-policy row stores as its noise vector


def roll_out_source(pi0, learner, env, variant, seed: int, episode_limit: int, key):
    """Run one pool episode; return its rows (image, state, latent), its success and the key.

    learner None is the base policy, which draws i.i.d. standard-normal pi0 noise per decision.
    """
    import jax

    env.seed(seed)
    observation = env.reset()
    rows = []
    success = False
    for env_step in range(episode_limit):
        if env_step % variant.query_freq == 0:
            actor_input = agent_lib.actor_observation(observation, variant)
            if learner is None:
                key, draw_key = jax.random.split(key)
                noise_input = agent_lib.base_noise(draw_key)
            else:
                noise_input = agent_lib.actor_noise(learner, actor_input)
            rows.append(
                (
                    actor_input["pixels"][0, ..., 0].astype(np.uint8),
                    actor_input["state"][0, :, 0].astype(np.float32),
                    np.asarray(noise_input)[0, 0].astype(np.float32),
                )
            )
            pi0_input = agent_lib.pi0_observation(observation, variant)
            actions = pi0.infer(pi0_input, noise=noise_input)["actions"]
        observation, reward, done, _ = env.step(actions[env_step % variant.query_freq])
        if done:
            success = agent_lib.is_success(reward, done)
            break
    return rows, bool(success), key


def collect_pool(config: dict, outputs, tag: str) -> None:
    """Roll out every pool source and write pool.npz and pool_meta.json, unless the pool exists.

    pool_meta.json is written after pool.npz, so its presence marks a finished pool.
    """
    import jax

    if layout.pool_meta_path(outputs, tag).exists():
        print(f"{tag}: the SoftOPC pool exists, so it is not collected again")
        return
    entry = config_lib.run_entry(config, tag)
    settings = config["softopc"]
    episodes = settings["episodes_per_source"]
    seed_base = settings["pool_seed_base"]
    limit = config["evaluation"]["episode_limit"]
    train_dir = layout.train_dir(outputs, tag)
    variant = agent_lib.load_variant(train_dir)
    env, _ = agent_lib.make_libero_env(
        entry["suite"], entry["task"], config["evaluation"]["render_resolution"]
    )
    pi0 = agent_lib.load_pi0(config)
    learner = agent_lib.build_agent(variant, config["evaluation"]["agent_seed"])
    key = jax.random.PRNGKey(seed_base)  # draws the base policy's noise

    columns = {name: [] for name in ("pixels", "state", "latent", "episode", "source", "success")}
    per_source = {}
    episode_id = 0
    for source in settings["pool_sources"]:
        if source != layout.BASE_STEP:
            agent_lib.restore(learner, train_dir, source, config["evaluation"]["agent_seed"])
        policy = None if source == layout.BASE_STEP else learner
        successes = 0
        for episode in range(episodes):
            rows, success, key = roll_out_source(
                pi0, policy, env, variant, seed_base + episode, limit, key
            )
            for pixels, state, latent in rows:
                columns["pixels"].append(pixels)
                columns["state"].append(state)
                columns["latent"].append(latent)
                columns["episode"].append(episode_id)
                columns["source"].append(source)
                columns["success"].append(success)
            successes += int(success)
            episode_id += 1
        per_source[str(source)] = {"episodes": episodes, "successes": successes}
        print(f"{tag} pool source {source}: {successes} of {episodes} episodes succeed")

    total_successes = sum(counts["successes"] for counts in per_source.values())
    layout.softopc_dir(outputs, tag).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        layout.pool_path(outputs, tag),
        **{name: np.asarray(values) for name, values in columns.items()},
    )
    layout.write_json(
        layout.pool_meta_path(outputs, tag),
        {
            "run_tag": tag,
            "task": entry["task"],
            "sources": settings["pool_sources"],
            "episodes_per_source": episodes,
            "seed_base": seed_base,
            "per_source": per_source,
            "successes": total_successes,
            "failures": episode_id - total_successes,
            "n_rows": len(columns["episode"]),
            "base_latent": BASE_LATENT,
        },
    )


def critic_means(learner, pixels, state, latent, batch: int, add_states: bool) -> np.ndarray:
    """Return the mean over critic heads at every pool row, computed in batches.

    The dsrl_pi0 critic ensemble stacks its heads on axis 0 (jaxrl2 StateActionEnsemble).
    """
    from jaxrl2.agents.pixel_sac.pixel_sac_learner import get_value

    means = []
    for start in range(0, len(latent), batch):
        rows = slice(start, start + batch)
        observation = {"pixels": pixels[rows][..., np.newaxis]}
        if add_states:
            observation["state"] = state[rows][..., np.newaxis]
        actions = latent[rows][:, np.newaxis, :]
        values = np.asarray(get_value(actions, observation, learner._critic), dtype=np.float64)
        means.append(values.mean(axis=0))
    return np.concatenate(means)


def score_pool(config: dict, outputs, tag: str) -> None:
    """Write SoftOPC and the label-free mean Q of every checkpoint, if the pool is balanced.

    A run whose scores file exists is not scored again.
    """
    if layout.softopc_scores_path(outputs, tag).exists():
        print(f"{tag}: the SoftOPC scores exist, so the checkpoints are not scored again")
        return
    meta = layout.read_json(layout.pool_meta_path(outputs, tag))
    minimum = config["softopc"]["min_per_class"]
    if meta["successes"] < minimum or meta["failures"] < minimum:
        print(
            f"{tag}: the pool holds {meta['successes']} successful and {meta['failures']} failed "
            f"episodes, fewer than {minimum} of each, so no checkpoint is scored"
        )
        return
    with np.load(layout.pool_path(outputs, tag)) as pool_file:
        pixels = pool_file["pixels"]
        state = pool_file["state"]
        latent = pool_file["latent"]
        success = pool_file["success"].astype(bool)
    train_dir = layout.train_dir(outputs, tag)
    variant = agent_lib.load_variant(train_dir)
    learner = agent_lib.build_agent(variant, config["evaluation"]["agent_seed"])
    scores = {}
    for step in config_lib.checkpoint_steps(config, tag):
        agent_lib.restore(learner, train_dir, step, config["evaluation"]["agent_seed"])
        q_values = critic_means(
            learner, pixels, state, latent, config["softopc"]["score_batch"], variant.add_states
        )
        scores[str(step)] = {
            "softopc": float(q_values[success].mean() - q_values.mean()),
            "label_free_mean_q": float(q_values.mean()),
        }
        print(f"{tag} step {step}: SoftOPC {scores[str(step)]['softopc']:+.4f}")
    layout.write_json(
        layout.softopc_scores_path(outputs, tag),
        {"run_tag": tag, "n_rows": len(latent), "scores": scores},
    )


def main() -> None:
    """Collect or score one run's SoftOPC pool."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("collect", "score"))
    parser.add_argument("--run", required=True, help="run tag from config/experiments.toml")
    parser.add_argument("--outputs", default=None, help="outputs folder (default from the config)")
    args = parser.parse_args()

    config = config_lib.load_config()
    outputs = config_lib.outputs_root(config, args.outputs)
    if args.command == "collect":
        collect_pool(config, outputs, args.run)
    else:
        score_pool(config, outputs, args.run)


if __name__ == "__main__":
    main()
