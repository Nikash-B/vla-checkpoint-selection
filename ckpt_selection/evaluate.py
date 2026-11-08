"""Scan and confirm episodes for one run (Sec. 3 "Evaluation"; Algorithm 1, lines 2 and 5).

The scan runs the base policy (step -1) and every checkpoint on the scan seed pool. The confirm
runs the checkpoints in the run's confirm_set.json on the confirm seed pool. Both pools come from
[evaluation] in config/experiments.toml. Episode i of a pool uses environment seed
seed_base + i, so every checkpoint meets the same initial states.

Each step's episodes go to <tag>/<pool>/<pool>_<step>.json. Steps whose file exists are skipped,
so an interrupted evaluation resumes where it stopped.

    python -m ckpt_selection.evaluate --run TAG --pool {scan,confirm} [--outputs DIR]
"""

import argparse
from pathlib import Path
from types import SimpleNamespace

from ckpt_selection import agent as agent_lib
from ckpt_selection import config as config_lib
from ckpt_selection import layout


def run_episodes(
    pi0,
    agent,
    env,
    variant: SimpleNamespace,
    seed_base: int,
    episodes: int,
    episode_limit: int,
) -> list[dict]:
    """Roll out one policy on a seed pool; agent=None runs the base policy.

    The agent picks pi0's noise once per decision (every `variant.query_freq` simulator steps).
    The base policy draws i.i.d. standard-normal noise from a key seeded with seed_base.
    """
    import jax

    noise_key = jax.random.PRNGKey(seed_base)
    rows = []
    for episode in range(episodes):
        seed = seed_base + episode
        env.seed(seed)
        observation = env.reset()
        success = False
        env_steps = 0
        for env_steps in range(1, episode_limit + 1):
            decision_step = (env_steps - 1) % variant.query_freq
            if decision_step == 0:
                if agent is None:
                    noise_key, key = jax.random.split(noise_key)
                    noise = agent_lib.base_noise(key)
                else:
                    noise = agent_lib.actor_noise(
                        agent, agent_lib.actor_observation(observation, variant)
                    )
                pi0_input = agent_lib.pi0_observation(observation, variant)
                actions = pi0.infer(pi0_input, noise=noise)["actions"]
            observation, reward, done, _ = env.step(actions[decision_step])
            if done:
                success = agent_lib.is_success(reward, done)
                break
        rows.append({"episode": episode, "seed": seed, "env_steps": env_steps, "success": success})
        print(f"  episode {episode} seed {seed}: {env_steps} steps, success {success}", flush=True)
    return rows


def steps_for_pool(config: dict, outputs: Path, tag: str, pool: str) -> list[int]:
    """Return the steps to evaluate: base and every checkpoint (scan), or the confirm set."""
    if pool == "scan":
        return [layout.BASE_STEP] + config_lib.checkpoint_steps(config, tag)
    confirm_set_file = layout.confirm_set_path(outputs, tag)
    if not confirm_set_file.exists():
        raise FileNotFoundError(f"{confirm_set_file} is missing; run the confirm_set stage first")
    return sorted(int(step) for step in layout.read_json(confirm_set_file)["steps"])


def evaluate_run(config: dict, outputs: Path, tag: str, pool: str) -> None:
    """Write <pool>_<step>.json for every step of the pool that has no file yet."""
    entry = config_lib.run_entry(config, tag)
    settings = config["evaluation"]
    seed_base = settings[pool]["seed_base"]
    episodes = settings[pool]["episodes"]
    steps = steps_for_pool(config, outputs, tag, pool)
    todo = [step for step in steps if not layout.evaluation_path(outputs, tag, pool, step).exists()]
    print(f"{tag} {pool}: {len(steps) - len(todo)} of {len(steps)} steps done", flush=True)
    if not todo:
        return
    train_dir = layout.train_dir(outputs, tag)
    missing = [
        step
        for step in todo
        if step != layout.BASE_STEP and not layout.checkpoint_dir(outputs, tag, step).exists()
    ]
    if missing:
        raise FileNotFoundError(f"{tag}: no checkpoint under {train_dir} for steps {missing}")

    variant = agent_lib.load_variant(train_dir)
    env, _ = agent_lib.make_libero_env(entry["suite"], entry["task"], settings["render_resolution"])
    pi0 = agent_lib.load_pi0(config)
    agent = None
    if any(step != layout.BASE_STEP for step in todo):
        agent = agent_lib.build_agent(variant, settings["agent_seed"])
    for step in todo:
        print(f"{tag} {pool} step {step}", flush=True)
        policy = None
        if step != layout.BASE_STEP:
            agent_lib.restore(agent, train_dir, step, settings["agent_seed"])
            policy = agent
        rows = run_episodes(
            pi0, policy, env, variant, seed_base, episodes, settings["episode_limit"]
        )
        result = {
            "run_tag": tag,
            "suite": entry["suite"],
            "task": entry["task"],
            "step": step,
            "pool": pool,
            "seed_base": seed_base,
            "episodes": episodes,
            "episode_limit": settings["episode_limit"],
            "agent_seed": settings["agent_seed"],
            "success_rate": sum(row["success"] for row in rows) / len(rows),
            "rows": rows,
        }
        path = layout.evaluation_path(outputs, tag, pool, step)
        layout.write_json(path, result)
        print(f"{tag} {pool} step {step}: success rate {result['success_rate']:.3f}", flush=True)
    env.close()


def main() -> None:
    """Scan or confirm one run."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="run tag from config/experiments.toml")
    parser.add_argument("--pool", required=True, choices=layout.POOLS)
    parser.add_argument("--outputs", default=None, help="outputs folder (default: from config)")
    args = parser.parse_args()

    config = config_lib.load_config()
    outputs = config_lib.outputs_root(config, args.outputs)
    evaluate_run(config, outputs, args.run, args.pool)


if __name__ == "__main__":
    main()
