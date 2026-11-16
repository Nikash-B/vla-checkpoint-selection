"""Held-out statistics of every checkpoint of one run at demonstration states (GPU; App. C).

The states come from the task's LIBERO demonstration file, which RL training never reads: the
first `demos` demonstrations, `states_per_demo` evenly spaced states each, every state with the
demonstrated action prefix of one decision. At each state this script computes, per checkpoint:

    l2               mean distance between the mode-action prefix and the demonstrated prefix
    entropy          log standard deviation across sampled prefixes, averaged over action entries
    kl               closed-form KL from a diagonal Gaussian fitted to the sampled noise vectors
                     to the standard normal that pi0 expects
    mmd              biased RBF-kernel MMD^2 between sampled prefixes and the cached base-policy
                     prefixes at the same state, with a median-heuristic bandwidth
    critic_value     mean of the critic heads at the mode noise (its run mean is held-out Q)
    ensemble_spread  standard deviation of the critic heads at the mode noise

The base policy (step -1) runs first. Its mode is zero noise, its samples draw every noise entry
independently, its KL is that of the standard normal itself, and it has no critic. Its sampled
prefixes are cached for the MMD of every checkpoint.

GPU-only dependencies (LIBERO, h5py, openpi_client, PIL, jax, jaxrl2) are imported inside the
functions that use them, so the CPU analysis can import this package without them.

Usage: python -m ckpt_selection.heldout_stats --run TAG [--outputs DIR]
"""

import argparse
from pathlib import Path

import numpy as np

from ckpt_selection import agent as agent_lib
from ckpt_selection import config as config_lib
from ckpt_selection import layout, noise

PI0_IMAGE_SIZE = 224  # pi0 image input, as upstream obs_to_pi_zero_input resizes it
LOG_STD_EPSILON = 1e-8  # keeps log std finite when sampled prefixes agree exactly
LATENT_STD_EPSILON = 1e-6  # keeps the fitted Gaussian's log std finite
BANDWIDTH_EPSILON = 1e-12  # keeps the RBF bandwidth finite when all prefixes coincide

# Observation names in the LIBERO demonstration files.
DEMO_AGENTVIEW = "agentview_rgb"
DEMO_WRIST = "eye_in_hand_rgb"
DEMO_EE_POS = "ee_pos"
DEMO_EE_ORI = "ee_ori"
DEMO_GRIPPER = "gripper_states"


def demo_file(suite: str, task_id: int) -> tuple[Path, str]:
    """Return the task's LIBERO demonstration file and its language instruction."""
    from libero.libero import benchmark, get_libero_path

    task_suite = benchmark.get_benchmark_dict()[suite]()
    path = Path(get_libero_path("datasets")) / task_suite.get_task_demonstration(task_id)
    return path, str(task_suite.get_task(task_id).language)


def load_demo_states(path: Path, demos: int, per_demo: int, query_freq: int) -> list[dict]:
    """Return demos x per_demo states, evenly spaced in each of the first demonstrations.

    Each state keeps both camera images, the 8-d proprioceptive state (ee_pos 3 + ee_ori 3 +
    gripper_states 2, the layout of upstream obs_to_pi_zero_input) and the demonstrated actions
    of the next query_freq steps. The last chosen index leaves room for a full prefix.
    """
    import h5py

    states = []
    with h5py.File(path, "r") as handle:
        data = handle["data"]
        names = sorted(data.keys(), key=lambda name: int(name.split("_")[-1]))[:demos]
        for name in names:
            observations = data[name]["obs"]
            actions = np.asarray(data[name]["actions"])
            last = max(actions.shape[0] - query_freq - 1, 0)
            for t in np.linspace(0, last, per_demo).astype(int):
                pose = np.concatenate([observations[DEMO_EE_POS][t], observations[DEMO_EE_ORI][t]])
                gripper = np.asarray(observations[DEMO_GRIPPER][t])
                states.append(
                    {
                        "demo": name,
                        "t": int(t),
                        "agentview": np.asarray(observations[DEMO_AGENTVIEW][t]),
                        "wrist": np.asarray(observations[DEMO_WRIST][t]),
                        "state": np.concatenate([pose, gripper]).astype(np.float32),
                        "demo_actions": actions[t : t + query_freq].astype(np.float32),
                    }
                )
    return states


def flip(image: np.ndarray) -> np.ndarray:
    """Rotate a LIBERO camera image by 180 degrees, as upstream does for environment images."""
    return np.ascontiguousarray(image[::-1, ::-1])


def pi0_image(image: np.ndarray) -> np.ndarray:
    """Flip, pad-resize to pi0's image size and convert to uint8, as upstream does."""
    from openpi_client import image_tools

    resized = image_tools.resize_with_pad(flip(image), PI0_IMAGE_SIZE, PI0_IMAGE_SIZE)
    return image_tools.convert_to_uint8(resized)


def pi0_observation_from_demo(state: dict, language: str) -> dict:
    """Return pi0's input at a demonstration state, laid out as upstream obs_to_pi_zero_input."""
    return {
        "observation/image": pi0_image(state["agentview"]),
        "observation/wrist_image": pi0_image(state["wrist"]),
        "observation/state": state["state"],
        "prompt": language,
    }


def actor_observation_from_demo(state: dict, variant) -> dict:
    """Return the actor and critic input at a demonstration state, as upstream obs_to_img."""
    import PIL.Image

    image = flip(state["agentview"])
    size = variant.resize_image
    if size > 0:
        image = np.array(PIL.Image.fromarray(image).resize((size, size)))
    observation = {"pixels": image[np.newaxis, ..., np.newaxis]}
    if variant.add_states:
        observation["state"] = state["state"][np.newaxis, ..., np.newaxis]
    return observation


def gaussian_kl_to_standard_normal(mean: np.ndarray, std: np.ndarray) -> float:
    """Return KL(N(mean, diag(std^2)) || N(0, I)) in closed form."""
    return float(0.5 * np.sum(std**2 + mean**2 - 1.0 - 2.0 * np.log(std)))


def rbf_mmd2(samples: np.ndarray, reference: np.ndarray) -> float:
    """Return the biased RBF-kernel MMD^2 of two sample sets (rows are samples).

    The bandwidth is the median squared distance over all pairs of the pooled samples.
    """
    samples = np.asarray(samples, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    pooled = np.concatenate([samples, reference])
    squared = np.sum((pooled[:, None] - pooled[None, :]) ** 2, axis=-1)
    gamma = 1.0 / (np.median(squared[np.triu_indices_from(squared, k=1)]) + BANDWIDTH_EPSILON)
    count = len(samples)
    kernel = np.exp(-gamma * squared)
    within_samples = kernel[:count, :count].mean()
    within_reference = kernel[count:, count:].mean()
    across = kernel[:count, count:].mean()
    return float(within_samples + within_reference - 2 * across)


def decode_prefix(pi0, observation: dict, noise_input, query_freq: int) -> np.ndarray:
    """Return the first query_freq actions that pi0 decodes from one noise input."""
    return np.asarray(pi0.infer(observation, noise=noise_input)["actions"])[:query_freq]


def prefix_statistics(mode_prefix: np.ndarray, sampled: np.ndarray, demo_actions) -> dict:
    """Return L2 to the demonstrated prefix and the entropy of the sampled prefixes."""
    length = min(len(demo_actions), len(mode_prefix))
    distances = np.linalg.norm(mode_prefix[:length] - demo_actions[:length], axis=-1)
    flat = sampled.reshape(len(sampled), -1)
    return {
        "l2": float(np.mean(distances)),
        "entropy": float(np.mean(np.log(np.std(flat, axis=0) + LOG_STD_EPSILON))),
    }


def base_statistics(pi0, states: list, language: str, samples: int, key, query_freq: int):
    """Return the base policy's per-state rows and sampled prefixes (states x samples x entries).

    The mode is zero noise. Each sample draws all pi0 noise entries independently.
    """
    import jax

    noise_shape = (noise.PI0_NOISE_ROWS, noise.PI0_NOISE_DIM)
    zero_noise = np.zeros((1, *noise_shape), dtype=np.float32)
    # Zero by construction: the base policy draws its noise from the standard normal itself.
    base_kl = gaussian_kl_to_standard_normal(
        np.zeros(noise.PI0_NOISE_DIM), np.ones(noise.PI0_NOISE_DIM)
    )
    rows = []
    prefixes = []
    for state in states:
        observation = pi0_observation_from_demo(state, language)
        # One key split per state, so each state's draws are independent of the others.
        key, sample_key = jax.random.split(key)
        draws = jax.random.normal(sample_key, (samples, *noise_shape))
        mode_prefix = decode_prefix(pi0, observation, zero_noise, query_freq)
        sampled = np.stack(
            [decode_prefix(pi0, observation, np.asarray(draw)[None], query_freq) for draw in draws]
        )
        row = {"demo": state["demo"], "t": state["t"]}
        row.update(prefix_statistics(mode_prefix, sampled, state["demo_actions"]))
        row["kl"] = base_kl
        row.update({"mmd": None, "critic_value": None, "ensemble_spread": None})
        rows.append(row)
        prefixes.append(sampled.reshape(samples, -1))
    return rows, np.stack(prefixes)


def checkpoint_statistics(
    learner, pi0, states, language, base_prefixes, samples, variant, query_freq
) -> list:
    """Return one checkpoint's per-state rows: L2, entropy, KL, MMD and the critic heads."""
    from jaxrl2.agents.pixel_sac.pixel_sac_learner import get_value

    rows = []
    for state, base_sampled in zip(states, base_prefixes):
        observation = pi0_observation_from_demo(state, language)
        actor_input = actor_observation_from_demo(state, variant)
        mode_latent = np.reshape(learner.eval_actions(actor_input), learner.action_chunk_shape)
        mode_noise = noise.fill_noise_rows(mode_latent)
        sample_noise = [agent_lib.actor_noise(learner, actor_input) for _ in range(samples)]
        latents = np.stack([np.asarray(sample)[0, 0] for sample in sample_noise])
        mode_prefix = decode_prefix(pi0, observation, mode_noise, query_freq)
        sampled = np.stack(
            [decode_prefix(pi0, observation, sample, query_freq) for sample in sample_noise]
        )
        heads = np.asarray(
            get_value(mode_noise[:, :1, :], actor_input, learner._critic), dtype=np.float64
        )
        row = {"demo": state["demo"], "t": state["t"]}
        row.update(prefix_statistics(mode_prefix, sampled, state["demo_actions"]))
        row["kl"] = gaussian_kl_to_standard_normal(
            latents.mean(axis=0), latents.std(axis=0) + LATENT_STD_EPSILON
        )
        row["mmd"] = rbf_mmd2(sampled.reshape(samples, -1), base_sampled)
        row["critic_value"] = float(np.mean(heads))
        row["ensemble_spread"] = float(np.std(heads))
        rows.append(row)
    return rows


def run_mean(rows: list, key: str):
    """Return the mean of one per-state field, or None when the field is null."""
    values = [row[key] for row in rows if row[key] is not None]
    return float(np.mean(values)) if values else None


def summary(tag: str, step: int, rows: list, samples: int) -> dict:
    """Return the heldout_<step>.json content: run means at full precision and per-state rows."""
    return {
        "run_tag": tag,
        "step": step,
        "n_states": len(rows),
        "samples_per_state": samples,
        "l2": run_mean(rows, "l2"),
        "action_entropy": run_mean(rows, "entropy"),
        "kl": run_mean(rows, "kl"),
        "mmd": run_mean(rows, "mmd"),
        "heldout_q": run_mean(rows, "critic_value"),
        "ensemble_spread": run_mean(rows, "ensemble_spread"),
        "per_state": rows,
    }


def load_base_prefixes(path: Path, states: list) -> np.ndarray:
    """Read the cached base prefixes and check that they belong to the same states."""
    cached = np.load(path)
    expected = [(state["demo"], state["t"]) for state in states]
    found = list(zip(cached["demo"].tolist(), cached["t"].tolist()))
    if found != expected:
        raise ValueError(f"{path} holds other demonstration states; delete it and rerun")
    return cached["prefixes"]


def main() -> None:
    """Write the base row and one held-out statistics file per checkpoint of one run."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="run tag from config/experiments.toml")
    parser.add_argument("--outputs", default=None, help="outputs folder (default from the config)")
    args = parser.parse_args()

    import jax

    config = config_lib.load_config()
    outputs = config_lib.outputs_root(config, args.outputs)
    entry = config_lib.run_entry(config, args.run)
    heldout = config["heldout"]
    samples = heldout["samples_per_state"]
    train_dir = layout.train_dir(outputs, args.run)
    variant = agent_lib.load_variant(train_dir)
    query_freq = variant.query_freq
    path, language = demo_file(entry["suite"], entry["task"])
    states = load_demo_states(path, heldout["demos"], heldout["states_per_demo"], query_freq)
    print(f"{args.run}: {len(states)} held-out states from {path.name}")
    pi0 = agent_lib.load_pi0(config)

    base_file = layout.heldout_path(outputs, args.run, layout.BASE_STEP)
    prefixes_file = layout.base_prefixes_path(outputs, args.run)
    if base_file.exists() and prefixes_file.exists():
        base_prefixes = load_base_prefixes(prefixes_file, states)
    else:
        key = jax.random.PRNGKey(heldout["base_sample_seed"])
        rows, base_prefixes = base_statistics(pi0, states, language, samples, key, query_freq)
        prefixes_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            prefixes_file,
            demo=np.array([state["demo"] for state in states]),
            t=np.array([state["t"] for state in states]),
            prefixes=base_prefixes,
        )
        layout.write_json(base_file, summary(args.run, layout.BASE_STEP, rows, samples))
        print(f"{args.run} base policy -> {base_file}")

    # restore resets the sampling RNG, so every checkpoint samples from the same seeded stream.
    learner = agent_lib.build_agent(variant, config["evaluation"]["agent_seed"])
    for step in config_lib.checkpoint_steps(config, args.run):
        out_file = layout.heldout_path(outputs, args.run, step)
        if out_file.exists():
            continue
        agent_lib.restore(learner, train_dir, step, config["evaluation"]["agent_seed"])
        rows = checkpoint_statistics(
            learner, pi0, states, language, base_prefixes, samples, variant, query_freq
        )
        result = summary(args.run, step, rows, samples)
        layout.write_json(out_file, result)
        print(f"{args.run} step {step}: held-out Q {result['heldout_q']:.3f} -> {out_file}")


if __name__ == "__main__":
    main()
