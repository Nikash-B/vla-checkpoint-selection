"""Rebuild a trained DSRL agent, the frozen pi0 policy and a LIBERO task for rollouts (GPU).

A flax checkpoint restores only into a structurally identical agent, so `build_agent` repeats the
construction in dsrl_pi0/examples/train_sim.py (`DummyEnv` and `PixelSACLearner`) from the
run's variant.json. jax, openpi, LIBERO and dsrl_pi0 are imported inside functions.

Run as a module to check that LIBERO renders on the GPU before any rollout:

    python -m ckpt_selection.agent
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ckpt_selection import config as config_lib
from ckpt_selection import layout
from ckpt_selection.noise import PI0_NOISE_DIM, PI0_NOISE_ROWS, fill_noise_rows

LIBERO_STATE_DIM = 8  # proprioceptive state size, as DummyEnv in examples/train_sim.py (Table 2)
ACTOR_LATENT_STEPS = 1  # the actor emits one 32-dimensional latent per decision (DummyEnv)
SUCCESS_REWARD = 1  # LIBERO's reward when the goal fires (train_sim.py: env_max_reward = 1)
CAMERA_KEY = "agentview_image"  # the camera the actor and pi0 see
SOFTWARE_RENDERERS = ("llvmpipe", "softpipe", "swrast", "mesa offscreen")


def load_variant(train_dir: Path) -> SimpleNamespace:
    """Read the run configuration that the training code wrote into its run folder."""
    with open(Path(train_dir) / layout.VARIANT_NAME) as handle:
        return SimpleNamespace(**json.load(handle))


def base_variant(config: dict, task_description: str) -> SimpleNamespace:
    """Return the settings a base-policy rollout needs when no training run exists (task screen)."""
    return SimpleNamespace(
        env=config["training"]["env"],
        query_freq=config["training"]["query_freq"],
        task_description=task_description,
    )


def build_agent(variant: SimpleNamespace, seed: int):
    """Build a PixelSACLearner with the training-time spaces and hyperparameters."""
    from gym.spaces import Box, Dict
    from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner
    from jaxrl2.utils.general_utils import add_batch_dim

    image_shape = (variant.resize_image, variant.resize_image, 3 * variant.num_cameras, 1)
    spaces = {"pixels": Box(low=0, high=255, shape=image_shape, dtype=np.uint8)}
    if variant.add_states:
        spaces["state"] = Box(low=-1.0, high=1.0, shape=(LIBERO_STATE_DIM, 1), dtype=np.float32)
    observation_space = Dict(spaces)
    action_space = Box(low=-1, high=1, shape=(ACTOR_LATENT_STEPS, PI0_NOISE_DIM), dtype=np.float32)
    sample_observation = add_batch_dim(observation_space.sample())
    sample_action = add_batch_dim(action_space.sample())
    return PixelSACLearner(seed, sample_observation, sample_action, **variant.train_kwargs)


def restore(agent, train_dir: Path, step: int, seed: int) -> None:
    """Load the checkpoint saved at one gradient step into the agent and reset its sampling RNG.

    Checkpoints do not store the actor's sampling RNG. The reset to `seed` makes a checkpoint's
    samples independent of which checkpoints the same process restored before it.
    """
    import jax

    checkpoint = Path(train_dir) / f"checkpoint{step}"  # flax's name; see layout.checkpoint_dir
    if not checkpoint.exists():
        raise FileNotFoundError(f"no checkpoint at {checkpoint}")
    agent.restore_checkpoint(str(checkpoint))
    agent._rng = jax.random.PRNGKey(seed)


def make_libero_env(suite: str, task_id: int, resolution: int):
    """Build the LIBERO task as examples/train_sim.py does; return (env, task)."""
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task = benchmark.get_benchmark_dict()[suite]().get_task(task_id)
    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file, camera_heights=resolution, camera_widths=resolution
    )
    return env, task


def suite_size(suite: str) -> int:
    """Return the number of tasks in a LIBERO suite."""
    from libero.libero import benchmark

    return benchmark.get_benchmark_dict()[suite]().n_tasks


def load_pi0(config: dict):
    """Load the frozen pi0 policy at the pinned config and checkpoint (Sec. 3)."""
    from openpi.policies import policy_config
    from openpi.shared import download
    from openpi.training import config as openpi_config

    train_config = openpi_config.get_config(config["upstream"]["pi0_config"])
    noise_shape = (train_config.model.action_horizon, train_config.model.action_dim)
    if noise_shape != (PI0_NOISE_ROWS, PI0_NOISE_DIM):
        raise ValueError(f"pi0 noise shape {noise_shape} != {(PI0_NOISE_ROWS, PI0_NOISE_DIM)}")
    checkpoint = download.maybe_download(config["upstream"]["pi0_checkpoint"])
    return policy_config.create_trained_policy(train_config, checkpoint)


def actor_observation(env_observation: dict, variant: SimpleNamespace) -> dict:
    """Convert a LIBERO observation to the actor and critic input (pixels and state)."""
    from examples.train_utils_sim import obs_to_img, obs_to_qpos

    pixels = obs_to_img(env_observation, variant)
    observation = {"pixels": pixels[np.newaxis, ..., np.newaxis]}
    if variant.add_states:
        state = obs_to_qpos(env_observation, variant)
        observation["state"] = state[np.newaxis, ..., np.newaxis]
    return observation


def pi0_observation(env_observation: dict, variant: SimpleNamespace) -> dict:
    """Convert a LIBERO observation to pi0's input, as the training code does."""
    from examples.train_utils_sim import obs_to_pi_zero_input

    return obs_to_pi_zero_input(env_observation, variant)


def actor_noise(agent, observation: dict) -> np.ndarray:
    """Sample one actor latent and repeat it over pi0's noise rows; shape (1, 50, 32)."""
    latent = np.reshape(agent.sample_actions(observation), agent.action_chunk_shape)
    return fill_noise_rows(latent)


def base_noise(key) -> np.ndarray:
    """Draw i.i.d. standard-normal noise for every entry of pi0's input (the base policy)."""
    import jax

    return np.asarray(jax.random.normal(key, (1, PI0_NOISE_ROWS, PI0_NOISE_DIM)))


def is_success(reward: float, done: bool) -> bool:
    """Return True when the episode ended because the task's goal fired."""
    return bool(done) and reward == SUCCESS_REWARD


def check_rendering(config: dict) -> str:
    """Render one LIBERO frame; fail on an empty image or a software renderer.

    Returns the OpenGL renderer name. A software renderer gives correct images but makes every
    rollout far slower, so setup stops there.
    """
    from OpenGL import GL

    first_run = config["runs"][0]
    env, _ = make_libero_env(
        first_run["suite"], first_run["task"], config["evaluation"]["render_resolution"]
    )
    try:
        observation = env.reset()
        if not np.asarray(observation[CAMERA_KEY]).any():
            raise RuntimeError(f"LIBERO rendered an empty {CAMERA_KEY}")
        renderer = GL.glGetString(GL.GL_RENDERER)
        renderer = (
            renderer.decode(errors="replace") if isinstance(renderer, bytes) else str(renderer)
        )
    finally:
        env.close()
    if any(name in renderer.lower() for name in SOFTWARE_RENDERERS):
        raise RuntimeError(f"LIBERO renders in software ({renderer}); check MUJOCO_GL=egl")
    return renderer


def main() -> None:
    """Check that LIBERO renders on the GPU."""
    renderer = check_rendering(config_lib.load_config())
    print(f"LIBERO renders on the GPU: {renderer}")


if __name__ == "__main__":
    main()
