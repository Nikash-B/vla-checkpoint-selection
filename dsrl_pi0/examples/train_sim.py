#! /usr/bin/env python
"""Build the LIBERO task, the pi0 policy and the DSRL agent, then run the training loop."""

import os

# Tell XLA to use Triton GEMM, which improves steps/sec by about 30% on some GPUs. Source:
# https://github.com/huggingface/gym-aloha/tree/main?tab=readme-ov-file#-gpu-rendering-egl
# The flag is set before jax is imported.
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags

import json  # noqa: E402
import pathlib  # noqa: E402
from functools import partial  # noqa: E402

import jax  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from gym.spaces import Box, Dict  # noqa: E402
from jax.experimental.compilation_cache import compilation_cache  # noqa: E402
from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner  # noqa: E402
from jaxrl2.data import ReplayBuffer  # noqa: E402
from jaxrl2.utils.general_utils import add_batch_dim  # noqa: E402
from jaxrl2.utils.wandb_logger import WandBLogger, create_exp_name  # noqa: E402
from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
from openpi.policies import policy_config  # noqa: E402
from openpi.shared import download  # noqa: E402
from openpi.training import config as openpi_config  # noqa: E402

from examples.train_utils_sim import trajwise_alternating_training_loop  # noqa: E402

home_dir = os.environ["HOME"]
compilation_cache.initialize_cache(os.path.join(home_dir, "jax_compilation_cache"))


def _get_libero_env(task, resolution, seed):
    """Initialize and return the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    # The seed affects object positions even when the initial state is fixed.
    env.seed(seed)
    return env, task_description


def shard_batch(batch, sharding):
    """Shard a batch across devices along its first dimension.

    Args:
        batch: A pytree of arrays.
        sharding: A jax Sharding object with shape (num_devices,).
    """
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(x, sharding.reshape(sharding.shape[0], *((1,) * (x.ndim - 1)))),
        batch,
    )


# Changed from upstream 7f48937: DummyEnv is a plain class (it never used the gymnasium wrapper
# it subclassed) and does not keep the variant.
class DummyEnv:
    """Hold the observation and action spaces that the agent and replay buffer are built from."""

    def __init__(self, variant):
        """Build the pixel (and optional state) observation space and the noise action space."""
        self.image_shape = (variant.resize_image, variant.resize_image, 3 * variant.num_cameras, 1)
        obs_dict = {}
        obs_dict["pixels"] = Box(low=0, high=255, shape=self.image_shape, dtype=np.uint8)
        if variant.add_states:
            state_dim = 8  # LIBERO
            obs_dict["state"] = Box(low=-1.0, high=1.0, shape=(state_dim, 1), dtype=np.float32)
        self.observation_space = Dict(obs_dict)
        # 32 is the size of the pi0 noise (action) space.
        self.action_space = Box(low=-1, high=1, shape=(1, 32), dtype=np.float32)


def main(variant):
    """Build the environment, the pi0 policy, the agent and the replay buffer, then train."""
    devices = jax.local_devices()
    num_devices = len(devices)
    assert variant.batch_size % num_devices == 0
    # Changed from upstream 7f48937: the debug prints of the device count, batch size and sample
    # shapes are removed.
    # Shard the leading (batch) dimension evenly across all devices.
    sharding = jax.sharding.PositionalSharding(devices)
    shard_fn = partial(shard_batch, sharding=sharding)

    # Prevent TensorFlow from using the GPUs.
    tf.config.set_visible_devices([], "GPU")

    train_kwargs = variant["train_kwargs"]

    exp_name = create_exp_name(variant.prefix, seed=variant.seed)

    # Changed from upstream 7f48937: the run folder is the required --output_dir (upstream:
    # $EXP/<run name>), and the random-prefix and --suffix fallbacks are removed.
    output_dir = variant.output_dir
    variant.outputdir = output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("writing to output dir ", output_dir)

    # Changed from upstream 7f48937: the ALOHA branches are removed; every run trains a LIBERO task.
    if variant.env != "libero":
        raise NotImplementedError(variant.env)
    benchmark_dict = benchmark.get_benchmark_dict()
    # Changed from upstream 7f48937: suite and task from the command line
    # (upstream: libero_90, 57).
    task_suite = benchmark_dict[variant.task_suite]()
    task_id = variant.task_id
    task = task_suite.get_task(task_id)
    env, task_description = _get_libero_env(task, 256, variant.seed)
    eval_env = env
    variant.task_description = task_description
    variant.env_max_reward = 1
    variant.max_timesteps = 400

    # Changed from upstream 7f48937: save the full run configuration, which evaluation reads to
    # rebuild the agent with the training-time spaces and hyperparameters.
    with open(os.path.join(output_dir, "variant.json"), "w") as variant_file:
        json.dump(dict(variant), variant_file, indent=2, default=str)

    # Changed from upstream 7f48937: the W&B group is the prefix (upstream appended "_" and
    # --launch_group_id), and the W&B log stays in the run folder (upstream: a temp folder).
    wandb_logger = WandBLogger(
        variant.prefix != "",
        variant,
        variant.wandb_project,
        experiment_id=exp_name,
        output_dir=output_dir,
        group_name=variant.prefix,
    )

    dummy_env = DummyEnv(variant)
    sample_obs = add_batch_dim(dummy_env.observation_space.sample())
    sample_action = add_batch_dim(dummy_env.action_space.sample())

    pi0_config = openpi_config.get_config("pi0_libero")
    checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_libero")
    agent_dp = policy_config.create_trained_policy(pi0_config, checkpoint_dir)
    # Changed from upstream 7f48937: print the checkpoint path (upstream passed a "%s" format
    # string and the path as two separate print arguments).
    print(f"Loaded pi0 policy from {checkpoint_dir}")
    agent = PixelSACLearner(variant.seed, sample_obs, sample_action, **train_kwargs)

    online_buffer_size = variant.max_steps // variant.multi_grad_step
    online_replay_buffer = ReplayBuffer(
        dummy_env.observation_space, dummy_env.action_space, int(online_buffer_size)
    )
    replay_buffer = online_replay_buffer
    # Changed from upstream 7f48937: replay_buffer.seed() is removed with the Dataset base class;
    # its generator was never read (sampling uses the global numpy RNG).
    trajwise_alternating_training_loop(
        variant,
        agent,
        env,
        eval_env,
        online_replay_buffer,
        replay_buffer,
        wandb_logger,
        shard_fn=shard_fn,
        agent_dp=agent_dp,
    )
