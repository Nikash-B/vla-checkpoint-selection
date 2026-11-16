"""Pixel-based SAC agent that DSRL trains, plus its Q-value plots."""

import matplotlib

# Select the non-interactive backend before pyplot is imported.
matplotlib.use("Agg")

import copy
import functools
import pathlib
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training import checkpoints, train_state
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from jaxrl2.agents.agent import Agent
from jaxrl2.agents.pixel_sac.actor_updater import update_actor
from jaxrl2.agents.pixel_sac.critic_updater import update_critic
from jaxrl2.agents.pixel_sac.temperature import Temperature
from jaxrl2.agents.pixel_sac.temperature_updater import update_temperature
from jaxrl2.data.augmentations import batched_random_crop, color_transform
from jaxrl2.networks.encoders.networks import Encoder, PixelMultiplexer
from jaxrl2.networks.learned_std_normal_policy import LearnedStdTanhNormalPolicy
from jaxrl2.networks.values import StateActionEnsemble
from jaxrl2.types import DatasetDict, Params, PRNGKey
from jaxrl2.utils.target_update import soft_target_update


class TrainState(train_state.TrainState):
    """Flax train state that also carries batch-norm statistics."""

    batch_stats: Any


@functools.partial(jax.jit, static_argnames=("critic_reduction", "aug_next"))
def _update_jit(
    rng: PRNGKey,
    actor: TrainState,
    critic: TrainState,
    target_critic_params: Params,
    temp: TrainState,
    batch: FrozenDict,
    discount: float,
    tau: float,
    target_entropy: float,
    critic_reduction: str,
    aug_next: bool,
) -> Tuple[PRNGKey, TrainState, TrainState, Params, TrainState, Dict[str, float]]:
    """Augment the batch, then update the critic, target critic, actor and temperature."""
    # Changed from upstream 7f48937: the color_jitter switch (never passed, so always on), the
    # pixel-ndim guard (always true for (B, H, W, 3, 1) pixels) and the num_cameras > 1 loops
    # (every run has one camera) are removed. The jax.random.split calls keep their order.
    rng, key = jax.random.split(rng)
    aug_pixels = batched_random_crop(key, batch["observations"]["pixels"])
    rng, key = jax.random.split(rng)
    aug_pixels = (color_transform(key, aug_pixels.astype(jnp.float32) / 255.0) * 255).astype(
        jnp.uint8
    )

    observations = batch["observations"].copy(add_or_replace={"pixels": aug_pixels})
    batch = batch.copy(add_or_replace={"observations": observations})

    key, rng = jax.random.split(rng)
    if aug_next:
        rng, key = jax.random.split(rng)
        aug_next_pixels = batched_random_crop(key, batch["next_observations"]["pixels"])
        rng, key = jax.random.split(rng)
        aug_next_pixels = (
            color_transform(key, aug_next_pixels.astype(jnp.float32) / 255.0) * 255
        ).astype(jnp.uint8)
        next_observations = batch["next_observations"].copy(
            add_or_replace={"pixels": aug_next_pixels}
        )
        batch = batch.copy(add_or_replace={"next_observations": next_observations})

    key, rng = jax.random.split(rng)
    target_critic = critic.replace(params=target_critic_params)
    new_critic, critic_info = update_critic(
        key, actor, critic, target_critic, temp, batch, discount, critic_reduction=critic_reduction
    )
    new_target_critic_params = soft_target_update(new_critic.params, target_critic_params, tau)

    key, rng = jax.random.split(rng)
    new_actor, actor_info = update_actor(
        key, actor, new_critic, temp, batch, critic_reduction=critic_reduction
    )
    new_temp, alpha_info = update_temperature(temp, actor_info["entropy"], target_entropy)

    return (
        rng,
        new_actor,
        new_critic,
        new_target_critic_params,
        new_temp,
        {**critic_info, **actor_info, **alpha_info},
    )


class PixelSACLearner(Agent):
    """Soft actor-critic on pixel observations, as described in arXiv:1812.05905."""

    def __init__(
        self,
        seed: int,
        observations: Union[jnp.ndarray, DatasetDict],
        actions: jnp.ndarray,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        # Changed from upstream 7f48937: decay_steps and its cosine actor schedule are
        # removed; no run passes it.
        hidden_dims: Sequence[int] = (256, 256),
        cnn_features: Sequence[int] = (32, 32, 32, 32),
        cnn_strides: Sequence[int] = (2, 1, 1, 1),
        cnn_padding: str = "VALID",
        latent_dim: int = 50,
        discount: float = 0.99,
        tau: float = 0.005,
        critic_reduction: str = "mean",
        dropout_rate: Optional[float] = None,
        # Changed from upstream 7f48937: the default is "small", the only encoder kept below.
        encoder_type="small",
        # Changed from upstream 7f48937: encoder_norm, use_spatial_softmax and
        # softmax_temperature only configured the removed ResNet encoders and are not read.
        # They stay as parameters because the variant.json of every trained run passes them.
        encoder_norm="group",
        use_spatial_softmax=True,
        softmax_temperature=1,
        # Changed from upstream 7f48937: color_jitter, which no run passes, is removed (crop and
        # color jitter always run).
        aug_next=True,
        use_bottleneck=True,
        init_temperature: float = 1.0,
        num_qs: int = 2,
        target_entropy: Optional[Union[float, str]] = None,
        action_magnitude: float = 1.0,
        # Changed from upstream 7f48937: num_cameras is not read here (every run has one camera);
        # it stays a parameter because launch_train_sim.py passes it with the training arguments.
        num_cameras: int = 1,
    ):
        """Build the actor, critic, target critic and temperature with their optimizers."""
        self.aug_next = aug_next

        self.action_dim = np.prod(actions.shape[-2:])
        self.action_chunk_shape = actions.shape[-2:]

        self.tau = tau
        self.discount = discount
        self.critic_reduction = critic_reduction

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        # Changed from upstream 7f48937: only the 'small' encoder that every run uses
        # (launch_train_sim.py) is kept; the Impala and ResNet encoder branches are removed.
        if encoder_type == "small":
            encoder_def = Encoder(cnn_features, cnn_strides, cnn_padding)
        else:
            raise ValueError("encoder type not found!")

        if len(hidden_dims) == 1:
            hidden_dims = (hidden_dims[0], hidden_dims[0], hidden_dims[0])

        policy_def = LearnedStdTanhNormalPolicy(
            hidden_dims,
            self.action_dim,
            dropout_rate=dropout_rate,
            low=-action_magnitude,
            high=action_magnitude,
        )

        actor_def = PixelMultiplexer(
            encoder=encoder_def,
            network=policy_def,
            latent_dim=latent_dim,
            use_bottleneck=use_bottleneck,
        )
        # Changed from upstream 7f48937: the debug prints of the actor and critic definitions,
        # of critic_reduction, and the unused self._critic_init_params are removed.
        # Changed from upstream 7f48937: no network has batch norm, so batch_stats is None
        # directly (upstream read it from the init output, where it is never present).
        actor_params = actor_def.init(actor_key, observations)["params"]

        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=actor_lr),
            batch_stats=None,
        )

        critic_def = StateActionEnsemble(hidden_dims, num_qs=num_qs)
        critic_def = PixelMultiplexer(
            encoder=encoder_def,
            network=critic_def,
            latent_dim=latent_dim,
            use_bottleneck=use_bottleneck,
        )
        critic_params = critic_def.init(critic_key, observations, actions)["params"]
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=critic_lr),
            batch_stats=None,
        )
        target_critic_params = copy.deepcopy(critic_params)

        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)["params"]
        temp = TrainState.create(
            apply_fn=temp_def.apply,
            params=temp_params,
            tx=optax.adam(learning_rate=temp_lr),
            batch_stats=None,
        )

        self._rng = rng
        self._actor = actor
        self._critic = critic
        self._target_critic_params = target_critic_params
        self._temp = temp
        if target_entropy is None or target_entropy == "auto":
            self.target_entropy = -self.action_dim / 2
        else:
            self.target_entropy = float(target_entropy)
        print(f"target_entropy: {self.target_entropy}")

    def update(self, batch: FrozenDict) -> Dict[str, float]:
        """Take one update step on a batch and return the logged statistics."""
        new_rng, new_actor, new_critic, new_target_critic, new_temp, info = _update_jit(
            self._rng,
            self._actor,
            self._critic,
            self._target_critic_params,
            self._temp,
            batch,
            self.discount,
            self.tau,
            self.target_entropy,
            self.critic_reduction,
            self.aug_next,
        )

        self._rng = new_rng
        self._actor = new_actor
        self._critic = new_critic
        self._target_critic_params = new_target_critic
        self._temp = new_temp
        return info

    # Changed from upstream 7f48937: the unused variant, eval_buffer_iterator and eval_env
    # parameters of perform_eval and make_value_reward_visulization are removed.
    def perform_eval(self, i, wandb_logger, eval_buffer):
        """Log Q-value and reward plots for trajectories from the evaluation buffer."""
        from examples.train_utils_sim import make_multiple_value_reward_visulizations

        make_multiple_value_reward_visulizations(self, i, eval_buffer, wandb_logger)

    def make_value_reward_visulization(self, trajs):
        """Return one stacked image of Q-value, reward and mask plots per trajectory."""
        num_trajs = len(trajs["rewards"])
        traj_images = []

        for traj_index in range(num_trajs):
            observations = trajs["observations"][traj_index]
            actions = trajs["actions"][traj_index]
            rewards = trajs["rewards"][traj_index]
            masks = trajs["masks"][traj_index]

            q_pred = []

            for t in range(len(actions)):
                action = actions[t][None]
                obs_pixels = observations["pixels"][t]

                # Changed from upstream 7f48937: the next-observation dict, which was built
                # here but never used, is removed.
                obs_dict = {"pixels": obs_pixels[None]}
                for obs_key, obs_value in observations.items():
                    if "pixels" not in obs_key:
                        obs_dict[obs_key] = obs_value[t][None]

                q_value = get_value(action, obs_dict, self._critic)
                q_pred.append(q_value)

            traj_images.append(make_visual(q_pred, rewards, masks, observations["pixels"]))
        print("finished reward value visuals.")
        return np.concatenate(traj_images, 0)

    @property
    def _save_dict(self):
        """Return the train states that a checkpoint stores."""
        save_dict = {
            "critic": self._critic,
            "target_critic_params": self._target_critic_params,
            "actor": self._actor,
            "temp": self._temp,
        }
        return save_dict

    def restore_checkpoint(self, checkpoint_dir):
        """Load the actor, critic, target critic and temperature from a checkpoint directory."""
        assert pathlib.Path(checkpoint_dir).exists(), f"Checkpoint {checkpoint_dir} does not exist."
        output_dict = checkpoints.restore_checkpoint(checkpoint_dir, self._save_dict)
        self._actor = output_dict["actor"]
        self._critic = output_dict["critic"]
        self._target_critic_params = output_dict["target_critic_params"]
        self._temp = output_dict["temp"]
        print("restored from ", checkpoint_dir)


@jax.jit
def get_value(action, observation, critic):
    """Return the critic ensemble's Q-values for one observation and action."""
    input_collections = {"params": critic.params}
    q_pred = critic.apply_fn(input_collections, observation, action)
    return q_pred


def np_unstack(array, axis):
    """Split an array along an axis into a list of squeezed slices."""
    pieces = np.split(array, array.shape[axis], axis)
    pieces = [piece.squeeze() for piece in pieces]
    return pieces


def make_visual(q_estimates, rewards, masks, images):
    """Plot sampled frames, Q-values, rewards and masks of one trajectory as an RGB image."""
    q_estimates_np = np.stack(q_estimates, 0).squeeze()
    fig, axes = plt.subplots(4, 1, figsize=(8, 12))
    canvas = FigureCanvas(fig)
    plt.xlim([0, len(q_estimates_np)])

    assert len(images.shape) == 5
    images = images[..., -1]  # only taking the most recent image of the stack
    assert images.shape[-1] == 3

    interval = max(1, images.shape[0] // 4)
    selected_images = images[::interval]
    selected_images = np.concatenate(np_unstack(selected_images, 0), 1)

    axes[0].imshow(selected_images)
    if len(q_estimates_np.shape) == 2:
        for q_index in range(q_estimates_np.shape[1]):
            axes[1].plot(q_estimates_np[:, q_index], linestyle="--", marker="o")
    else:
        axes[1].plot(q_estimates_np, linestyle="--", marker="o")
    axes[1].set_ylabel("q values")
    axes[2].plot(rewards, linestyle="--", marker="o")
    axes[2].set_ylabel("rewards")
    axes[2].set_xlim([0, len(rewards)])

    axes[3].plot(masks, linestyle="--", marker="d")
    axes[3].set_ylabel("masks")
    axes[3].set_xlim([0, len(masks)])

    plt.tight_layout()

    canvas.draw()  # draw the canvas, cache the renderer
    out_image = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
    out_image = out_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)
    return out_image
