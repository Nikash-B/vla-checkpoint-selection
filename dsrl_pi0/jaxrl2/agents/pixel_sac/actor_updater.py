"""Actor update step for pixel-based SAC."""

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from jaxrl2.types import DatasetDict, Params, PRNGKey


def update_actor(
    key: PRNGKey,
    actor: TrainState,
    critic: TrainState,
    temp: TrainState,
    batch: DatasetDict,
    critic_reduction: str = "min",
) -> Tuple[TrainState, Dict[str, float]]:
    """Take one gradient step on the SAC actor loss and return the new actor and its logs."""
    _, key_act = jax.random.split(key, num=2)

    # Changed from upstream 7f48937: the cross_norm option and the batch_stats branches are
    # removed (no network has batch norm), and so is next_dist, which was computed but never used.
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        """Compute the entropy-regularised actor loss and the values to log."""
        dist = actor.apply_fn({"params": actor_params}, batch["observations"])

        # For logging only
        mean_dist = dist.distribution._loc
        std_diag_dist = dist.distribution._scale_diag
        mean_dist_norm = jnp.linalg.norm(mean_dist, axis=-1)
        std_dist_norm = jnp.linalg.norm(std_diag_dist, axis=-1)

        actions, log_probs = dist.sample_and_log_prob(seed=key_act)

        qs = critic.apply_fn({"params": critic.params}, batch["observations"], actions)

        if critic_reduction == "min":
            q = qs.min(axis=0)
        elif critic_reduction == "mean":
            q = qs.mean(axis=0)
        else:
            raise ValueError(f"Invalid critic reduction: {critic_reduction}")
        actor_loss = (log_probs * temp.apply_fn({"params": temp.params}) - q).mean()

        things_to_log = {
            "actor_loss": actor_loss,
            "entropy": -log_probs.mean(),
            "q_pi_in_actor": q.mean(),
            "mean_pi_norm": mean_dist_norm.mean(),
            "std_pi_norm": std_dist_norm.mean(),
            "mean_pi_avg": mean_dist.mean(),
            "mean_pi_max": mean_dist.max(),
            "mean_pi_min": mean_dist.min(),
            "std_pi_avg": std_diag_dist.mean(),
            "std_pi_max": std_diag_dist.max(),
            "std_pi_min": std_diag_dist.min(),
        }
        return actor_loss, things_to_log

    grads, info = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    new_actor = actor.apply_gradients(grads=grads)

    return new_actor, info
