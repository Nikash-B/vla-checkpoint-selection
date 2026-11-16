"""Critic update for the pixel SAC agent: one gradient step on the TD loss."""

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from jaxrl2.types import DatasetDict, Params, PRNGKey


def update_critic(
    key: PRNGKey,
    actor: TrainState,
    critic: TrainState,
    target_critic: TrainState,
    temp: TrainState,
    batch: DatasetDict,
    discount: float,
    critic_reduction: str = "min",
) -> Tuple[TrainState, Dict[str, float]]:
    """Take one gradient step on the critic toward the target-critic TD backup."""
    next_action_dist = actor.apply_fn({"params": actor.params}, batch["next_observations"])
    next_actions, next_log_probs = next_action_dist.sample_and_log_prob(seed=key)
    next_qs = target_critic.apply_fn(
        {"params": target_critic.params}, batch["next_observations"], next_actions
    )
    if critic_reduction == "min":
        next_q = next_qs.min(axis=0)
    elif critic_reduction == "mean":
        next_q = next_qs.mean(axis=0)
    else:
        # Changed from upstream 7f48937: NotImplementedError instead of the NotImplemented
        # constant, which is not an exception. No run reaches this branch.
        raise NotImplementedError(critic_reduction)

    # Changed from upstream 7f48937: the backup_entropy option, which no run sets, is removed.
    target_q = batch["rewards"] + batch["discount"] * batch["masks"] * next_q

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        """Return the mean squared TD error and the logged critic statistics."""
        qs = critic.apply_fn({"params": critic_params}, batch["observations"], batch["actions"])
        critic_loss = ((qs - target_q) ** 2).mean()
        return critic_loss, {
            "critic_loss": critic_loss,
            "q": qs.mean(),
            "target_actor_entropy": -next_log_probs.mean(),
            "next_actions_sampled": next_actions.mean(),
            "next_log_probs": next_log_probs.mean(),
            "next_q_pi": next_qs.mean(),
            "target_q": target_q.mean(),
            "next_actions_mean": next_actions.mean(),
            "next_actions_std": next_actions.std(),
            "next_actions_min": next_actions.min(),
            "next_actions_max": next_actions.max(),
        }

    grads, info = jax.grad(critic_loss_fn, has_aux=True)(critic.params)
    new_critic = critic.apply_gradients(grads=grads)

    return new_critic, info
