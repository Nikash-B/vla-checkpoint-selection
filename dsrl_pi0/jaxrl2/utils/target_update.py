"""Polyak (soft) update of target network parameters."""

import jax

from jaxrl2.types import Params


def soft_target_update(critic_params: Params, target_critic_params: Params, tau: float) -> Params:
    """Move the target parameters a fraction tau of the way towards the online parameters."""
    new_target_params = jax.tree_util.tree_map(
        lambda param, target_param: param * tau + target_param * (1 - tau),
        critic_params,
        target_critic_params,
    )
    return new_target_params


# Changed from upstream 7f48937: the unused pmap variant soft_target_update_parallel is removed.
