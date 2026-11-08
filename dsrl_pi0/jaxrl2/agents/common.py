"""Jitted helpers that query the actor for deterministic and sampled actions."""

from functools import partial
from typing import Callable, Tuple

import distrax
import jax
import jax.numpy as jnp
import numpy as np

from jaxrl2.types import Params, PRNGKey

# Changed from upstream 7f48937: the unused _unpack, eval_log_prob_jit, eval_mse_jit,
# eval_reward_function_jit and ModuleDict are removed, and so is the actor_batch_stats argument
# of the two helpers below (no network has batch norm, so it was always None).


@partial(jax.jit, static_argnames="actor_apply_fn")
def eval_actions_jit(
    actor_apply_fn: Callable[..., distrax.Distribution],
    actor_params: Params,
    observations: np.ndarray,
) -> jnp.ndarray:
    """Return the mode of the actor's action distribution for the observations."""
    dist = actor_apply_fn({"params": actor_params}, observations, training=False, mutable=False)
    return dist.mode()


@partial(jax.jit, static_argnames="actor_apply_fn")
def sample_actions_jit(
    rng: PRNGKey,
    actor_apply_fn: Callable[..., distrax.Distribution],
    actor_params: Params,
    observations: np.ndarray,
) -> Tuple[PRNGKey, jnp.ndarray]:
    """Sample actions from the actor and return the advanced RNG key with them."""
    dist = actor_apply_fn({"params": actor_params}, observations)
    rng, key = jax.random.split(rng)
    return rng, dist.sample(seed=key)
