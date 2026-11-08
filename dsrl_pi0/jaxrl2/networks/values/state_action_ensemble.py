"""An ensemble of state-action value networks with independent parameters."""

from typing import Callable, Sequence

import flax.linen as nn
import jax.numpy as jnp

from jaxrl2.networks.values.state_action_value import StateActionValue


class StateActionEnsemble(nn.Module):
    """Stack num_qs state-action value networks with vmap over their parameters."""

    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    num_qs: int = 2
    # Changed from upstream 7f48937: the use_action_sep option, which no run sets, is removed.

    @nn.compact
    def __call__(self, states, actions, training: bool = False):
        """Return the Q-values of every ensemble member, stacked on axis 0."""
        vmap_critic = nn.vmap(
            StateActionValue,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num_qs,
        )
        qs = vmap_critic(self.hidden_dims, activations=self.activations)(states, actions, training)
        return qs
