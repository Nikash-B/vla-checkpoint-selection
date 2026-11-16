"""State-action value network Q(s, a) for the critic."""

from typing import Callable, Sequence

import flax.linen as nn
import jax.numpy as jnp

from jaxrl2.networks.mlp import MLP

# Changed from upstream 7f48937: the unused type aliases and default_kernel_init are removed, and
# so is the use_action_sep option (MLPActionSep), which no run sets.


class StateActionValue(nn.Module):
    """MLP with layer norm that maps an observation and an action to one Q-value."""

    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray, training: bool = False):
        """Return the Q-value for each observation-action pair, without a trailing axis."""
        inputs = {"states": observations, "actions": actions}
        critic = MLP((*self.hidden_dims, 1), activations=self.activations, use_layer_norm=True)(
            inputs, training=training
        )
        return jnp.squeeze(critic, -1)
