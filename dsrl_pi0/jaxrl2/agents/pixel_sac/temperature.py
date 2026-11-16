"""Learnable SAC entropy temperature, stored as its logarithm."""

import flax.linen as nn
import jax.numpy as jnp


class Temperature(nn.Module):
    """Flax module that holds the entropy temperature alpha = exp(log_temp)."""

    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        """Return the current temperature as a positive scalar."""
        log_temp = self.param(
            "log_temp", init_fn=lambda key: jnp.full((), jnp.log(self.initial_temperature))
        )
        return jnp.exp(log_temp)
