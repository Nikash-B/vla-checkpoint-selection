"""Weight initializers shared by the networks."""

import flax.linen as nn
import jax.numpy as jnp


def default_init(scale: float = jnp.sqrt(2)):
    """Return an orthogonal initializer with the given scale."""
    return nn.initializers.orthogonal(scale)


def xavier_init():
    """Return a Xavier (Glorot) normal initializer."""
    return nn.initializers.xavier_normal()


# Changed from upstream 7f48937: kaiming_init, used only by the removed encoders, is removed.
