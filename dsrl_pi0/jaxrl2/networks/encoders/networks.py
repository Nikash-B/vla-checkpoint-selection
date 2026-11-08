"""Convolutional image encoder and the module that joins it to a policy or critic head."""

from typing import Dict, Optional, Sequence, Union

import flax.linen as nn
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from jaxrl2.networks.constants import default_init, xavier_init

# Changed from upstream 7f48937: the unused ModuleDef alias is removed.


class Encoder(nn.Module):
    """Stack of 3x3 convolutions with ReLU that flattens images to a feature vector."""

    features: Sequence[int] = (32, 32, 32, 32)
    strides: Sequence[int] = (2, 1, 1, 1)
    padding: str = "VALID"

    @nn.compact
    def __call__(self, observations: jnp.ndarray, training=False) -> jnp.ndarray:
        """Encode uint8 pixel observations into one flat feature vector per sample."""
        assert len(self.features) == len(self.strides)

        x = observations.astype(jnp.float32) / 255.0
        x = jnp.reshape(x, (*x.shape[:-2], -1))

        for num_filters, stride in zip(self.features, self.strides):
            x = nn.Conv(
                num_filters,
                kernel_size=(3, 3),
                strides=(stride, stride),
                kernel_init=default_init(),
                padding=self.padding,
            )(x)
            x = nn.relu(x)

        return x.reshape((*x.shape[:-3], -1))


class PixelMultiplexer(nn.Module):
    """Encode the pixels, optionally pass them through a bottleneck, and call the head network."""

    encoder: Union[nn.Module, list]
    network: nn.Module
    latent_dim: int
    use_bottleneck: bool = True

    @nn.compact
    def __call__(
        self,
        observations: Union[FrozenDict, Dict],
        actions: Optional[jnp.ndarray] = None,
        training: bool = False,
    ):
        """Replace observations['pixels'] with its encoding and apply the head network."""
        observations = FrozenDict(observations)

        x = self.encoder(observations["pixels"], training)
        if self.use_bottleneck:
            x = nn.Dense(self.latent_dim, kernel_init=xavier_init())(x)
            x = nn.LayerNorm()(x)
            x = nn.tanh(x)

        x = observations.copy(add_or_replace={"pixels": x})

        if actions is None:
            return self.network(x, training=training)
        else:
            return self.network(x, actions, training=training)
