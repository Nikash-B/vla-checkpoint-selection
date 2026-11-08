"""Tanh-squashed Gaussian policy with a learned, state-dependent standard deviation."""

from typing import Optional, Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp

from jaxrl2.networks.constants import default_init
from jaxrl2.networks.mlp import MLP

# Changed from upstream 7f48937: the unused LearnedStdNormalPolicy is removed, and MLP is imported
# from its module (the package no longer re-exports it).


class TanhMultivariateNormalDiag(distrax.Transformed):
    """Diagonal Gaussian squashed by tanh and rescaled to (low, high)."""

    def __init__(
        self,
        loc: jnp.ndarray,
        scale_diag: jnp.ndarray,
        low: jnp.ndarray,
        high: jnp.ndarray,
    ):
        """Build the tanh bijector chain on top of a diagonal Gaussian."""
        distribution = distrax.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)

        # Changed from upstream 7f48937: low and high are required (the learner always passes
        # them), so the rescaling layer is always built.
        def rescale_from_tanh(x):
            """Map a tanh output in (-1, 1) to the range (low, high)."""
            x = (x + 1) / 2  # (-1, 1) => (0, 1)
            return x * (high - low) + low

        def forward_log_det_jacobian(x):
            """Return the log-determinant of the Jacobian of the rescaling."""
            high_ = jnp.broadcast_to(high, x.shape)
            low_ = jnp.broadcast_to(low, x.shape)
            return jnp.sum(jnp.log(0.5 * (high_ - low_)), -1)

        layers = [
            distrax.Lambda(
                rescale_from_tanh,
                forward_log_det_jacobian=forward_log_det_jacobian,
                event_ndims_in=1,
                event_ndims_out=1,
            )
        ]

        layers.append(distrax.Block(distrax.Tanh(), 1))

        bijector = distrax.Chain(layers)

        super().__init__(distribution=distribution, bijector=bijector)

    def mode(self) -> jnp.ndarray:
        """Return the squashed mode of the underlying Gaussian."""
        return self.bijector.forward(self.distribution.mode())


class LearnedStdTanhNormalPolicy(nn.Module):
    """MLP policy that outputs a tanh-squashed Gaussian with learned log-std."""

    hidden_dims: Sequence[int]
    action_dim: int
    # Changed from upstream 7f48937: low and high are required and come before the defaulted
    # fields (upstream defaulted them to None, which no caller used).
    low: float
    high: float
    dropout_rate: Optional[float] = None
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2

    @nn.compact
    def __call__(self, observations: jnp.ndarray, training: bool = False) -> distrax.Distribution:
        """Return the action distribution for a batch of observations."""
        outputs = MLP(self.hidden_dims, activate_final=True, dropout_rate=self.dropout_rate)(
            observations, training=training
        )

        means = nn.Dense(self.action_dim, kernel_init=default_init(1e-2))(outputs)

        log_stds = nn.Dense(self.action_dim, kernel_init=default_init(1e-2))(outputs)
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = TanhMultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds), low=self.low, high=self.high
        )
        return distribution
