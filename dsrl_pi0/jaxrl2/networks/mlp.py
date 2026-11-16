"""Multilayer perceptron that flattens dictionary observations before its dense layers."""

from typing import Callable, Optional, Sequence, Union

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict

from jaxrl2.networks.constants import default_init


def _flatten_dict(x: Union[FrozenDict, jnp.ndarray]):
    """Concatenate the entries of a (nested) dict into one array, flattening action chunks."""
    if hasattr(x, "values"):
        flat_parts = []
        for key, value in sorted(x.items()):
            if key == "state":  # flatten action chunk to 1D
                flat_parts.append(
                    jnp.reshape(value, [*value.shape[:-2], np.prod(value.shape[-2:])])
                )
            # Changed from upstream 7f48937: no observation has a "prev_action" key, so only
            # "actions" is matched.
            elif key == "actions":
                if value.ndim > 2:
                    # deal with action chunk
                    flat_parts.append(
                        jnp.reshape(value, [*value.shape[:-2], np.prod(value.shape[-2:])])
                    )
                else:
                    flat_parts.append(value)
            else:
                flat_parts.append(_flatten_dict(value))
        return jnp.concatenate(flat_parts, -1)
    else:
        return x


# Changed from upstream 7f48937: _flatten_dict_special and MLPActionSep, used only by the critic's
# use_action_sep option that no run sets, are removed.


class MLP(nn.Module):
    """Stack of dense layers with optional dropout and layer norm before each activation."""

    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: bool = False
    dropout_rate: Optional[float] = None
    init_scale: float = 1.0
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Flatten the input and apply the dense layers."""
        x = _flatten_dict(x)

        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init(self.init_scale))(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.dropout_rate is not None:
                    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
                if self.use_layer_norm:
                    x = nn.LayerNorm()(x)
                x = self.activations(x)
        return x
