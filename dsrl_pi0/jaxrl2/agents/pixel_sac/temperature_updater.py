"""Gradient step on the SAC entropy temperature (alpha)."""

from typing import Dict, Tuple

import jax
from flax.training.train_state import TrainState


def update_temperature(
    temp: TrainState, entropy: float, target_entropy: float
) -> Tuple[TrainState, Dict[str, float]]:
    """Take one gradient step on the temperature toward the target entropy."""

    def temperature_loss_fn(temp_params):
        """Return the temperature loss and the logged temperature values."""
        temperature = temp.apply_fn({"params": temp_params})
        temp_loss = temperature * (entropy - target_entropy).mean()
        return temp_loss, {"temperature": temperature, "temperature_loss": temp_loss}

    grads, info = jax.grad(temperature_loss_fn, has_aux=True)(temp.params)
    new_temp = temp.apply_gradients(grads=grads)

    return new_temp, info
