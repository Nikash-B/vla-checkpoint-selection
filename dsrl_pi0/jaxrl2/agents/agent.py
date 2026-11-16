"""Base agent with action sampling and checkpoint saving shared by the learners."""

import numpy as np
from flax.training import checkpoints
from flax.training.train_state import TrainState

from jaxrl2.agents.common import eval_actions_jit, sample_actions_jit
from jaxrl2.types import PRNGKey


class Agent:
    """Agent that holds an actor, a critic and a random key."""

    # Changed from upstream 7f48937: the unused eval_log_probs, eval_mse and eval_reward_function
    # are removed, and so are the _save_dict and restore_checkpoint stubs that PixelSACLearner
    # overrides. get_batch_stats is removed with the actor_batch_stats argument of the jitted
    # helpers in jaxrl2/agents/common.py.

    _actor: TrainState
    _critic: TrainState
    _rng: PRNGKey

    def eval_actions(self, observations: np.ndarray) -> np.ndarray:
        """Return the deterministic (mode) actions of the actor for the observations."""
        actions = eval_actions_jit(self._actor.apply_fn, self._actor.params, observations)
        return np.asarray(actions)

    def sample_actions(self, observations: np.ndarray) -> np.ndarray:
        """Sample actions from the actor and advance the random key."""
        rng, actions = sample_actions_jit(
            self._rng,
            self._actor.apply_fn,
            self._actor.params,
            observations,
        )

        self._rng = rng
        return np.asarray(actions)

    def save_checkpoint(self, checkpoint_dir, step, keep_every_n_steps):
        """Save the agent state to checkpoint_dir under the given step."""
        checkpoints.save_checkpoint(
            checkpoint_dir,
            self._save_dict,
            step,
            prefix="checkpoint",
            overwrite=False,
            keep_every_n_steps=keep_every_n_steps,
        )
