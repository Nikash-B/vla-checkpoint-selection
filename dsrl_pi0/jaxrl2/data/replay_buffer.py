"""Growable replay buffer that stores transitions and trajectory bounds."""

import collections
from typing import Union

import gym
import gym.spaces
import jax
import numpy as np
from flax.core import frozen_dict

from jaxrl2.types import DatasetDict


def _init_replay_dict(obs_space: gym.Space, capacity: int) -> Union[np.ndarray, DatasetDict]:
    """Allocate empty arrays that match an observation space, recursing into dict spaces."""
    if isinstance(obs_space, gym.spaces.Box):
        return np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
    elif isinstance(obs_space, gym.spaces.Dict):
        data_dict = {}
        for key, subspace in obs_space.spaces.items():
            data_dict[key] = _init_replay_dict(subspace, capacity)
        return data_dict
    else:
        raise TypeError()


# Changed from upstream 7f48937: ReplayBuffer is a plain class (its Dataset base only added an
# unused seed()), and it no longer stores next_actions, which no update reads.
class ReplayBuffer:
    """Replay buffer that doubles its capacity when full and records trajectory bounds."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
    ):
        """Allocate storage for the given spaces and initial capacity."""
        self.observation_space = observation_space
        self.action_space = action_space
        self.capacity = capacity

        print("making replay buffer of capacity ", self.capacity)

        observations = _init_replay_dict(self.observation_space, self.capacity)
        next_observations = _init_replay_dict(self.observation_space, self.capacity)
        actions = np.empty((self.capacity, *self.action_space.shape), dtype=self.action_space.dtype)
        rewards = np.empty((self.capacity,), dtype=np.float32)
        masks = np.empty((self.capacity,), dtype=np.float32)
        discount = np.empty((self.capacity,), dtype=np.float32)

        self.data = {
            "observations": observations,
            "next_observations": next_observations,
            "actions": actions,
            "rewards": rewards,
            "masks": masks,
            "discount": discount,
        }

        self.size = 0
        self._traj_counter = 0
        self._start = 0
        self.traj_bounds = dict()
        # Changed from upstream 7f48937: streaming_buffer_size, always None, is removed.

    def __len__(self) -> int:
        """Return the number of stored transitions."""
        return self.size

    def increment_traj_counter(self):
        """Close the current trajectory and start a new one at the current size."""
        self.traj_bounds[self._traj_counter] = (self._start, self.size)  # [start, end)
        self._start = self.size
        self._traj_counter += 1

    def get_random_trajs(self, num_trajs: int):
        """Return num_trajs whole trajectories drawn uniformly with replacement."""
        # Changed from upstream 7f48937: the next-observation and terminal lists, which no caller
        # reads, are removed, and which_trajs is a local variable instead of an attribute.
        which_trajs = np.random.randint(0, self._traj_counter, num_trajs)
        observations_list = []
        actions_list = []
        rewards_list = []
        masks_list = []

        for traj_index in which_trajs:
            start, end = self.traj_bounds[traj_index]

            # handle this as a dictionary
            obs_dict_curr_traj = dict()
            for key in self.data["observations"]:
                obs_dict_curr_traj[key] = self.data["observations"][key][start:end]
            observations_list.append(obs_dict_curr_traj)

            actions_list.append(self.data["actions"][start:end])
            rewards_list.append(self.data["rewards"][start:end])
            masks_list.append(self.data["masks"][start:end])

        batch = {
            "observations": observations_list,
            "actions": actions_list,
            "rewards": rewards_list,
            "masks": masks_list,
        }
        return batch

    def insert(self, data_dict: DatasetDict):
        """Store one transition, doubling the capacity first if the buffer is full."""
        if self.size == self.capacity:
            # Double the capacity
            observations = _init_replay_dict(self.observation_space, self.capacity)
            next_observations = _init_replay_dict(self.observation_space, self.capacity)
            actions = np.empty(
                (self.capacity, *self.action_space.shape), dtype=self.action_space.dtype
            )
            rewards = np.empty((self.capacity,), dtype=np.float32)
            masks = np.empty((self.capacity,), dtype=np.float32)
            discount = np.empty((self.capacity,), dtype=np.float32)

            data_new = {
                "observations": observations,
                "next_observations": next_observations,
                "actions": actions,
                "rewards": rewards,
                "masks": masks,
                "discount": discount,
            }

            for key in data_new:
                if isinstance(self.data[key], np.ndarray):
                    self.data[key] = np.concatenate((self.data[key], data_new[key]), axis=0)
                elif isinstance(self.data[key], dict):
                    for sub_key in self.data[key]:
                        self.data[key][sub_key] = np.concatenate(
                            (self.data[key][sub_key], data_new[key][sub_key]), axis=0
                        )
                else:
                    raise TypeError()
            self.capacity *= 2

        for key in data_dict:
            if key in self.data:
                if isinstance(data_dict[key], dict):
                    for sub_key in data_dict[key]:
                        self.data[key][sub_key][self.size] = data_dict[key][sub_key]
                else:
                    self.data[key][self.size] = data_dict[key]
        self.size += 1

    # Changed from upstream 7f48937: the unused keys and indx parameters of sample and get_iterator
    # are removed.
    def sample(self, batch_size: int) -> frozen_dict.FrozenDict:
        """Return a frozen batch of batch_size transitions drawn uniformly with replacement."""
        indices = np.random.randint(0, self.size, batch_size)
        data_dict = {}
        for key in self.data:
            if isinstance(self.data[key], np.ndarray):
                data_dict[key] = self.data[key][indices]
            elif isinstance(self.data[key], dict):
                data_dict[key] = {}
                for sub_key in self.data[key]:
                    data_dict[key][sub_key] = self.data[key][sub_key][indices]
            else:
                raise TypeError()

        return frozen_dict.freeze(data_dict)

    def get_iterator(self, batch_size: int, queue_size: int = 2):
        """Yield sampled batches endlessly, prefetching queue_size of them to the device."""
        # See https://flax.readthedocs.io/en/latest/_modules/flax/jax_utils.html#prefetch_to_device
        # queue_size = 2 should be ok for one GPU.

        queue = collections.deque()

        def enqueue(num_batches):
            """Sample num_batches batches and push them onto the device queue."""
            for _ in range(num_batches):
                data = self.sample(batch_size)
                queue.append(jax.device_put(data))

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)

    # Changed from upstream 7f48937: the unused length, compute_action_stats, normalize_actions,
    # save and restore are removed.
