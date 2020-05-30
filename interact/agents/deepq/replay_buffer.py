"""Provides the experience replay buffer used by the DQN agent.

This implementation is adapted from OpenAI Baselines:
https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

Author: Ryan Strauss
"""

import numpy as np

from interact.common.segment_tree import SumSegmentTree, MinSegmentTree


class ReplayBuffer:
    """A memory buffer used for storing experiences from an environment."""

    def __init__(self, buffer_size):
        """Initializes the experience replay buffer.

        Args:
            buffer_size: The size of the buffer.
        """
        self._buffer = []
        self._max_size = buffer_size
        self._next_index = 0

    def __len__(self):
        return len(self._buffer)

    def add(self, state, action, reward, next_state, done):
        """Adds and experience to the buffer.

        Args:
            state: An observed state.
            action: The action that was taken in the state.
            reward: The reward that was gained.
            next_state: The new observed state.
            done: Whether or not the episode is done.

        Returns:
            None
        """
        experience = (state, action, reward, next_state, done)
        if self._next_index >= len(self._buffer):
            self._buffer.append(experience)
        else:
            self._buffer[self._next_index] = experience
        self._next_index = (self._next_index + 1) % self._max_size

    def _encode_sample(self, indices):
        """Extracts a sample from the buffer.

        Args:
            indices: The buffer indices to be put into the sample.

        Returns:
            The tuple (states, actions, rewards, next_states, dones).
        """
        states, actions, rewards, next_states, dones = [], [], [], [], []

        for i in indices:
            state, action, reward, next_state, done = self._buffer[i]

            states.append(np.array(state, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            next_states.append(np.array(next_state, copy=False))
            dones.append(done)

        return np.array(states), np.array(actions), np.array(rewards), np.array(
            next_states), np.array(dones)

    def sample(self, batch_size):
        """Samples a batch from the buffer.

        Args:
            batch_size: The size of the batch to sample.

        Returns:
            A sample from the replay buffer with `batch_size` experiences.
        """
        indices = np.random.choice(len(self._buffer), batch_size)
        return self._encode_sample(indices)


class PrioritizedReplayBuffer(ReplayBuffer):
    """A prioritized experience replay buffer.

    Args:
        size (int): Max number of transitions to store in the buffer.
            When the buffer overflows the old memories are dropped.
        alpha (float): The amount of prioritization that is used
            (0 - no prioritization, 1 - full prioritization).

    See Also:
        ReplayBuffer

    References:
        https://arxiv.org/abs/1511.05952
    """

    def __init__(self, size, alpha):
        super().__init__(size)
        if alpha < 0:
            raise ValueError('alpha must be at least zero.')

        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        index = self._next_index
        super().add(*args, **kwargs)
        self._it_sum[index] = self._max_priority ** self._alpha
        self._it_min[index] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self._buffer) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = np.random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.

        Compared to ExperienceReplay.sample, it also returns importance weights and indices of sampled experiences.

        Args:
            batch_size (int): Number of transitions to sample.
            beta (float): To what degree to use importance weights
                (0 - no corrections, 1 - full correction)

        Returns:
            obs_batch: np.array
                batch of observations
            act_batch: np.array
                batch of actions executed given obs_batch
            rew_batch: np.array
                rewards received as results of executing act_batch
            next_obs_batch: np.array
                next set of observations seen after executing act_batch
            done_mask: np.array
                done_mask[i] = 1 if executing act_batch[i] resulted in
                the end of an episode and 0 otherwise.
            weights: np.array
                Array of shape (batch_size,) and dtype np.float32
                denoting importance weight of each sampled transition
            indices: np.array
                Array of shape (batch_size,) and dtype np.int32
                indexes in buffer of sampled experiences
        """
        if beta <= 0:
            raise ValueError('beta must be > 0.')

        indices = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._buffer)) ** (-beta)

        for idx in indices:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._buffer)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(indices)

        return encoded_sample, weights, indices

    def update_priorities(self, indices, priorities):
        """Update priorities of sampled transitions.

        Sets priority of transition at index idxes[i] in buffer to priorities[i].

        Args:
            indices ([int]): List of idxes of sampled transitions.
            priorities ([float]): List of updated priorities corresponding
                to transitions at the sampled idxes denoted by variable `idxes`.
        """
        assert len(indices) == len(priorities)
        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._buffer)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)
