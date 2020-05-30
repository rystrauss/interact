"""Provides the experience replay buffer used by the DQN agent.

This implementation is adapted from OpenAI Baselines:
https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

Author: Ryan Strauss
"""

import numpy as np


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
