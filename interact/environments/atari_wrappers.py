"""This module provides wrappers for Atari environments.

These wrappers are adapted from OpenAI's Baselines.

Author: Ryan Strauss
"""

from collections import deque

import cv2
import gym
import numpy as np
from gym import spaces

cv2.ocl.setUseOpenCL(False)


class NoopResetEnv(gym.Wrapper):
    """Sample initial states by taking random number of no-ops on reset.

    No-op is assumed to be action 0.

    Args:
        env: the environment to wrap
        noop_max: the maximum value of no-ops to run
    """

    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        self.env.reset(**kwargs)

        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        assert noops > 0

        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)

        return obs

    def step(self, action):
        return self.env.step(action)


class FireResetEnv(gym.Wrapper):
    """Take action on reset for environments that are fixed until firing.

    Args:
        env: the environment to wrap
    """

    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)

        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)

        return obs

    def step(self, action):
        return self.env.step(action)


class EpisodicLifeEnv(gym.Wrapper):
    """Make end-of-life == end-of-episode, but only reset on true game over.

    Done by DeepMind for DQN and co. since it helps value estimation.

    Args:
        env: the environment to wrap
    """

    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done

        # Check current lives, make loss of life terminal, then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            # For Qbert sometimes we stay in lives == 0 condition for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Calls the Gym environment reset, only when lives are exhausted.

        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.

        Args:
            kwargs: Extra keywords passed to env.reset() call

        Returns:
            The first observation of the environment.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    """Return only every `skip`-th frame.

    Args:
        env: the environment to wrap
        skip: number of `skip`-th frame
    """

    def __init__(self, env, skip=4):
        super().__init__(env)
        # Most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=env.observation_space.dtype)
        self._skip = skip

    def step(self, action):
        """Step the environment with the given action, repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break

        # Note that the observation on the done=True frame doesn't matter
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ClipRewardEnv(gym.RewardWrapper):
    """Clips the reward to {+1, 0, -1} by its sign.

    Args:
        env: the environment to wrap
    """

    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):
    """Warp frames to 84x84 as done in the Nature paper and later work.

    Args:
        env: the environment to wrap
    """

    def __init__(self, env):
        super().__init__(env)
        self.width = 84
        self.height = 84
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, 1),
                                            dtype=env.observation_space.dtype)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class FrameStack(gym.Wrapper):
    """Stacks the `n_frames` last frames.

    Returns lazy array, which is much more memory efficient.

    Args:
        env: the environment to wrap
        n_frames: the number of frames to stack

    See Also:
        interact.common.atari_wrappers.LazyFrames
    """

    def __init__(self, env, n_frames):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque([], maxlen=n_frames)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * n_frames),
                                            dtype=env.observation_space.dtype)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._get_ob()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.n_frames
        return LazyFrames(list(self.frames))


class ScaledFloatFrame(gym.ObservationWrapper):
    """Scales an 8-bit image observation to the range [0,1].

    Args:
        env: the environment to be wrapped
    """

    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = spaces.Box(low=0, high=1.0, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        return np.array(observation).astype(np.float32) / 255.0


class LazyFrames:
    """This object ensures that common frames between the observations are only stored once.

    It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay buffers.

    This object should only be converted to np.ndarray before being passed to the model.

    Args:
        frames: environment frames
    """

    def __init__(self, frames):
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=2)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]


def wrap_atari(env, episode_life=True, clip_rewards=True, frame_stack=True, scale=False, noop_max=30):
    """Configures an Atari environment with common modifications.

    Args:
        env: the atari environment
        episode_life: wrap the episode life wrapper
        clip_rewards: wrap the reward clipping wrapper
        frame_stack: wrap the frame stacking wrapper
        scale: wrap the scaling observation wrapper
        noop_max: Maximum number of noops at the beginning of an episode.

    Returns:
        The wrapped atari environment.
    """
    assert 'NoFrameskip' in env.spec.id

    env = NoopResetEnv(env, noop_max=noop_max)
    env = MaxAndSkipEnv(env, skip=4)

    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env
