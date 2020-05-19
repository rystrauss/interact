import gym


class Monitor(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)

        self.rewards = None

    def reset(self, **kwargs):
        self.rewards = []
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.rewards.append(reward)

        if done:
            ep_rew = sum(self.rewards)
            eplen = len(self.rewards)
            ep_info = {'r': round(ep_rew, 6), 'l': eplen}
            info['episode'] = ep_info

        return observation, reward, done, info
