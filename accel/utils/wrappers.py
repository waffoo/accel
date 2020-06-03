import gym


class RewardScaler(gym.RewardWrapper):
    def __init__(self, env, scale=1.0):
        super().__init__(env)
        self.scale = scale

    def reward(self, reward):
        return reward * self.scale
