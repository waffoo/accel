import torch
from collections import namedtuple

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward', 'valid'))

class PPO:
    def __init__(self, envs, eval_envs, optimizer, horizon=128):
        self.envs = envs
        self.eval_envs = eval_envs
        self.optimizer = optimizer
        self.horizon = horizon

    def act(self, obs, greedy=False):
        pass

    def update(self, obs, action, next_obs, reward, valid):
        pass

    def train(self):
        pass

    def run(self):
        self.envs.reset()
        self.eval_envs.reset()

        for x in range(10):
            actions = self.envs.action_space.sample()
            self.envs.step(actions)
            print(actions)




        # collect trajectories
        for i in range(self.horizon):
            pass

        # compute reward to go R_t

        # compute advantage estimates A_t using GAE

        # update the policy by maximizing PPO-Clip objective

        # update value function

