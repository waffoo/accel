import math
import random

import torch


class EpsilonGreedyBase:
    def calc_eps(self, step):
        raise NotImplementedError

    def act(self, step, action_value, greedy=False):
        sample = random.random()
        eps = self.calc_eps(step)

        n_actions = action_value.shape[1]
        if greedy or sample > eps:
            return action_value.max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(n_actions)]])


class ConstantEpsilonGreedy(EpsilonGreedyBase):
    def __init__(self, eps):
        self.eps = eps

    def calc_eps(self, step):
        return self.eps


class LinearDecayEpsilonGreedy(EpsilonGreedyBase):
    def __init__(self, start_eps, end_eps, decay_steps):
        self.start_eps = start_eps
        self.end_eps = end_eps
        self.decay_steps = decay_steps

    def calc_eps(self, step):
        if step >= self.decay_steps:
            return self.end_eps

        return self.start_eps - \
            (self.start_eps - self.end_eps) * step / self.decay_steps


class ExpDecayEpsilonGreedy(EpsilonGreedyBase):
    def __init__(self, start_eps, end_eps, decay):
        self.start_eps = start_eps
        self.end_eps = end_eps
        self.decay = decay

    def calc_eps(self, step):
        return self.end_eps + (self.start_eps - self.end_eps) * \
            math.exp(-1. * step / self.decay)
