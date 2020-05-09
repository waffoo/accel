import random
import math
import torch


class ExpDecayEpsilonGreedy:
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def act(self, step, action_value):
        sample = random.random()
        eps = self.end + (self.start - self.end) * \
            math.exp(-1. * step / self.decay)

        n_actions = action_value.shape[1]
        if sample > eps:
            return action_value.max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(n_actions)]])
