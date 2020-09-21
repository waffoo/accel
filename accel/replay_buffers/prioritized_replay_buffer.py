from collections import namedtuple
from accel.replay_buffers.sum_tree import SumTree
import random
import numpy as np

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward', 'valid'))


class PrioritizedReplayBuffer(object):
    def __init__(self, capacity, alpha=0.6, beta0=0.4, eps=0.01, beta_steps=int(2e5)):
        self.capacity = capacity
        self.memory = SumTree(capacity)
        self.eps = eps
        self.beta0 = beta0
        self.alpha = alpha
        self.steps = 0
        self.beta_steps = beta_steps
        self.max_err = 1e-15

    def _get_priority(self, error):
        return (error + self.eps) ** self.alpha

    def max_priority(self):
        # TODO max maybe too big
        return self._get_priority(self.max_err)

    def push(self, error, *args):
        self.max_err = max(error, self.max_err)
        self.steps += 1

        p = self._get_priority(error)
        self.memory.add(p, Transition(*args))

    def sample(self, batch_size):
        batch = []
        idx_batch = []
        weights = []
        pris = []
        total_pri = self.memory.total()

        # TODO replace it with common annealing function
        progress = min(1.0, self.steps / self.beta_steps)
        beta = self.beta0 + (1.0 - self.beta0) * progress

        for _ in range(batch_size):
            s = random.uniform(0, self.memory.total())
            idx, pri, data = self.memory.get(s)
            prob = pri / total_pri

            batch.append(data)
            idx_batch.append(idx)
            pris.append(pri)

            weight = (len(self.memory) * prob) ** (-beta)
            weights.append(weight)

        weights = np.array(weights)
        _initial_data_ratio = (np.array(pris, dtype=np.float32) == self.max_err).mean()

        return batch, idx_batch, weights / weights.max()

    def update(self, idx, error):
        p = self._get_priority(error)
        self.memory.update(idx, p)

    def __len__(self):
        return len(self.memory)
