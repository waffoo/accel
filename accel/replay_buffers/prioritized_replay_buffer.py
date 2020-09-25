from collections import namedtuple
from accel.replay_buffers.sum_tree import SumTree
import random
import numpy as np

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward', 'valid'))


class PrioritizedReplayBuffer(object):
    def __init__(self, capacity, alpha=0.6, beta0=0.4, eps=1e-6, beta_steps=int(2e5)):
        self.capacity = capacity
        self.sum_tree = SumTree(capacity)
        self.data = np.zeros(capacity, dtype=object)
        self.eps = eps
        self.beta0 = beta0
        self.alpha = alpha
        self.steps = 0
        self.beta_steps = beta_steps
        self.max_err = 1e-15
        self.write = 0
        self.len = 0

    def _get_priority(self, error):
        return (error + self.eps) ** self.alpha

    def max_error(self):
        # TODO max maybe too big
        return self.max_err

    def push(self, error, *args):
        self.max_err = max(error, self.max_err)
        self.steps += 1

        p = self._get_priority(error)

        self.data[self.write] = Transition(*args)

        self.sum_tree.update(self.write, p)

        self.write += 1

        self.len = max(self.len, self.write)
        if self.write >= self.capacity:
            self.write = 0

    def sample(self, batch_size):
        batch = []
        idx_batch = []
        weights = []
        pris = []
        total_pri = self.sum_tree.top()

        # TODO replace it with common annealing function
        progress = min(1.0, self.steps / self.beta_steps)
        beta = self.beta0 + (1.0 - self.beta0) * progress

        segment = total_pri / batch_size

        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            s = random.uniform(a, b)
            data_idx, pri = self.sum_tree.get(s)
            prob = pri / total_pri

            batch.append(self.data[data_idx])
            idx_batch.append(data_idx)
            pris.append(pri)

            weight = (self.len * prob) ** (-beta)
            weights.append(weight)

        weights = np.array(weights)
        _initial_data_ratio = (np.array(pris, dtype=np.float32) == self._get_priority(self.max_err)).mean()

        return batch, idx_batch, weights / weights.max()

    def update(self, data_idx, error):
        self.max_err = max(error, self.max_err)
        p = self._get_priority(error)
        self.sum_tree.update(data_idx, p)

    def __len__(self):
        return self.len
