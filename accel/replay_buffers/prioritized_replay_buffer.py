import random
from collections import deque, namedtuple

import numpy as np

from accel.replay_buffers.binary_tree import MinTree, SumTree

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward', 'valid'))


class PrioritizedReplayBuffer(object):
    def __init__(self, capacity, alpha=0.6, beta0=0.4,
                 eps=1e-6, beta_steps=int(2e5), nstep=1):
        self.capacity = capacity
        self.sum_tree = SumTree(capacity)
        self.min_tree = MinTree(capacity)
        self.data = np.zeros(capacity, dtype=object)
        self.eps = eps
        self.beta0 = beta0
        self.alpha = alpha
        self.steps = 0
        self.beta_steps = beta_steps
        self.max_err = 1.0
        self.write = 0
        self.len = 0
        self.nstep = nstep
        self.tmp_buffer = deque(maxlen=self.nstep)

    def _get_priority(self, error):
        return (error + self.eps) ** self.alpha

    def push(self, *args):
        self.steps += 1

        p = self._get_priority(self.max_err)

        transition = Transition(*args)
        self.tmp_buffer.append(transition)

        if not transition.valid:
            while len(self.tmp_buffer) > 0:
                self.data[self.write] = list(self.tmp_buffer)
                self.sum_tree.update(self.write, p)
                self.min_tree.update(self.write, p)

                self.write += 1

                self.len = max(self.len, self.write)
                if self.write >= self.capacity:
                    self.write = 0

                self.tmp_buffer.popleft()
        else:
            if len(self.tmp_buffer) == self.tmp_buffer.maxlen:
                self.data[self.write] = list(self.tmp_buffer)
                self.sum_tree.update(self.write, p)
                self.min_tree.update(self.write, p)

                self.write += 1

                self.len = max(self.len, self.write)
                if self.write >= self.capacity:
                    self.write = 0

        '''
        self.data[self.write] = transition

        self.sum_tree.update(self.write, p)
        self.min_tree.update(self.write, p)

        self.write += 1

        self.len = max(self.len, self.write)
        if self.write >= self.capacity:
            self.write = 0
        '''

    def sample(self, batch_size):
        batch = []
        idx_batch = []
        weights = []
        pris = []
        total_pri = self.sum_tree.top()

        # TODO replace it with common annealing function
        progress = min(1.0, self.steps / self.beta_steps)
        beta = self.beta0 + (1.0 - self.beta0) * progress

        prob_min = self.min_tree.top() / total_pri
        max_weight = (self.len * prob_min) ** (-beta)

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
        _initial_data_ratio = (
            np.array(pris, dtype=np.float32) == self._get_priority(self.max_err)).mean()

        return batch, idx_batch, weights / max_weight

    def update(self, data_idx, error):
        self.max_err = max(error, self.max_err)
        p = self._get_priority(error)
        self.sum_tree.update(data_idx, p)
        self.min_tree.update(data_idx, p)

    def __len__(self):
        return self.len
