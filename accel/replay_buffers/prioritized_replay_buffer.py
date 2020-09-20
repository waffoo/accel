from collections import namedtuple
from accel.replay_buffers.sum_tree import SumTree
import random

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward', 'valid'))


class PrioritizedReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = SumTree(capacity)
        self.e = 0.01
        self.a = 0.6

    def _get_priority(self, error):
        return (error + self.e) ** self.a

    def push(self, error, *args):
        p = self._get_priority(error)
        self.memory.add(p, Transition(*args))

    def sample(self, batch_size):
        batch = []
        idx_batch = []
        segment = self.memory.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.memory.get(s)
            batch.append(data)
            idx_batch.append(idx)
        return batch, idx_batch

    def update(self, idx, error):
        p = self._get_priority(error)
        self.memory.update(idx, p)

    def __len__(self):
        return len(self.memory)
