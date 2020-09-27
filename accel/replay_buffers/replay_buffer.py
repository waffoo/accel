from collections import namedtuple, deque
import random

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward', 'valid'))


class ReplayBuffer(object):

    def __init__(self, capacity, nstep=1):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.nstep = nstep
        self.tmp_buffer = deque(maxlen=self.nstep)

    def push(self, *args):
        """Saves a transition."""

        transition = Transition(*args)

        self.tmp_buffer.append(transition)

        if not transition.valid:
            while len(self.tmp_buffer) > 0:
                if len(self.memory) < self.capacity:
                    self.memory.append(None)
                self.memory[self.position] = list(self.tmp_buffer)
                self.position = (self.position + 1) % self.capacity
                self.tmp_buffer.popleft()
        else:
            if len(self.tmp_buffer) == self.tmp_buffer.maxlen:
                if len(self.memory) < self.capacity:
                    self.memory.append(None)
                self.memory[self.position] = list(self.tmp_buffer)
                self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
