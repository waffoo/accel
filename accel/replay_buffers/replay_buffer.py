import os
import random
from collections import deque, namedtuple

import numpy as np

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward', 'valid'))


class ReplayBuffer(object):

    def __init__(self, capacity, nstep=1, record=False,
                 record_size=1_000_000, record_outdir=None):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.nstep = nstep
        self.tmp_buffer = deque(maxlen=self.nstep)
        self.record = record

        if record:
            assert record_outdir is not None
            self.reset_record()
            self.record_size = record_size
            self.record_cnt = 0
            self.record_outdir = record_outdir
            os.makedirs(self.record_outdir, exist_ok=True)

    def reset_record(self):
        self.state_array = []
        self.action_array = []
        self.reward_array = []
        self.terminal_array = []

    def record_transition(self, transition):
        self.state_array.append(transition.state)
        self.action_array.append(transition.action)
        self.reward_array.append(transition.reward)
        self.terminal_array.append(1 - transition.valid)

        if len(self.state_array) == self.record_size:
            self.output_transitions(os.path.join(
                self.record_outdir, f'{self.record_cnt}.npz'))
            self.reset_record()
            self.record_cnt += 1

    def output_transitions(self, filename='hoge.npz'):
        np.savez_compressed(filename,
                            state=np.array(self.state_array, dtype=np.uint8),
                            action=np.array(self.action_array, dtype=np.uint8),
                            reward=np.array(self.reward_array,
                                            dtype=np.float32),
                            terminal=np.array(self.terminal_array, dtype=np.uint8))
        print(f'{filename} was successfully saved!')

    def push(self, *args):
        """Saves a transition."""

        transition = Transition(*args)

        self.tmp_buffer.append(transition)
        if self.record:
            self.record_transition(transition)

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
