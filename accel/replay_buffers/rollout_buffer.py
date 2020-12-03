import numpy as np
from collections import namedtuple
import torch
from torch.utils.data import TensorDataset

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward', 'valid', 'log_prob'))


class RolloutBuffer:
    def __init__(self, gamma, lmd):
        self.gamma = gamma
        self.lmd = lmd

        self.buffer = []
        self.values = []
        self.gae = None
        self.cum_reward = None

    def clear(self):
        self.buffer = []
        self.values = []
        self.gae = None
        self.cum_reward = None

    def push(self, transition: Transition, v_s):
        self.buffer.append(transition)
        self.values.append(v_s)

    def final_state_value(self, v_s):
        assert len(self.buffer) > 0

        self.values.append(v_s)
        self._compute_gae()

    def _compute_gae(self):
        # the last element of the buffer is dummy transition for value calculation
        self.gae = [None for _ in range(len(self.buffer))]
        deltas = [None for _ in range(len(self.buffer))]
        self.cum_reward = [None for _ in range(len(self.buffer))]

        for i in reversed(range(len(self.gae))):
            delta = self.buffer[i].reward + \
                    self.gamma * self.buffer[i].valid * self.values[i + 1] - self.values[i]
            deltas[i] = delta

        # normalize?
        for i in reversed(range(len(self.gae))):
            if i == len(self.gae) - 1:
                self.gae[i] = deltas[i]
            else:
                self.gae[i] = deltas[i] + self.buffer[i].valid * (self.lmd * self.gamma) * self.gae[i+1]

        for i in reversed(range(len(self.cum_reward))):
            if i == len(self.cum_reward) - 1:
                self.cum_reward[i] = self.buffer[i].reward + self.buffer[i].valid * self.gamma * self.values[i+1]
            else:
                self.cum_reward[i] = self.buffer[i].reward + self.buffer[i].valid * self.gamma * self.cum_reward[i+1]

    def create_dataset(self):
        assert self.gae is not None and self.cum_reward is not None

        batch = Transition(*zip(*self.buffer))

        state_batch = np.array(batch.state, dtype=np.float32)
        state_shape = state_batch.shape
        state_batch = state_batch.reshape(-1, state_shape[2], state_shape[3], state_shape[4])

        reward_batch = np.array(self.cum_reward, dtype=np.float32).flatten()

        action_batch = np.array(batch.action).flatten()
        log_prob_batch = np.array(batch.log_prob, dtype=np.float32).flatten()
        gae_batch = np.array(self.gae, dtype=np.float32).flatten()

        state_batch = torch.tensor(state_batch)
        reward_batch = torch.tensor(reward_batch)
        action_batch = torch.tensor(action_batch)
        log_prob_batch = torch.tensor(log_prob_batch)
        gae_batch = torch.tensor(gae_batch)

        dataset = TensorDataset(state_batch, reward_batch, action_batch, log_prob_batch, gae_batch)

        return dataset

