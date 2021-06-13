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
        self._compute_cum_reward()

    def _compute_gae(self):
        # the last element of the buffer is dummy transition for value calculation
        self.gae = [None for _ in range(len(self.buffer))]
        deltas = [None for _ in range(len(self.buffer))]

        for i in reversed(range(len(self.gae))):
            delta = self.buffer[i].reward + \
                    self.gamma * self.buffer[i].valid * self.values[i + 1] - self.values[i]
            deltas[i] = delta

        for i in reversed(range(len(self.gae))):
            if i == len(self.gae) - 1:
                self.gae[i] = deltas[i]
            else:
                self.gae[i] = deltas[i] + self.buffer[i].valid * (self.lmd * self.gamma) * self.gae[i+1]

    def _compute_cum_reward(self):
        self.cum_reward = [None for _ in range(len(self.buffer))]

        for i in reversed(range(len(self.cum_reward))):
            self.cum_reward[i] = self.values[i] + self.gae[i]

    def create_dataset(self):
        assert self.gae is not None and self.cum_reward is not None

        batch = Transition(*zip(*self.buffer))

        state_batch = np.array(batch.state, dtype=np.float32)
        state_shape = state_batch.shape
        state_batch = state_batch.reshape(-1, *state_shape[2:])

        reward_batch = np.array(self.cum_reward, dtype=np.float32).flatten()

        action_batch = np.array(batch.action).flatten()
        log_prob_batch = np.array(batch.log_prob, dtype=np.float32).flatten()
        gae_batch = np.array(self.gae, dtype=np.float32).flatten()

        state_batch = torch.tensor(state_batch)
        reward_batch = torch.tensor(reward_batch)
        action_batch = torch.tensor(action_batch)
        log_prob_batch = torch.tensor(log_prob_batch)
        gae_batch = torch.tensor(gae_batch)

        values_batch = np.array(self.values, dtype=np.float32)[:-1].flatten()
        values_batch = torch.tensor(values_batch)

        gae_batch -= gae_batch.mean()
        gae_batch /= (gae_batch.std() + 1e-6)

        dataset = TensorDataset(state_batch, reward_batch, action_batch, log_prob_batch, gae_batch, values_batch)

        return dataset

