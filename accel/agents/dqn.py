import torch
import numpy as np
import copy
import torch.nn.functional as F

from accel.replay_buffers.replay_buffer import Transition


class DQN:
    def __init__(self, q_func, optimizer, replay_buffer, gamma, explorer,
                 device,
                 batch_size=32,
                 update_interval=4,
                 target_update_interval=200, huber=False):
        self.q_func = q_func.to(device)
        self.target_q_func = copy.deepcopy(self.q_func).to(device)
        self.optimizer = optimizer
        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.explorer = explorer
        self.device = device
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.target_update_interval = target_update_interval
        self.huber = huber
        self.total_steps = 0

        self.prev_target_update_time = 0

        self.target_q_func.eval()

    def act(self, obs, greedy=False):
        obs = torch.tensor(obs, device=self.device, dtype=torch.float32)

        with torch.no_grad():
            action_value = self.q_func(obs[None])

        action = self.explorer.act(
            self.total_steps, action_value, greedy=greedy)
        return action.item()

    def update(self, obs, action, next_obs, reward, done):
        if done:
            next_obs = None

        self.replay_buffer.push(obs, action, next_obs, np.float32(reward))

        self.total_steps += 1
        if self.total_steps % self.update_interval == 0:
            self.train()

    def non_final_next_state_value(self, non_final_next_states):
        return self.target_q_func(non_final_next_states).max(1)[0].detach()

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(
            lambda s: s is not None, batch.next_state)), device=self.device)
        non_final_next_states = torch.tensor(
            [s for s in batch.next_state if s is not None], device=self.device, dtype=torch.float32)

        state_batch = torch.tensor(
            batch.state, device=self.device, dtype=torch.float32)
        action_batch = torch.tensor(
            batch.action, device=self.device, dtype=torch.int64).unsqueeze(1)
        reward_batch = torch.tensor(
            batch.reward, device=self.device, dtype=torch.float32)

        state_action_values = self.q_func(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)

        next_state_values[non_final_mask] = self.non_final_next_state_value(
            non_final_next_states)

        expected_state_action_values = (
            next_state_values * self.gamma) + reward_batch

        if self.huber:
            loss = F.smooth_l1_loss(state_action_values,
                                    expected_state_action_values.unsqueeze(1))
        else:
            loss = F.mse_loss(state_action_values,
                              expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.q_func.parameters():
        #    param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        if self.total_steps - self.prev_target_update_time > self.target_update_interval:
            self.target_q_func.load_state_dict(self.q_func.state_dict())
            self.prev_target_update_time = self.total_steps


class DoubleDQN(DQN):
    def non_final_next_state_value(self, non_final_next_states):
        next_action_batch = self.q_func(non_final_next_states).max(1)[
            1].unsqueeze(1)
        return self.target_q_func(
            non_final_next_states).gather(1, next_action_batch).squeeze().detach()
