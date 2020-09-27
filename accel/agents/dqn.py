import torch
import numpy as np
import copy
import torch.nn.functional as F

from accel.replay_buffers.replay_buffer import Transition
from accel.replay_buffers.prioritized_replay_buffer import PrioritizedReplayBuffer


class DQN:
    def __init__(self, q_func, optimizer, replay_buffer, gamma, explorer,
                 device,
                 batch_size=32,
                 update_interval=4,
                 target_update_interval=200,
                 replay_start_step=10000,
                 huber=False):
        self.q_func = q_func.to(device)
        self.target_q_func = copy.deepcopy(self.q_func).to(device)
        self.optimizer = optimizer
        self.replay_buffer = replay_buffer
        self.prioritized = isinstance(replay_buffer, PrioritizedReplayBuffer)
        self.gamma = gamma
        self.explorer = explorer
        self.device = device
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.target_update_interval = target_update_interval
        self.huber = huber
        self.total_steps = 0
        self.replay_start_step = replay_start_step

        self.prev_target_update_time = 0

        self.target_q_func.eval()

    def act(self, obs, greedy=False):
        obs = torch.tensor(obs, device=self.device, dtype=torch.float32)

        with torch.no_grad():
            action_value = self.q_func(obs[None])

        action = self.explorer.act(
            self.total_steps, action_value, greedy=greedy)
        return action.item()

    def update(self, obs, action, next_obs, reward, valid):
        self.replay_buffer.push(obs, action, next_obs,
                                np.float32(reward), valid)

        self.total_steps += 1
        if self.total_steps % self.update_interval == 0:
            self.train()

    def next_state_value(self, next_states):
        return self.target_q_func(next_states).max(1)[0].detach()

    def train(self):
        if len(self.replay_buffer) < self.batch_size or len(self.replay_buffer) < self.replay_start_step:
            return

        if self.prioritized:
            transitions, idx_batch, weights = self.replay_buffer.sample(self.batch_size)
        else:
            transitions = self.replay_buffer.sample(self.batch_size)

        def f(trans):
            start_state = trans[0].state
            action = trans[0].action
            next_state = trans[-1].next_state
            valid = trans[-1].valid
            reward = 0.
            for i, data in enumerate(trans):
                reward += data.reward * self.gamma ** i

            return Transition(start_state, action, next_state, reward, valid)

        def extract_steps(trans):
            return len(trans)


        steps_batch = list(map(extract_steps, transitions))
        transitions = map(f, transitions)

        batch = Transition(*zip(*transitions))

        state_batch = torch.tensor(
            np.array(batch.state, dtype=np.float32), device=self.device)
        action_batch = torch.tensor(
            batch.action, device=self.device, dtype=torch.int64).unsqueeze(1)
        next_state_batch = torch.tensor(
            np.array(batch.next_state, dtype=np.float32), device=self.device)
        reward_batch = torch.tensor(
            np.array(batch.reward, dtype=np.float32), device=self.device)
        valid_batch = torch.tensor(
            np.array(batch.valid, dtype=np.float32), device=self.device)
        steps_batch = torch.tensor(np.array(steps_batch, dtype=np.float32),device=self.device)


        state_action_values = self.q_func(state_batch).gather(1, action_batch)

        expected_state_action_values = reward_batch + \
            valid_batch * (self.gamma ** steps_batch) * \
            self.next_state_value(next_state_batch)

        if self.prioritized:
            td_error = abs(expected_state_action_values - state_action_values.squeeze(1)).tolist()
            for data_idx, err in zip(idx_batch, td_error):
                self.replay_buffer.update(data_idx, err)

        if self.huber:
            if self.prioritized:
                loss_each = F.smooth_l1_loss(state_action_values,
                                        expected_state_action_values.unsqueeze(1), reduction='none')
                loss = torch.sum(loss_each * torch.tensor(weights, device=self.device))
            else:
                loss = F.smooth_l1_loss(state_action_values,
                                        expected_state_action_values.unsqueeze(1))
        else:
            if self.prioritized:
                loss_each = F.mse_loss(state_action_values,
                                  expected_state_action_values.unsqueeze(1), reduction='none')
                loss = torch.sum(loss_each * torch.tensor(weights, device=self.device))

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
    def next_state_value(self, next_states):
        next_action_batch = self.q_func(next_states).max(1)[
            1].unsqueeze(1)
        return self.target_q_func(
            next_states).gather(1, next_action_batch).squeeze().detach()
