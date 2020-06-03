import torch
from torch.distributions import Normal
import numpy as np
import copy
import torch.nn.functional as F
from accel.replay_buffers.replay_buffer import Transition


class SAC:
    def __init__(self, critic1, critic2,  actor,
                 q1_optim, q2_optim, actor_optim, device,
                 action_space,
                 gamma, replay_buffer, tau=0.005,
                 lr=3e-4,
                 batch_size=32,
                 update_interval=4,
                 target_update_interval=1):
        self.critic1 = critic1
        self.critic2 = critic2
        self.gamma = gamma
        self.actor = actor
        self.device = device
        self.q1_optim = q1_optim
        self.q2_optim = q2_optim
        self.actor_optim = actor_optim

        self.target_critic1 = copy.deepcopy(critic1).to(device)
        self.target_critic2 = copy.deepcopy(critic2).to(device)
        self.target_update_interval = target_update_interval

        self.replay_buffer = replay_buffer

        self.target_critic1.eval()
        self.target_critic2.eval()
        self.total_steps = 0
        self.n_actions = len(action_space.low)
        self.action_scale = torch.tensor((action_space.high - action_space.low) / 2)
        self.action_bias = torch.tensor((action_space.low + action_space.high) / 2)


        # that is -|A|
        self.target_entropy = -self.n_actions

        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr)

        self.tau = tau
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.prev_target_update_time = 0

    def act(self, obs, greedy=False):
        obs = torch.tensor(obs, device=self.device, dtype=torch.float32)

        with torch.no_grad():
            mean, log_std = torch.split(self.actor(obs[None]), self.n_actions, dim=1)
            torch.clamp(log_std, -20, 2)

        if greedy:
            action = torch.tanh(mean) * self.action_scale + self.action_bias
            return action[0]
        else:
            normal = Normal(mean, log_std.exp())
            x_t = normal.rsample()  # latent space
            y_t = torch.tanh(x_t)   # squash
            action = y_t * self.action_scale + self.action_bias
            return action[0]

    def try_act(self, obs):
        mean, log_std = torch.split(self.actor(obs), self.n_actions, dim=1)
        torch.clamp(log_std, -20, 2)

        normal = Normal(mean, log_std.exp())
        x_t = normal.rsample()  # latent space
        y_t = torch.tanh(x_t)   # squash
        action = y_t * self.action_scale + self.action_bias

        # enforcing bound
        eps = 1e-6
        log_pi = (normal.log_prob(x_t) \
                 - torch.log(self.action_scale * (1-y_t.pow(2)) + eps)).sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_pi, mean

    def calc_target_q(self, obs, action, reward, next_states, done):
        with torch.no_grad():
            action, log_pi, _ = self.try_act(obs)
            q1 = self.target_critic1(torch.cat([obs, action], dim=1))
            q2 = self.target_critic2(torch.cat([obs, action], dim=1))
            q = torch.min(q1, q2) - self.alpha * log_pi

            return reward + (1 - done) * self.gamma * q

    def update(self, obs, action, next_obs, reward, done):
        self.replay_buffer.push(obs, action, next_obs, np.float32(reward), done)

        self.total_steps += 1

        if self.total_steps % self.update_interval == 0:
            self.train()

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.tensor(
            np.stack(batch.state), device=self.device, dtype=torch.float32)
        action_batch = torch.tensor(
            np.stack(batch.action), device=self.device, dtype=torch.float32)
        next_state_batch = torch.tensor(
            np.stack(batch.next_state), device=self.device, dtype=torch.float32)
        reward_batch = torch.tensor(
            batch.reward, device=self.device, dtype=torch.float32).unsqueeze(1)
        done_batch = torch.tensor(
            batch.done, device=self.device, dtype=torch.float32).unsqueeze(1)

        q1 = self.critic1(torch.cat([state_batch, action_batch], dim=1))
        q2 = self.critic2(torch.cat([state_batch, action_batch], dim=1))
        target_q = self.calc_target_q(
            state_batch, action_batch, reward_batch, next_state_batch, done_batch)
        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)

        action, log_pi, _ = self.try_act(state_batch)
        with torch.no_grad():
            q1_pi = self.critic1(torch.cat([state_batch, action], dim=1))
            q2_pi = self.critic2(torch.cat([state_batch, action], dim=1))
            q_pi = torch.min(q1_pi, q2_pi)

        pi_loss = torch.mean(self.alpha.detach() * log_pi - q_pi)

        self.q1_optim.zero_grad()
        self.q2_optim.zero_grad()
        q1_loss.backward()
        q2_loss.backward()
        self.q1_optim.step()
        self.q2_optim.step()

        self.actor_optim.zero_grad()
        pi_loss.backward()
        self.actor_optim.step()

        # adjust alpha
        alpha_loss = -(self.log_alpha * (self.target_entropy + log_pi).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()


        if self.total_steps - self.prev_target_update_time > self.target_update_interval:
            self.soft_update(self.target_critic1, self.critic1)
            self.soft_update(self.target_critic2, self.critic2)
            self.prev_target_update_time = self.total_steps

    def soft_update(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(t.data * (1.0 - self.tau) + s.data * self.tau)

