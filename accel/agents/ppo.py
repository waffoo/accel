import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim

from accel.replay_buffers.rollout_buffer import RolloutBuffer, Transition
from torch.distributions import Categorical


class ActorNet(nn.Module):
    def __init__(self, input, output, high_reso=False):
        super().__init__()
        self.conv1 = nn.Conv2d(input, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        linear_size = 7 * 7 * 64 if not high_reso else 12 * 12 * 64
        self.fc1 = nn.Linear(linear_size, 512)
        self.fc2 = nn.Linear(512, output)

    def forward(self, x):
        x = x / 255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)

        adv = F.relu(self.fc1(x))
        adv = self.fc2(adv)

        # return raw logits
        return adv


class CriticNet(nn.Module):
    def __init__(self, input, high_reso=False):
        super().__init__()
        self.conv1 = nn.Conv2d(input, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        linear_size = 7 * 7 * 64 if not high_reso else 12 * 12 * 64
        self.fc1 = nn.Linear(linear_size, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = x / 255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)

        adv = F.relu(self.fc1(x))
        adv = self.fc2(adv)
        return adv



class PPO:
    def __init__(self, envs, eval_envs, dim_state, dim_action, steps, lmd, gamma, device, batch_size, lr,
                 horizon=128, clip_eps=0.2, high_reso=False):
        self.envs = envs
        self.eval_envs = eval_envs
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.lmd = lmd
        self.gamma = gamma
        self.device = device
        self.batch_size = batch_size
        self.clip_eps = clip_eps

        self.actor = ActorNet(dim_state, dim_action, high_reso=high_reso)
        self.critic = CriticNet(dim_state, high_reso=high_reso)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.steps = steps
        self.horizon = horizon
        self.elapsed_step = 0

    def act(self, obs, greedy=False):
        pass

    def update(self, obs, action, next_obs, reward, valid):
        pass

    def train(self):
        pass

    def run(self):
        obs = self.envs.reset()

        buffer = RolloutBuffer(self.gamma, self.lmd)

        while self.elapsed_step < self.steps:
            # collect trajectories
            buffer.clear()
            self.critic.eval()
            for i in range(self.horizon):
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)

                with torch.no_grad():
                    action_logits = self.actor(obs_tensor)
                dist = Categorical(logits=action_logits)
                actions = dist.sample()
                log_prob = dist.log_prob(actions).cpu().numpy()
                actions = actions.cpu().numpy()
                next_obs, reward, done, info = self.envs.step(actions)

                transition = Transition(obs, actions, next_obs, reward, ~done, log_prob)

                with torch.no_grad():
                    values = self.critic(obs_tensor).flatten().detach().cpu().numpy()
                buffer.push(transition, values)

                obs = next_obs
                self.elapsed_step += sum(~done).item()

            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                values = self.critic(obs_tensor).flatten().detach().cpu().numpy()

            # compute advantage estimates A_t using GAE
            buffer.final_state_value(values)

            dataset = buffer.create_dataset()
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            self.critic.train()
            for (ob_, val_, ac_, log_prob_old_, gae_) in dataloader:
                # update the policy by maximizing PPO-Clip objective
                action_logits = self.actor(ob_)
                dist = Categorical(logits=action_logits)
                log_prob = dist.log_prob(ac_)
                ratio = torch.exp(log_prob - log_prob_old_)

                surr1 = ratio * gae_
                surr2 = torch.clip(ratio, 1-self.clip_eps, 1+self.clip_eps) * gae_

                # minus means "ascent"
                loss = -torch.min(surr1, surr2).mean()

                self.actor_optimizer.zero_grad()
                loss.backward()
                self.actor_optimizer.step()

                # value function learning
                ob_, val_ = ob_.to(self.device), val_.to(self.device)
                pred = self.critic(ob_).flatten()
                loss = F.mse_loss(pred, val_)

                self.critic_optimizer.zero_grad()
                loss.backward()
                self.critic_optimizer.step()



