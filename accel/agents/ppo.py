import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward', 'valid'))


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
                 horizon=128, high_reso=False):
        self.envs = envs
        self.eval_envs = eval_envs
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.lmd = lmd
        self.gamma = gamma
        self.device = device
        self.batch_size = batch_size

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
        from accel.replay_buffers.rollout_buffer import RolloutBuffer

        buffer = RolloutBuffer(self.gamma, self.lmd)

        while self.elapsed_step < self.steps:
            # collect trajectories
            buffer.clear()
            for i in range(self.horizon):
                actions = self.envs.action_space.sample()
                next_obs, reward, done, info = self.envs.step(actions)

                transition = Transition(obs, actions, next_obs, reward, ~done)

                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)

                with torch.no_grad():
                    values = self.critic(obs_tensor).flatten().detach().cpu().numpy()
                buffer.push(transition, values)

                obs = next_obs

            # note: only V(obs) will be used
            transition = Transition(obs, actions, obs, reward, np.zeros(8, dtype=bool))

            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                values = self.critic(obs_tensor).flatten().detach().cpu().numpy()
            buffer.push(transition, values)

            # compute advantage estimates A_t using GAE
            buffer.compute_gae()

            # update the policy by maximizing PPO-Clip objective

            # update value function
            dataset = buffer.value_func_dataset()
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            self.critic.train()
            for (ob_, val_) in dataloader:
                ob_, val_ = ob_.to(self.device), val_.to(self.device)
                pred = self.critic(ob_).flatten()
                loss = F.mse_loss(pred, val_)

                self.critic_optimizer.zero_grad()
                loss.backward()
                self.critic_optimizer.step()
            self.critic.eval()


            self.elapsed_step += 1

            #break


