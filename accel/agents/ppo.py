import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
import time

from accel.replay_buffers.rollout_buffer import RolloutBuffer, Transition
from torch.distributions import Categorical
from accel.explorers.epsilon_greedy import LinearDecayEpsilonGreedy



class PPO:
    def __init__(self, envs, eval_env, steps, actor, critic,
                 device, lmd=0.95, gamma=0.99, batch_size=128,
                 lr=2.5e-4, horizon=128, clip_eps=0.2, epoch_per_update=3, entropy_coef=0.01,
                 load="", eval_interval=50000, run_per_eval=3):
        self.envs = envs
        self.eval_env = eval_env
        self.lmd = lmd
        self.gamma = gamma
        self.device = device
        self.batch_size = batch_size
        self.clip_eps = clip_eps
        self.epoch_per_update = epoch_per_update
        self.entropy_coef = entropy_coef
        self.eval_interval = eval_interval
        self.run_per_eval = run_per_eval
        self.lr = lr

        self.steps = steps
        self.horizon = horizon
        self.elapsed_step = 0
        self.best_score = -1e10

        self.actor = actor
        self.critic = critic
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # TODO detach scheduler from epsilon greedy
        self.lr_scheduler = LinearDecayEpsilonGreedy(start_eps=1, end_eps=0.01, decay_steps=self.steps)

        if load:
            self.actor.load_state_dict(torch.load(f'{load}/actor.model', map_location=device))
            self.critic.load_state_dict(torch.load(f'{load}/critic.model', map_location=device))


    def act(self, obs, greedy=False):
        pass

    def update(self, obs, action, next_obs, reward, valid):
        pass

    def train(self):
        pass

    def run(self):
        self.train_start_time = time.time()
        self.log_file_name = 'scores.txt'

        obs = self.envs.reset()

        buffer = RolloutBuffer(self.gamma, self.lmd)
        self.next_eval_cnt = 1

        while self.elapsed_step < self.steps:
            # collect trajectories
            buffer.clear()
            self.critic.eval()
            self.actor.eval()
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

            self.elapsed_step += self.horizon * self.envs.num_envs

            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                values = self.critic(obs_tensor).flatten().detach().cpu().numpy()

            # compute advantage estimates A_t using GAE
            buffer.final_state_value(values)

            next_lr = self.lr_scheduler.calc_eps(self.elapsed_step) * self.lr
            assert len(self.actor_optimizer.param_groups) == 1
            self.actor_optimizer.param_groups[0]['lr'] = next_lr
            self.critic_optimizer.param_groups[0]['lr'] = next_lr

            dataset = buffer.create_dataset()
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            self.critic.train()
            self.actor.train()
            for _ in range(self.epoch_per_update):
                for (ob_, val_, ac_, log_prob_old_, gae_) in dataloader:
                    # update the policy by maximizing PPO-Clip objective
                    action_logits = self.actor(ob_)
                    dist = Categorical(logits=action_logits)
                    log_prob = dist.log_prob(ac_)
                    entropy_bonus = dist.entropy()
                    ratio = torch.exp(log_prob - log_prob_old_)

                    surr1 = ratio * gae_
                    surr2 = torch.clip(ratio, 1-self.clip_eps, 1+self.clip_eps) * gae_

                    # minus means "ascent"
                    loss = -torch.min(surr1, surr2).mean() - (entropy_bonus * self.entropy_coef).mean()

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

            self.critic.eval()
            self.actor.eval()

            if self.elapsed_step >= self.next_eval_cnt * self.eval_interval:
                self.evaluate()

        print('Complete')

    def evaluate(self):
        self.next_eval_cnt += 1

        total_reward = 0.

        for _ in range(self.run_per_eval):
            while True:
                done = False
                obs = self.eval_env.reset()

                while not done:
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        action_logits = self.actor(obs_tensor)
                    dist = Categorical(logits=action_logits)
                    actions = dist.sample().cpu().numpy()

                    obs, reward, done, _ = self.eval_env.step(actions[0])

                    total_reward += reward

                if done:
                    break

        total_reward /= self.run_per_eval

        if total_reward > self.best_score:
            self.best_score = total_reward
            dirname = f'{self.elapsed_step}'
            if not os.path.exists(dirname):
                os.mkdir(dirname)

            torch.save(self.actor.state_dict(), f'{dirname}/actor.model')
            torch.save(self.critic.state_dict(), f'{dirname}/critic.model')

        now = time.time()
        elapsed = now - self.train_start_time
        log = f'{self.elapsed_step} {total_reward} {elapsed:.1f}\n'
        print(log, end='')

        with open(self.log_file_name, 'a') as f:
            f.write(log)
