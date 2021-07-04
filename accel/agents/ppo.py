import os
import time
from collections import namedtuple

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader

from accel.explorers.epsilon_greedy import LinearDecayEpsilonGreedy
from accel.replay_buffers.rollout_buffer import RolloutBuffer, Transition


class PPO:
    def __init__(self, envs, eval_env, steps, actor, critic,
                 device, lmd=0.95, gamma=0.99, batch_size=128,
                 lr=2.5e-4, horizon=128, clip_eps=0.2, epoch_per_update=4, entropy_coef=0.01,
                 load="", eval_interval=50000, epoch_per_eval=3, mlflow=False, value_loss_coef=0.5,
                 value_clipping=True, atari=False):
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
        self.epoch_per_eval = epoch_per_eval
        self.lr = lr
        self.mlflow = mlflow
        self.value_loss_coef = value_loss_coef
        self.value_clipping = value_clipping
        self.atari = atari

        self.steps = steps
        self.horizon = horizon
        self.elapsed_step = 0
        self.best_score = -1e10
        self.max_grad_norm = 0.5

        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr, eps=1e-5)

        # TODO detach scheduler from epsilon greedy
        self.lr_scheduler = LinearDecayEpsilonGreedy(
            start_eps=1, end_eps=0.01, decay_steps=self.steps)

        if load:
            self.actor.load_state_dict(torch.load(
                f'{load}/actor.model', map_location=device))
            self.critic.load_state_dict(torch.load(
                f'{load}/critic.model', map_location=device))

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
                obs_tensor = torch.tensor(
                    obs, dtype=torch.float32).to(self.device)

                with torch.no_grad():
                    action_logits = self.actor(obs_tensor)
                dist = Categorical(logits=action_logits)
                actions = dist.sample()
                log_prob = dist.log_prob(actions).cpu().numpy()
                actions = actions.cpu().numpy()
                next_obs, reward, done, info = self.envs.step(actions)

                transition = Transition(
                    obs, actions, next_obs, reward, ~done, log_prob)

                with torch.no_grad():
                    values = self.critic(
                        obs_tensor).flatten().detach().cpu().numpy()
                buffer.push(transition, values)

                obs = next_obs

            self.elapsed_step += self.horizon * self.envs.num_envs

            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                values = self.critic(
                    obs_tensor).flatten().detach().cpu().numpy()

            # compute advantage estimates A_t using GAE
            buffer.final_state_value(values)

            next_lr = self.lr_scheduler.calc_eps(self.elapsed_step) * self.lr
            for pg in self.optimizer.param_groups:
                pg['lr'] = next_lr

            dataset = buffer.create_dataset()
            dataloader = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True)

            value_loss_epoch = 0.
            actor_loss_epoch = 0.
            entropy_bonus_epoch = 0.
            added_cnt = 0

            self.critic.train()
            self.actor.train()
            for _ in range(self.epoch_per_update):
                for (ob_, ret_, ac_, log_prob_old_,
                     gae_, value_) in dataloader:
                    added_cnt += 1
                    ob_, ac_ = ob_.to(self.device), ac_.to(self.device)
                    log_prob_old_ = log_prob_old_.to(self.device)
                    gae_ = gae_.to(self.device)

                    # update the policy by maximizing PPO-Clip objective
                    action_logits = self.actor(ob_)
                    dist = Categorical(logits=action_logits)
                    log_prob = dist.log_prob(ac_)
                    entropy_bonus = dist.entropy().mean()
                    ratio = torch.exp(log_prob - log_prob_old_)

                    surr1 = ratio * gae_
                    surr2 = torch.clip(ratio, 1 - self.clip_eps,
                                       1 + self.clip_eps) * gae_

                    # minus means "ascent"
                    actor_loss = -torch.min(surr1, surr2).mean()

                    # value function learning
                    ret_, value_ = ret_.to(self.device), value_.to(self.device)
                    pred = self.critic(ob_).flatten()

                    if self.value_clipping:
                        clipped_pred = value_ + \
                            (pred - value_).clamp(-self.clip_eps, self.clip_eps)

                        value_loss = (pred - ret_).pow(2)
                        clipped_value_loss = (clipped_pred - ret_).pow(2)
                        value_loss = torch.max(
                            value_loss, clipped_value_loss).mean() / 2

                    else:
                        value_loss = F.mse_loss(pred, ret_) / 2

                    loss = value_loss * self.value_loss_coef + \
                        actor_loss - entropy_bonus * self.entropy_coef

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(self.actor.parameters()) + list(self.critic.parameters()), self.max_grad_norm)
                    self.optimizer.step()

                    value_loss_epoch += value_loss.item()
                    actor_loss_epoch += actor_loss.item()
                    entropy_bonus_epoch += entropy_bonus.item()

            value_loss_epoch /= added_cnt
            actor_loss_epoch /= added_cnt
            entropy_bonus_epoch /= added_cnt

            self.critic.eval()
            self.actor.eval()

            if self.elapsed_step >= self.next_eval_cnt * self.eval_interval:
                mlflow.log_metric(
                    'value_loss', value_loss_epoch, step=self.elapsed_step)
                mlflow.log_metric(
                    'actor_loss', actor_loss_epoch, step=self.elapsed_step)
                mlflow.log_metric(
                    'entropy_bonus', entropy_bonus_epoch, step=self.elapsed_step)

                self.evaluate()

        if self.mlflow:
            now = time.time()
            elapsed = now - self.train_start_time
            mlflow.log_metric('duration', np.round(elapsed / 60 / 60, 2))
        print('Complete')

    def evaluate(self):
        self.next_eval_cnt += 1

        total_reward = 0.

        for _ in range(self.epoch_per_eval):
            while True:
                done = False
                obs = self.eval_env.reset()

                while not done:
                    obs_tensor = torch.tensor(
                        obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        action_logits = self.actor(obs_tensor)
                    dist = Categorical(logits=action_logits)
                    actions = dist.sample().cpu().numpy()

                    obs, reward, done, _ = self.eval_env.step(actions[0])

                    total_reward += reward

                if self.atari:
                    if self.eval_env.was_real_done:
                        break
                else:
                    if done:
                        break

        total_reward /= self.epoch_per_eval
        if self.mlflow:
            mlflow.log_metric('reward', total_reward, step=self.elapsed_step)

        if total_reward > self.best_score:
            self.best_score = total_reward
            dirname = f'{self.elapsed_step}'
            if not os.path.exists(dirname):
                os.mkdir(dirname)

            torch.save(self.actor.state_dict(), f'{dirname}/actor.model')
            torch.save(self.critic.state_dict(), f'{dirname}/critic.model')

        now = time.time()
        elapsed = now - self.train_start_time
        log = f'{self.elapsed_step} {total_reward:.1f} {elapsed:.1f}\n'
        print(log, end='')

        with open(self.log_file_name, 'a') as f:
            f.write(log)
