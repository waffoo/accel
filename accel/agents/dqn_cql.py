import copy
import os
import time
from os import system

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from accel.agents.dqn import DQN
from accel.replay_buffers.prioritized_replay_buffer import \
    PrioritizedReplayBuffer
from accel.replay_buffers.replay_buffer import Transition


class OfflineDQN(DQN):
    def __init__(self, eval_env,
                 outdir,  # e.g. timestamp/results
                 # dqn parameters
                 q_func, optimizer, replay_buffer, gamma, explorer,
                 device,
                 batch_size=32,
                 update_interval=4,
                 target_update_interval=200,
                 replay_start_step=10000,
                 huber=False):
        super().__init__(q_func, optimizer, replay_buffer, gamma, explorer,
                         device,
                         batch_size=batch_size,
                         update_interval=update_interval,
                         target_update_interval=target_update_interval,
                         replay_start_step=0,
                         huber=huber)

        self.best_score = -1e10
        self.score_steps = []
        self.scores = []
        self.eval_env = eval_env
        self.outdir = outdir
        self.log_file_name = f'{self.outdir}/scores.txt'

    def set_dataset(self, dataset, num=None):
        print('data registreation start')
        st = time.time()

        ob = dataset['state']
        ac = dataset['action']
        ne = np.concatenate((ob[1:], ob[-1:]))
        re = dataset['reward']
        va = 1 - dataset['terminal']

        if num is not None:
            ob, ac, ne, re, va = ob[:num], ac[:num], ne[:num], re[:num], va[:num]

        all_li = list(zip(ob, ac, ne, re, va))
        result = list(map(lambda x: [Transition(*x)], all_li))
        gl = time.time()
        print(f'data registreation took {np.round(gl-st, 2)} sec.')
        print(f'{len(result) / 1e6} M transitions were registered.')

        self.replay_buffer.memory = result

    def fit(self, steps, eval_interval=5 * 10**3):
        next_eval_cnt = 1

        self.total_steps = 0
        self.train_cnt = 0
        self.train_start_time = time.time()

        while self.total_steps < steps:
            self.total_steps += 1
            self.train_cnt += 1
            self.train()
            if self.total_steps >= next_eval_cnt * eval_interval:
                self.eval()
                next_eval_cnt += 1

    def eval(self, render=False):
        total_reward = 0
        t = 0

        def output_png(env, dirname, t):
            img = env.render(mode='rgb_array')
            img = Image.fromarray(img)
            img.save(f'{dirname}/{t:04d}.png')

        if render:
            dirname = f'{self.outdir}/{self.total_steps}'
            os.makedirs(dirname, exist_ok=True)
            system(f'rm {dirname}/*.png')
            output_png(self.eval_env, dirname, t)

        while True:
            obs = self.eval_env.reset()
            done = False

            while not done:
                t += 1
                action = self.act(obs, greedy=True)
                obs, reward, done, _ = self.eval_env.step(action)
                if render:
                    output_png(self.eval_env, dirname, t)

                total_reward += reward

            if self.eval_env.was_real_done:
                break

        self.score_steps.append(self.total_steps)
        self.scores.append(total_reward)

        if render:
            system(f'ffmpeg -hide_banner -loglevel panic -f image2 -r 30 -y -i '
                   f'{dirname}/%04d.png -an -vcodec libx264 -pix_fmt yuv420p {dirname}/out.mp4')
            system(f'rm {dirname}/*.png')

        if total_reward > self.best_score:
            dirname = f'{self.outdir}/{self.total_steps}'
            os.makedirs(dirname, exist_ok=True)

            model_name = f'{dirname}/{self.total_steps}.model'
            torch.save(self.q_func.state_dict(), model_name)
            self.best_score = total_reward

        mlflow.log_metric('reward', total_reward,
                          step=self.total_steps)

        now = time.time()
        log = f'{self.total_steps} {total_reward}'
        if self.train_start_time is not None:
            elapsed = now - self.train_start_time
            log += f' {elapsed:.1f}'
        log += '\n'
        print(log, end='')

        with open(self.log_file_name, 'a') as f:
            f.write(log)


class OfflineDoubleDQN(OfflineDQN):
    def next_state_value(self, next_states):
        next_action_batch = self.q_func(next_states).max(1)[
            1].unsqueeze(1)
        return self.target_q_func(
            next_states).gather(1, next_action_batch).squeeze().detach()


class DQN_CQL(OfflineDoubleDQN):
    def __init__(self, eval_env,
                 outdir,  # e.g. timestamp/results
                 cql_weight,
                 # dqn parameters
                 q_func, optimizer, replay_buffer, gamma, explorer,
                 device,
                 batch_size=32,
                 update_interval=4,
                 target_update_interval=200,
                 replay_start_step=10000,
                 huber=False
                 ):
        super().__init__(eval_env,
                         outdir,  # e.g. timestamp/results
                         # dqn parameters
                         q_func, optimizer, replay_buffer, gamma, explorer,
                         device,
                         batch_size=batch_size,
                         update_interval=update_interval,
                         target_update_interval=target_update_interval,
                         replay_start_step=replay_start_step,
                         huber=huber)

        self.cql_weight = cql_weight

    def train(self):
        # if len(self.replay_buffer) < self.batch_size or len(self.replay_buffer) < self.replay_start_step:
        #     return

        if self.prioritized:
            transitions, idx_batch, weights = self.replay_buffer.sample(
                self.batch_size)
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
        steps_batch = torch.tensor(
            np.array(steps_batch, dtype=np.float32), device=self.device)

        qout = self.q_func(state_batch)
        state_action_values = qout.gather(1, action_batch)

        expected_state_action_values = reward_batch + \
            valid_batch * (self.gamma ** steps_batch) * \
            self.next_state_value(next_state_batch)

        if self.prioritized:
            td_error = abs(expected_state_action_values -
                           state_action_values.squeeze(1)).tolist()
            for data_idx, err in zip(idx_batch, td_error):
                self.replay_buffer.update(data_idx, err)

        if self.huber:
            if self.prioritized:
                loss_each = F.smooth_l1_loss(state_action_values,
                                             expected_state_action_values.unsqueeze(1), reduction='none')
                dqn_loss = torch.sum(
                    loss_each * torch.tensor(weights, device=self.device))
            else:
                dqn_loss = F.smooth_l1_loss(state_action_values,
                                            expected_state_action_values.unsqueeze(1))
        else:
            if self.prioritized:
                loss_each = F.mse_loss(state_action_values,
                                       expected_state_action_values.unsqueeze(1), reduction='none')
                dqn_loss = torch.sum(
                    loss_each * torch.tensor(weights, device=self.device))

            else:
                dqn_loss = F.mse_loss(state_action_values,
                                      expected_state_action_values.unsqueeze(1))

        # add CQL loss
        policy_q = torch.logsumexp(qout, dim=-1, keepdim=True)
        data_q = state_action_values  # data_q, [32, 1]
        cql_loss = (policy_q - data_q).mean()
        loss = dqn_loss + self.cql_weight * cql_loss

        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.q_func.parameters():
        #    param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        if self.total_steps - self.prev_target_update_time >= self.target_update_interval:
            self.target_q_func.load_state_dict(self.q_func.state_dict())
            self.prev_target_update_time = self.total_steps
