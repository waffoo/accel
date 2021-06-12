import os
from os import system
import time

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from accel.agents.sac import SAC
from accel.replay_buffers.replay_buffer import Transition

class OfflineSAC(SAC):
    def __init__(self, eval_env,
                 outdir,  # e.g. timestamp/results
                 # sac params
                 device,
                 observation_space,
                 action_space,
                 gamma, replay_buffer, tau=0.005,
                 actor_lr=3e-4,
                 critic_lr=3e-4,
                 batch_size=256,
                 update_interval=1,
                 target_update_interval=1,
                 load=None):
        super().__init__(device,
                 observation_space,
                 action_space,
                 gamma, replay_buffer, tau=tau,
                 actor_lr=actor_lr,
                 critic_lr=critic_lr,
                 batch_size=batch_size,
                 update_interval=update_interval,
                 target_update_interval=target_update_interval,
                 load=load)

        self.eval_env = eval_env
        self.data = None

        self.best_score = -1e10
        self.score_steps = []
        self.scores = []
        self.outdir = outdir
        self.log_file_name = f'{self.outdir}/scores.txt'

        os.makedirs(self.outdir, exist_ok=True)
        self.train_start_time = None


    def set_dataset(self, dataset, num=None):
        print('data registreation start')
        self.data = dataset

        ob = dataset['observations']
        ac = dataset['actions']
        ne = np.concatenate((ob[1:], ob[-1:]))
        re = dataset['rewards']
        va = ~(dataset['terminals'] | dataset['timeouts'])

        if num is not None:
            ob, ac, ne, re, va = ob[:num], ac[:num], ne[:num], re[:num], va[:num]

        st = time.time()

        all_li = list(zip(ob, ac, ne, re, va))
        result = list(map(lambda x: [Transition(*x)], all_li))
        gl = time.time()
        print(f'data registreation took {np.round(gl-st, 2)} sec.')
        print(f'{len(result) / 1e6} M transitions were registered.')

        self.replay_buffer.memory = result

    def fit(self, steps, eval_interval=5*10**3):
        next_eval_cnt = 1

        self.total_steps = 0
        self.train_cnt = 0
        self.train_start_time = time.time()

        while self.total_steps < steps:
            self.train()
            if self.total_steps >= next_eval_cnt * eval_interval:
                self.eval()
                next_eval_cnt += 1

    def train(self):
        # copy and paste
        if len(self.replay_buffer) < self.batch_size:
            return

        self.train_cnt += 1
        self.total_steps = self.train_cnt

        transitions = self.replay_buffer.sample(self.batch_size)
        map_func = lambda x: x[0]
        batch = Transition(*zip(*map(map_func, transitions)))

        state_batch = torch.tensor(
            np.array(batch.state, dtype=np.float32), device=self.device)
        action_batch = torch.tensor(
            np.array(batch.action, dtype=np.float32), device=self.device)
        next_state_batch = torch.tensor(
            np.array(batch.next_state, dtype=np.float32), device=self.device)
        reward_batch = torch.tensor(
            np.array(batch.reward, dtype=np.float32), device=self.device).unsqueeze(1)
        valid_batch = torch.tensor(
            np.array(batch.valid, dtype=np.float32), device=self.device).unsqueeze(1)

        target_q = self.calc_target_q(
            state_batch, action_batch, reward_batch, next_state_batch, valid_batch)
        q1 = self.critic1(state_batch, action_batch)
        q2 = self.critic2(state_batch, action_batch)
        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)
        q_loss = q1_loss + q2_loss
        self.q1_optim.zero_grad()
        self.q2_optim.zero_grad()
        q_loss.backward()
        self.q1_optim.step()
        self.q2_optim.step()

        pi, log_pi, _ = self.try_act(state_batch)

        qf1_pi = self.critic1(state_batch, pi)
        qf2_pi = self.critic2(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        # adjust alpha
        alpha_loss = -(self.log_alpha *
                       (self.target_entropy + log_pi).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()

        if self.train_cnt % self.target_update_interval == 0:
            self.soft_update(self.target_critic1, self.critic1)
            self.soft_update(self.target_critic2, self.critic2)
            self.prev_target_update_time = self.total_steps

    def eval(self, render=False):
        total_reward = 0

        obs = self.eval_env.reset()
        done = False
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

        while not done:
            t += 1
            action = self.act(obs, greedy=True)
            obs, reward, done, _ = self.eval_env.step(action)
            if render:
                output_png(self.eval_env, dirname, t)

            total_reward += reward

        self.score_steps.append(self.total_steps)
        self.scores.append(total_reward)

        if render:
            system(f'ffmpeg -hide_banner -loglevel panic -f image2 -r 30 -y -i '
                   f'{dirname}/%04d.png -an -vcodec libx264 -pix_fmt yuv420p {dirname}/out.mp4')
            system(f'rm {dirname}/*.png')

        if total_reward > self.best_score:
            dirname = f'{self.outdir}/{self.total_steps}'
            os.makedirs(dirname, exist_ok=True)

            model_name = f'{dirname}/q1.model'
            torch.save(self.critic1.state_dict(), model_name)
            model_name = f'{dirname}/q2.model'
            torch.save(self.critic2.state_dict(), model_name)
            model_name = f'{dirname}/pi.model'
            torch.save(self.actor.state_dict(), model_name)

            self.best_score = total_reward

        now = time.time()
        log = f'{self.total_steps} {total_reward}'
        if self.train_start_time is not None:
            elapsed = now - self.train_start_time
            log += f' {elapsed:.1f}'
        log += '\n'
        print(log, end='')

        with open(self.log_file_name, 'a') as f:
            f.write(log)


class SAC_CQL(OfflineSAC):
    def __init__(self, eval_env,
                 outdir,  # e.g. timestamp/results
                 cql_weight,
                 # sac params
                 device,
                 observation_space,
                 action_space,
                 gamma, replay_buffer, tau=0.005,
                 actor_lr=3e-5,
                 critic_lr=3e-4,
                 batch_size=256,
                 update_interval=1,
                 target_update_interval=1,
                 load=None
                 ):
        super().__init__(eval_env,
                         outdir,
                         device,
                         observation_space,
                         action_space,
                         gamma, replay_buffer, tau=tau,
                         actor_lr=actor_lr,
                         critic_lr=critic_lr,
                         batch_size=batch_size,
                         update_interval=update_interval,
                         target_update_interval=target_update_interval,
                         load=load)

        self.cql_weight = cql_weight  # cql alpha

    def train(self):
        # copy and paste
        if len(self.replay_buffer) < self.batch_size:
            return

        self.train_cnt += 1
        self.total_steps = self.train_cnt

        transitions = self.replay_buffer.sample(self.batch_size)
        map_func = lambda x: x[0]
        batch = Transition(*zip(*map(map_func, transitions)))

        state_batch = torch.tensor(
            np.array(batch.state, dtype=np.float32), device=self.device)
        action_batch = torch.tensor(
            np.array(batch.action, dtype=np.float32), device=self.device)
        next_state_batch = torch.tensor(
            np.array(batch.next_state, dtype=np.float32), device=self.device)
        reward_batch = torch.tensor(
            np.array(batch.reward, dtype=np.float32), device=self.device).unsqueeze(1)
        valid_batch = torch.tensor(
            np.array(batch.valid, dtype=np.float32), device=self.device).unsqueeze(1)

        target_q = self.calc_target_q(
            state_batch, action_batch, reward_batch, next_state_batch, valid_batch)
        q1_data = self.critic1(state_batch, action_batch)
        q2_data = self.critic2(state_batch, action_batch)
        q1_mse_loss = F.mse_loss(q1_data, target_q)
        q2_mse_loss = F.mse_loss(q2_data, target_q)

        num_random = 10
        random_actions = torch.FloatTensor(q2_data.shape[0] * num_random, action_batch.shape[-1]).uniform_(-1, 1).to(self.device)
        current_states = state_batch.unsqueeze(1).repeat(1, num_random, 1).view(-1, state_batch.shape[-1])
        current_actions, current_logpis, _ = self.try_act(current_states)
        next_states = next_state_batch.unsqueeze(1).repeat(1, num_random, 1).view(-1, state_batch.shape[-1])
        next_actions, next_logpis, _ = self.try_act(next_states)

        q1_rand = self.critic1(current_states, random_actions).view(-1, num_random)
        q2_rand = self.critic2(current_states, random_actions).view(-1, num_random)
        q1_curr = self.critic1(current_states, current_actions).view(-1, num_random)
        q2_curr = self.critic2(current_states, current_actions).view(-1, num_random)
        q1_next = self.critic1(current_states, next_actions).view(-1, num_random)
        q2_next = self.critic2(current_states, next_actions).view(-1, num_random)
        current_logpis = current_logpis.view(-1, num_random)
        next_logpis = next_logpis.view(-1, num_random)

        random_density = np.log(0.5 ** current_actions.shape[-1])
        cat_q1 = torch.cat([q1_rand - random_density,
                            q1_curr - current_logpis.detach(),
                            q1_next - next_logpis.detach()], 1)
        cat_q2 = torch.cat([q2_rand - random_density,
                            q2_curr - current_logpis.detach(),
                            q2_next - next_logpis.detach()], 1)

        cql_loss1 = (torch.logsumexp(cat_q1, dim=1, keepdim=True) - q1_data).mean()
        cql_loss2 = (torch.logsumexp(cat_q2, dim=1, keepdim=True) - q2_data).mean()

        q1_loss = q1_mse_loss + self.cql_weight * cql_loss1
        q2_loss = q2_mse_loss + self.cql_weight * cql_loss2
        q_loss = q1_loss + q2_loss

        self.q1_optim.zero_grad()
        self.q2_optim.zero_grad()
        q_loss.backward()
        self.q1_optim.step()
        self.q2_optim.step()

        pi, log_pi, _ = self.try_act(state_batch)

        qf1_pi = self.critic1(state_batch, pi)
        qf2_pi = self.critic2(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        # adjust alpha
        alpha_loss = -(self.log_alpha *
                       (self.target_entropy + log_pi).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()

        if self.train_cnt % self.target_update_interval == 0:
            self.soft_update(self.target_critic1, self.critic1)
            self.soft_update(self.target_critic2, self.critic2)
            self.prev_target_update_time = self.total_steps
