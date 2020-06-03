import gym
import pybulletgym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from accel.agents.sac import SAC
from accel.replay_buffers.replay_buffer import ReplayBuffer
import datetime
import time
import os

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


env = gym.make('HumanoidPyBulletEnv-v0')
eval_env = gym.make('HumanoidPyBulletEnv-v0')
#env = gym.wrappers.Monitor(env, 'movie', force=True)
# env.render(mode='human')


class Net(torch.nn.Module):
    def __init__(self, n_input, n_output, n_hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

n_states = env.observation_space.shape[0]
n_actions = len(env.action_space.low)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

critic1 = Net(n_states+n_actions, 1)
critic2 = Net(n_states+n_actions, 1)
actor = Net(n_states, n_actions * 2)
q1_optim = torch.optim.Adam(critic1.parameters(), lr=3e-4)
q2_optim = torch.optim.Adam(critic2.parameters(), lr=3e-4)
actor_optim = torch.optim.Adam(actor.parameters(), lr=3e-4)

memory = ReplayBuffer(capacity=10**6)

agent = SAC(critic1, critic2, actor, q1_optim, q2_optim, actor_optim,
             device=device, action_space=env.action_space, gamma=0.99,
            replay_buffer=memory)

torch.autograd.set_detect_anomaly(True)

num_steps = 10**6
eval_interval = 10**4

next_eval_cnt = 1
episode_cnt = 0

score_steps = []
scores = []

if not os.path.exists('results'):
    os.mkdir('results')

timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
result_dir = f'results/{timestamp}'

if not os.path.exists(result_dir):
    os.mkdir(result_dir)

log_file_name = f'{result_dir}/scores.txt'
best_score = -1e10

train_start_time = time.time()

while agent.total_steps < num_steps:
    episode_cnt += 1

    obs = env.reset()
    done = False
    total_reward = 0
    step = 0

    while not done:
        action = agent.act(obs)
        next_obs, reward, done, _ = env.step(action)
        agent.update(obs, action, next_obs, reward, done)

        total_reward += reward
        step += 1

        obs = next_obs

    if agent.total_steps > next_eval_cnt * eval_interval:
        total_reward = 0

        while True:
            obs = eval_env.reset()
            done = False

            while not done:
                action = agent.act(obs, greedy=True)
                obs, reward, done, _ = eval_env.step(action)

                total_reward += reward

            if eval_env.was_real_done:
                break

        next_eval_cnt += 1

        score_steps.append(agent.total_steps)
        scores.append(total_reward)

        if total_reward > best_score:
            model_name = f'{result_dir}/{agent.total_steps}-q1.model'
            torch.save(critic1.state_dict(), model_name)
            model_name = f'{result_dir}/{agent.total_steps}-q2.model'
            torch.save(critic2.state_dict(), model_name)
            model_name = f'{result_dir}/{agent.total_steps}-pi.model'
            torch.save(actor.state_dict(), model_name)

            best_score = total_reward

        now = time.time()
        elapsed = now - train_start_time

        log = f'{agent.total_steps} {total_reward} {elapsed:.1f}\n'
        print(log, end='')

        with open(log_file_name, 'a') as f:
            f.write(log)


print('Complete')
env.close()
