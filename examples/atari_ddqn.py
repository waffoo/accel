import time
import gym
import copy
# from gym.utils.play import play import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
import argparse
import datetime
import os

from accel.utils.atari_wrappers import make_atari, make_atari_ram
from accel.explorers import epsilon_greedy
from accel.replay_buffers.replay_buffer import Transition, ReplayBuffer
from accel.agents import dqn

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


class Net(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.conv1 = nn.Conv2d(input, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, output)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class RamNet(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.fc1 = nn.Linear(input, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.relu(self.fc4(x))


parser = argparse.ArgumentParser()
parser.add_argument('--env', default='BreakoutNoFrameskip-v4',
                    help='name of environment')
parser.add_argument('--load', default=None,
                    help='model path')
parser.add_argument('--demo', action='store_true',
                    help='demo flag')
args = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
is_ram = '-ram-' in args.env

if is_ram:
    env = make_atari_ram(args.env)
    eval_env = make_atari_ram(args.env, clip_rewards=False)
else:
    env = make_atari(args.env)
    eval_env = make_atari(args.env, clip_rewards=False)

env.seed(seed)
eval_env.seed(seed)

dim_state = env.observation_space.shape[0]
dim_action = env.action_space.n


num_steps = 10**6
eval_interval = 10**4
GAMMA = 0.99

if is_ram:
    q_func = RamNet(dim_state, dim_action)
else:
    q_func = Net(dim_state, dim_action)

if args.load is not None:
    q_func.load_state_dict(torch.load(args.load, map_location=device))


optimizer = optim.RMSprop(q_func.parameters(), lr=0.00025)
memory = ReplayBuffer(capacity=10**6)

score_steps = []
scores = []

explorer = epsilon_greedy.LinearDecayEpsilonGreedy(
    start_eps=1.0, end_eps=0.1, decay_steps=1e6)

agent = dqn.DoubleDQN(q_func, optimizer, memory, GAMMA,
                      explorer, device, batch_size=32, target_update_interval=10000)

if args.demo:
    for x in range(10):
        total_reward = 0

        while True:
            obs = eval_env.reset()
            eval_env.render()
            done = False

            while not done:
                action = agent.act(obs, greedy=True)
                obs, reward, done, _ = eval_env.step(action)
                eval_env.render()

                print(reward)
                total_reward += reward

            if eval_env.was_real_done:
                break

        print('Episode:', x, 'Score:', total_reward)

    exit(0)


next_eval_cnt = 1
episode_cnt = 0

train_start_time = time.time()

if not os.path.exists('results'):
    os.mkdir('results')

timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
result_dir = f'results/{timestamp}'

if not os.path.exists(result_dir):
    os.mkdir(result_dir)

log_file_name = f'{result_dir}/scores.txt'
best_score = -1e10

while agent.total_steps < num_steps:
    episode_cnt += 1

    obs = env.reset()
    done = False
    total_reward = 0
    step = 0

    while not done:
        action = agent.act(obs)
        next_obs, reward, done, _ = env.step(action)
        total_reward += reward
        step += 1

        next_valid = 1 if step == env.spec.max_episode_steps else float(not done)
        agent.update(obs, action, next_obs, reward, next_valid)


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
            model_name = f'{result_dir}/{agent.total_steps}.model'
            torch.save(q_func.state_dict(), model_name)
            best_score = total_reward

        now = time.time()
        elapsed = now - train_start_time

        log = f'{agent.total_steps} {total_reward} {elapsed:.1f}\n'
        print(log, end='')

        with open(log_file_name, 'a') as f:
            f.write(log)

# final evaluation
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

score_steps.append(agent.total_steps)
scores.append(total_reward)

model_name = f'{result_dir}/final.model'
torch.save(q_func.state_dict(), model_name)

now = time.time()
elapsed = now - train_start_time

log = f'{agent.total_steps} {total_reward} {elapsed:.1f}\n'
print(log, end='')

with open(log_file_name, 'a') as f:
    f.write(log)

print('Complete')
env.close()
