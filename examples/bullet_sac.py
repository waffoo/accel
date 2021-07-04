import argparse
import datetime
import os
import random
import time

import gym
import numpy as np
import pybullet_envs
import torch
import torch.nn as nn
import torch.nn.functional as F

from accel.agents.sac import SAC
from accel.replay_buffers.replay_buffer import ReplayBuffer
from accel.utils.wrappers import RewardScaler

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--env', default='HumanoidBulletEnv-v0',
                    help='name of environment')
parser.add_argument('--load', default=None,
                    help='model path')
parser.add_argument('--demo', action='store_true',
                    help='demo flag')
parser.add_argument('--scale', type=float, default=1.,
                    help='reward scale')
args = parser.parse_args()


env = RewardScaler(gym.make(args.env), scale=args.scale)
eval_env = gym.make(args.env)
# env = gym.wrappers.Monitor(env, 'movie', force=True)
# env.render(mode='human')

n_states = env.observation_space.shape[0]
n_actions = len(env.action_space.low)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


memory = ReplayBuffer(capacity=10**6)  # assert nstep=1

agent = SAC(device=device, observation_space=env.observation_space, action_space=env.action_space, gamma=0.99,
            replay_buffer=memory, update_interval=1, load=args.load)

if args.demo:
    eval_env.render(mode='human')
    for x in range(10):
        total_reward = 0

        obs = eval_env.reset()
        done = False

        while not done:
            action = agent.act(obs, greedy=True)
            obs, reward, done, _ = eval_env.step(action)

            total_reward += reward

        print(f'Episode: {x+1}  Score:{total_reward}')

    exit(0)


num_steps = 5 * 10**6
eval_interval = 5 * 10**3
initial_random_steps = 10**3

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
        if agent.total_steps > initial_random_steps:
            action = agent.act(obs)
        else:
            action = env.action_space.sample()

        next_obs, reward, done, _ = env.step(action)
        total_reward += reward
        step += 1

        next_valid = 1 if step == env.spec.max_episode_steps else float(
            not done)
        agent.update(obs, action, next_obs, reward, next_valid)

        obs = next_obs

    if agent.total_steps >= next_eval_cnt * eval_interval:
        total_reward = 0

        obs = eval_env.reset()
        done = False

        while not done:
            action = agent.act(obs, greedy=True)
            obs, reward, done, _ = eval_env.step(action)

            total_reward += reward

        next_eval_cnt += 1

        score_steps.append(agent.total_steps)
        scores.append(total_reward)

        if total_reward > best_score:
            dirname = f'{result_dir}/{agent.total_steps}'
            if not os.path.exists(dirname):
                os.mkdir(dirname)
            model_name = f'{dirname}/q1.model'
            torch.save(agent.critic1.state_dict(), model_name)
            model_name = f'{dirname}/q2.model'
            torch.save(agent.critic2.state_dict(), model_name)
            model_name = f'{dirname}/pi.model'
            torch.save(agent.actor.state_dict(), model_name)

            best_score = total_reward

        now = time.time()
        elapsed = now - train_start_time

        log = f'{agent.total_steps} {total_reward} {elapsed:.1f}\n'
        print(log, end='')

        with open(log_file_name, 'a') as f:
            f.write(log)


print('Complete')
env.close()
