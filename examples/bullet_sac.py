import argparse
import datetime
import os
import random
import time

import gym
import hydra
import numpy as np
import pybullet_envs
import torch
import torch.nn as nn
import torch.nn.functional as F

from accel.agents.sac import SAC
from accel.replay_buffers.replay_buffer import ReplayBuffer
from accel.utils.utils import set_seed
from accel.utils.wrappers import RewardScaler


@hydra.main(config_path='config', config_name='bullet_sac')
def main(cfg):
    set_seed(cfg.seed)

    env = RewardScaler(gym.make(cfg.env), scale=cfg.reward_scale)
    eval_env = gym.make(cfg.env)
    # env = gym.wrappers.Monitor(env, 'movie', force=True)
    # env.render(mode='human')

    if not cfg.device:
        cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    memory = ReplayBuffer(capacity=cfg.replay_capacity)  # TODO assert nstep=1

    agent = SAC(device=cfg.device, observation_space=env.observation_space,
                action_space=env.action_space, gamma=cfg.gamma,
                replay_buffer=memory, update_interval=1, load=cfg.load,
                bullet=True)

    if cfg.demo:
        agent.eval(eval_env, n_epis=10, render=True, record_n_epis=1)
        exit(0)

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

    while agent.total_steps < cfg.steps:
        episode_cnt += 1

        obs = env.reset()
        done = False
        total_reward = 0
        step = 0

        while not done:
            if agent.total_steps > cfg.initial_random_steps:
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

        if agent.total_steps >= next_eval_cnt * cfg.eval_interval:
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


if __name__ == '__main__':
    main()
