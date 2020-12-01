from time import time
import gym
# from gym.utils.play import play import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import hydra
import mlflow
import os

from accel.utils.atari_wrappers import make_atari, make_atari_ram
from accel.explorers import epsilon_greedy
from accel.replay_buffers.replay_buffer import Transition, ReplayBuffer
from accel.replay_buffers.prioritized_replay_buffer import PrioritizedReplayBuffer
from accel.agents import ppo
from accel.utils.utils import set_seed
from accel.utils.atari_wrappers import callable_atari_wrapper


class Net(nn.Module):
    def __init__(self, input, output, dueling=False, high_reso=False):
        super().__init__()
        self.dueling = dueling
        self.conv1 = nn.Conv2d(input, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        linear_size = 7 * 7 * 64 if not high_reso else 12 * 12 * 64
        self.fc1 = nn.Linear(linear_size, 512)
        self.fc2 = nn.Linear(512, output)
        if self.dueling:
            self.v_fc1 = nn.Linear(linear_size, 512)
            self.v_fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = x / 255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)

        adv = F.relu(self.fc1(x))
        adv = self.fc2(adv)
        if not self.dueling:
            return adv

        v = F.relu(self.v_fc1(x))
        v = self.v_fc2(v)
        return v + adv - adv.mean(dim=1, keepdim=True)


@hydra.main(config_name='config/atari_dqn_config.yaml')
def main(cfg):
    set_seed(cfg.seed)

    cwd = hydra.utils.get_original_cwd()
    mlflow.set_tracking_uri(os.path.join(cwd, 'mlruns'))
    mlflow.set_experiment('atari_dqn')

    with mlflow.start_run(run_name=cfg.name):
        is_ram = '-ram' in cfg.env

        mlflow.log_param('seed', cfg.seed)
        mlflow.log_param('gamma', cfg.gamma)
        mlflow.log_param('replay', cfg.replay_capacity)
        mlflow.log_param('dueling', cfg.dueling)
        mlflow.log_param('prioritized', cfg.prioritized)
        mlflow.log_param('color', cfg.color)
        mlflow.log_param('high', cfg.high_reso)
        mlflow.log_param('no_stack', cfg.no_stack)
        mlflow.log_param('nstep', cfg.nstep)
        mlflow.log_param('ram', is_ram)
        mlflow.log_param('adam', cfg.adam)
        mlflow.log_param('end_eps', cfg.end_eps)
        mlflow.set_tag('env', cfg.env)

        if not cfg.device:
            cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if is_ram:
            env = make_atari_ram(cfg.env)
            eval_env = make_atari_ram(cfg.env, clip_rewards=False)
        else:
            if cfg.high_reso:
                env = make_atari(cfg.env, color=cfg.color,
                                 image_size=128, frame_stack=not cfg.no_stack)
                eval_env = make_atari(
                    cfg.env, clip_rewards=False, color=cfg.color, image_size=128, frame_stack=not cfg.no_stack)
            else:
                env = make_atari(cfg.env, color=cfg.color,
                                 frame_stack=not cfg.no_stack)
                eval_env = make_atari(
                    cfg.env, clip_rewards=False, color=cfg.color, frame_stack=not cfg.no_stack)

        env.seed(cfg.seed)
        eval_env.seed(cfg.seed)

        dim_state = env.observation_space.shape[0]
        dim_action = env.action_space.n

        q_func = Net(dim_state, dim_action,
                     dueling=cfg.dueling, high_reso=cfg.high_reso)

        if cfg.load:
            q_func.load_state_dict(torch.load(os.path.join(
                cwd, cfg.load), map_location=cfg.device))

        if cfg.adam:
            # same as Rainbow
            optimizer = optim.Adam(q_func.parameters(), lr=0.0000625, eps=1.5e-4)
        else:
            optimizer = optim.RMSprop(
                q_func.parameters(), lr=0.00025, alpha=0.95, eps=1e-2)

        if cfg.prioritized:
            memory = PrioritizedReplayBuffer(capacity=cfg.replay_capacity, beta_steps=cfg.steps - cfg.replay_start_step, nstep=cfg.nstep)
        else:
            memory = ReplayBuffer(capacity=cfg.replay_capacity, nstep=cfg.nstep)

        score_steps = []
        scores = []

        explorer = epsilon_greedy.LinearDecayEpsilonGreedy(
            start_eps=1.0, end_eps=cfg.end_eps, decay_steps=1e6)

        wrapper = callable_atari_wrapper(color=cfg.color, frame_stack=not cfg.no_stack)
        eval_wrapper = callable_atari_wrapper(color=cfg.color, frame_stack=not cfg.no_stack, clip_rewards=False)

        envs = gym.vector.make(cfg.env, 8, wrappers=wrapper)
        eval_envs = gym.vector.make(cfg.env, 8, wrappers=eval_wrapper)

        agent2 = ppo.PPO(envs, eval_envs, optimizer)

        agent2.run()

if __name__ == '__main__':
    main()
