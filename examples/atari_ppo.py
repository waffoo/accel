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

from accel.utils.atari_wrappers import make_atari
from accel.explorers import epsilon_greedy
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


@hydra.main(config_name='config/atari_ppo_config.yaml')
def main(cfg):
    set_seed(cfg.seed)

    cwd = hydra.utils.get_original_cwd()
    if cfg.load:
        cfg.load = os.path.join(cwd, cfg.load)
    mlflow.set_tracking_uri(os.path.join(cwd, 'mlruns'))
    mlflow.set_experiment('atari_ppo')

    with mlflow.start_run(run_name=cfg.name):
        mlflow.log_param('seed', cfg.seed)
        mlflow.log_param('gamma', cfg.gamma)
        mlflow.log_param('parallel', cfg.parallel)
        mlflow.log_param('color', cfg.color)
        mlflow.log_param('high', cfg.high_reso)
        mlflow.log_param('no_stack', cfg.no_stack)
        mlflow.set_tag('env', cfg.env)


        if not cfg.device:
            cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        wrapper = callable_atari_wrapper(color=cfg.color, frame_stack=not cfg.no_stack)
        eval_wrapper = callable_atari_wrapper(color=cfg.color, frame_stack=not cfg.no_stack, clip_rewards=False)

        envs = gym.vector.make(cfg.env, cfg.parallel, wrappers=wrapper)
        eval_env = make_atari(cfg.env, color=cfg.color, frame_stack=not cfg.no_stack, clip_rewards=False)

        dim_state = envs.observation_space.shape[1]
        dim_action = envs.action_space[0].n

        agent2 = ppo.PPO(envs, eval_env, dim_state, dim_action, cfg.steps, lmd=0.9, gamma=cfg.gamma,
                            device=cfg.device, batch_size=128, load=cfg.load, eval_interval=cfg.eval_interval)

        agent2.run()


if __name__ == '__main__':
    main()
