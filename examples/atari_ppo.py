import os
from logging import DEBUG, getLogger
from time import time

from comet_ml import Experiment  # isort: split
import gym
import hydra
import numpy as np
# from gym.utils.play import play import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from accel.agents import ppo
from accel.explorers import epsilon_greedy
from accel.utils.atari_wrappers import callable_atari_wrapper, make_atari
from accel.utils.utils import set_seed

logger = getLogger(__name__)
logger.setLevel(DEBUG)


def init(module, weight_init, bias_init, gain=1.):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class ActorNet(nn.Module):
    def __init__(self, input, output, high_reso=False):
        super().__init__()

        def init_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.
                        constant_(x, 0), nn.init.calculate_gain('relu'))
        self.conv1 = init_(nn.Conv2d(input, 32, kernel_size=8, stride=4))
        self.conv2 = init_(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = init_(nn.Conv2d(64, 64, kernel_size=3, stride=1))

        linear_size = 7 * 7 * 64 if not high_reso else 12 * 12 * 64
        self.fc1 = init_(nn.Linear(linear_size, 512))

        def init_(m):
            return init(
                m,
                nn.init.orthogonal_,
                lambda x: nn.init.constant_(x, 0), gain=0.01)

        self.fc2 = init_(nn.Linear(512, output))

    def forward(self, x):
        x = x / 255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)

        adv = F.relu(self.fc1(x))
        adv = self.fc2(adv)

        # return raw logits
        return adv


class CriticNet(nn.Module):
    def __init__(self, input, high_reso=False):
        super().__init__()

        def init_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.
                        constant_(x, 0), nn.init.calculate_gain('relu'))
        self.conv1 = init_(nn.Conv2d(input, 32, kernel_size=8, stride=4))
        self.conv2 = init_(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = init_(nn.Conv2d(64, 64, kernel_size=3, stride=1))

        linear_size = 7 * 7 * 64 if not high_reso else 12 * 12 * 64
        self.fc1 = init_(nn.Linear(linear_size, 512))

        def init_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.
                        constant_(x, 0))
        self.fc2 = init_(nn.Linear(512, 1))

    def forward(self, x):
        x = x / 255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)

        adv = F.relu(self.fc1(x))
        adv = self.fc2(adv)
        return adv


class ActorMLPNet(nn.Module):
    def __init__(self, input, output, hidden=64, high_reso=None):
        super().__init__()
        self.l1 = nn.Linear(input, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, output)

    def forward(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


class CriticMLPNet(nn.Module):
    def __init__(self, input, hidden=64, high_reso=None):
        super().__init__()
        self.l1 = nn.Linear(input, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, 1)

    def forward(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


@hydra.main(config_path='config', config_name='atari_ppo_config')
def main(cfg):
    set_seed(cfg.seed)

    cwd = hydra.utils.get_original_cwd()
    if cfg.comet:
        comet_username = os.environ['COMET_USERNAME']
        comet_api_token = os.environ['COMET_API_TOKEN']
        logger.debug(f'Comet username: {comet_username}')

    if cfg.load:
        cfg.load = os.path.join(cwd, cfg.load)

    if not cfg.device:
        cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if cfg.atari:
        wrapper = callable_atari_wrapper(
            color=cfg.color, frame_stack=not cfg.no_stack)
        envs = gym.vector.make(cfg.env, cfg.parallel, wrappers=wrapper)
        eval_env = make_atari(
            cfg.env, color=cfg.color, frame_stack=not cfg.no_stack, clip_rewards=False)
    else:
        envs = gym.vector.make(cfg.env, cfg.parallel)
        eval_env = gym.make(cfg.env)

    dim_state = envs.observation_space.shape[1]
    dim_action = envs.action_space[0].n

    if len(eval_env.observation_space.shape) == 3:
        actor = ActorNet(dim_state, dim_action, high_reso=cfg.high_reso)
        critic = CriticNet(dim_state, high_reso=cfg.high_reso)
    else:
        actor = ActorMLPNet(dim_state, dim_action, high_reso=cfg.high_reso)
        critic = CriticMLPNet(dim_state, high_reso=cfg.high_reso)

    if cfg.comet:
        comet_exp = Experiment(project_name='accel',
                               api_key=comet_api_token,
                               workspace=comet_username)
        comet_exp.add_tag('atari_ppo')
        comet_exp.add_tag(cfg.env)
        comet_exp.set_name(cfg.name)

        comet_params = {
            'seed': cfg.seed,
            'gamma': cfg.gamma,
            'parallel': cfg.parallel,
            'color': cfg.color,
            'high': cfg.high_reso,
            'no_stack': cfg.no_stack,
            'batch_size': cfg.batch_size,
            'value_loss_coef': cfg.value_loss_coef,
            'value_clipping': cfg.value_clipping,
            'env': cfg.env,
        }

        comet_exp.log_parameters(comet_params)
    else:
        comet_exp = None

    agent2 = ppo.PPO(envs, eval_env, cfg.steps,
                     actor=actor, critic=critic, lmd=0.9, gamma=cfg.gamma,
                     device=cfg.device, batch_size=cfg.batch_size, load=cfg.load, eval_interval=cfg.eval_interval,
                     clip_eps=0.1, comet_exp=comet_exp, value_loss_coef=cfg.value_loss_coef,
                     value_clipping=cfg.value_clipping, atari=True)

    agent2.run()


if __name__ == '__main__':
    main()
