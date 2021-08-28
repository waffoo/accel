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

from accel.agents import dqn_cql
from accel.explorers import epsilon_greedy
from accel.replay_buffers.prioritized_replay_buffer import \
    PrioritizedReplayBuffer
from accel.replay_buffers.replay_buffer import ReplayBuffer, Transition
from accel.utils.atari_wrappers import make_atari
from accel.utils.utils import set_seed

logger = getLogger(__name__)
logger.setLevel(DEBUG)


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


@hydra.main(config_path='config', config_name='atari_cql_config')
def main(cfg):
    set_seed(cfg.seed)

    cwd = hydra.utils.get_original_cwd()

    if cfg.comet:
        comet_username = os.environ['COMET_USERNAME']
        comet_api_token = os.environ['COMET_API_TOKEN']
        logger.debug(f'Comet username: {comet_username}')

    if not cfg.device:
        cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        optimizer = optim.Adam(q_func.parameters(),
                               lr=0.0000625, eps=1.5e-4)
    else:
        optimizer = optim.RMSprop(
            q_func.parameters(), lr=0.00025, alpha=0.95, eps=1e-2)

    if cfg.prioritized:
        memory = PrioritizedReplayBuffer(
            capacity=cfg.replay_capacity, beta_steps=cfg.steps - cfg.replay_start_step, nstep=cfg.nstep)
    else:
        memory = ReplayBuffer(capacity=cfg.replay_capacity, nstep=cfg.nstep, record=True, record_size=cfg.record_size,
                              record_outdir=os.path.join(cwd, cfg.record_outdir, cfg.name))

    explorer = epsilon_greedy.LinearDecayEpsilonGreedy(
        start_eps=1.0, end_eps=cfg.end_eps, decay_steps=1e6)

    if cfg.comet:
        comet_exp = Experiment(project_name='accel',
                               api_key=comet_api_token,
                               workspace=comet_username)
        comet_exp.add_tag('atari_cql')
        comet_exp.add_tag(cfg.env)
        comet_exp.set_name(cfg.name)

        comet_params = {
            'seed': cfg.seed,
            'gamma': cfg.gamma,
            'replay': cfg.replay_capacity,
            'dueling': cfg.dueling,
            'prioritized': cfg.prioritized,
            'color': cfg.color,
            'high': cfg.high_reso,
            'no_stack': cfg.no_stack,
            'nstep': cfg.nstep,
            'adam': cfg.adam,
            'end_eps': cfg.end_eps,
            'env': cfg.env,
        }

        comet_exp.log_parameters(comet_params)
    else:
        comet_exp = None

    outdir = os.path.join(cwd, 'results', cfg.name)
    agent = dqn_cql.DQN_CQL(eval_env=eval_env, outdir=outdir, cql_weight=4.,
                            q_func=q_func, optimizer=optimizer,
                            replay_buffer=memory, gamma=cfg.gamma,
                            explorer=explorer, device=cfg.device, batch_size=32,
                            target_update_interval=10000,
                            replay_start_step=cfg.replay_start_step,
                            comet_exp=comet_exp
                            )

    data_path = os.path.join(cwd, 'dataset', 'pong-1s', '9.npz')
    print(f'open {data_path}...')
    dataset = np.load(data_path)
    print(dataset.files)

    agent.set_dataset(dataset)

    agent.fit(cfg.steps, cfg.eval_interval)

    agent.eval(render=True)

    now = time()
    elapsed = now - agent.train_start_time

    duration = np.round(elapsed / 60 / 60, 2)
    if cfg.comet:
        comet_exp.log_metric('duration', duration)

    print('Complete')
    env.close()


if __name__ == '__main__':
    main()
