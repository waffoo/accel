import os
from logging import DEBUG, getLogger
from time import time

import gym
import hydra
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from accel.agents import dqn
from accel.explorers import epsilon_greedy
from accel.replay_buffers.prioritized_replay_buffer import \
    PrioritizedReplayBuffer
from accel.replay_buffers.replay_buffer import ReplayBuffer
from accel.utils.atari_wrappers import make_atari
from accel.utils.utils import save_as_video, set_seed

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


@hydra.main(config_path='config', config_name='atari_dqn_config')
def main(cfg):
    set_seed(cfg.seed)

    cwd = hydra.utils.get_original_cwd()

    if not cfg.device:
        cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    img_size = 128 if cfg.high_reso else 84
    env = make_atari(cfg.env, color=cfg.color,
                     image_size=img_size, frame_stack=not cfg.no_stack)
    eval_env = make_atari(cfg.env, color=cfg.color,
                          image_size=img_size, frame_stack=not cfg.no_stack, clip_rewards=False)

    env.seed(cfg.seed)
    eval_env.seed(cfg.seed + 1)

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
        memory = PrioritizedReplayBuffer(
            capacity=cfg.replay_capacity,
            beta_steps=cfg.steps -
            cfg.replay_start_step,
            nstep=cfg.nstep)
    else:
        memory = ReplayBuffer(capacity=cfg.replay_capacity, nstep=cfg.nstep, record=cfg.record, record_size=cfg.record_size,
                              record_outdir=os.path.join(cwd, cfg.record_outdir, cfg.name))

    explorer = epsilon_greedy.LinearDecayEpsilonGreedy(
        start_eps=1.0, end_eps=cfg.end_eps, decay_steps=1e6)

    agent = dqn.DoubleDQN(q_func, optimizer, memory, cfg.gamma,
                          explorer, cfg.device, batch_size=32,
                          target_update_interval=10000,
                          replay_start_step=cfg.replay_start_step)

    if cfg.demo:
        agent.eval(eval_env, n_epis=10, render=True)
        exit(0)

    next_eval_cnt = 1
    episode_cnt = 0

    train_start_time = time()

    log_file_name = 'scores.txt'
    best_score = -1e10

    mlflow.set_tracking_uri(os.path.join(cwd, 'mlruns'))
    mlflow.set_experiment('atari_dqn')

    with mlflow.start_run(run_name=cfg.name):

        mlflow.log_param('seed', cfg.seed)
        mlflow.log_param('gamma', cfg.gamma)
        mlflow.log_param('replay', cfg.replay_capacity)
        mlflow.log_param('dueling', cfg.dueling)
        mlflow.log_param('prioritized', cfg.prioritized)
        mlflow.log_param('color', cfg.color)
        mlflow.log_param('high', cfg.high_reso)
        mlflow.log_param('no_stack', cfg.no_stack)
        mlflow.log_param('nstep', cfg.nstep)
        mlflow.log_param('adam', cfg.adam)
        mlflow.log_param('end_eps', cfg.end_eps)
        mlflow.log_param('eval_times', cfg.eval_times)
        mlflow.set_tag('env', cfg.env)

        while agent.total_steps < cfg.steps:
            episode_cnt += 1

            train_frames = []
            total_reward = 0
            step = 0
            while True:
                obs = env.reset()
                if cfg.train_record:
                    agent._add_obs_to_frame(obs, train_frames)
                done = False

                while not done:
                    action = agent.act(obs)
                    next_obs, reward, done, info = env.step(action)
                    total_reward += reward
                    step += 1

                    timeout_label = 'TimeLimit.truncated'
                    timeout = hasattr(
                        info, timeout_label) and info[timeout_label]
                    next_valid = 1. if timeout else float(not done)
                    agent.update(obs, action, next_obs, reward, next_valid)

                    obs = next_obs
                    if cfg.train_record:
                        agent._add_obs_to_frame(obs, train_frames)

                if hasattr(env, 'was_real_done') and env.was_real_done:
                    break

            logger.info(f'Train episode: {episode_cnt} '
                        f'steps: {step} '
                        f'total_steps:{agent.total_steps} '
                        f'score:{total_reward}')

            final_flag = not (agent.total_steps < cfg.steps)

            if agent.total_steps > next_eval_cnt * cfg.eval_interval or final_flag:
                if cfg.train_record:
                    gifname = f'train{agent.total_steps}.gif'
                    save_as_video(gifname, train_frames)
                    mlflow.log_artifact(gifname, artifact_path='train')
                    logger.debug(f'save {gifname}')

                next_eval_cnt += 1
                ave_r, rewards, frames = agent.eval(eval_env,
                                                    n_epis=cfg.eval_times,
                                                    record_n_epis=1)

                gifname = f'eval{agent.total_steps}.gif'
                save_as_video(gifname, frames)
                mlflow.log_artifact(gifname, artifact_path='eval')
                logger.debug(f'save {gifname}')

                elapsed = time() - train_start_time
                logger.info(
                    f'Eval result | total_step: {agent.total_steps} '
                    f'score: {ave_r} elapsed: {elapsed:.1f}')
                mlflow.log_metric('reward', ave_r, step=agent.total_steps)

                log = f'{agent.total_steps} {ave_r} {elapsed:.1f}\n'
                with open(log_file_name, 'a') as f:
                    f.write(log)

                if ave_r > best_score:
                    model_name = f'best.model'
                    torch.save(q_func.state_dict(), model_name)
                    mlflow.log_artifact(model_name)

                    best_ts_file = 'best_timestep.txt'
                    best_sc_file = 'best_score.txt'
                    with open(best_ts_file, 'w') as f:
                        f.write(f'{agent.total_steps}\n')
                    with open(best_sc_file, 'w') as f:
                        f.write(f'{ave_r}\n')
                    mlflow.log_artifact(model_name)
                    mlflow.log_artifact(best_ts_file)
                    mlflow.log_artifact(best_sc_file)

                    logger.info(f'save {model_name}')
                    best_score = ave_r

                if final_flag:
                    model_name = f'final.model'
                    torch.save(q_func.state_dict(), model_name)
                    mlflow.log_artifact(model_name)
                    logger.info(f'save {model_name}')

        duration = np.round(elapsed / 60 / 60, 2)
        mlflow.log_metric('duration', duration)
        print('Complete')
        env.close()


if __name__ == '__main__':
    main()
