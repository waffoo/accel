import time
import gym
# from gym.utils.play import play import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import hydra
import mlflow
import os

from accel.utils.atari_wrappers import make_atari, make_atari_ram
from accel.explorers import epsilon_greedy
from accel.replay_buffers.replay_buffer import Transition, ReplayBuffer
from accel.replay_buffers.prioritized_replay_buffer import PrioritizedReplayBuffer
from accel.agents import dqn
from accel.utils.utils import set_seed


class Net(nn.Module):
    def __init__(self, input, output, dueling=False):
        super().__init__()
        self.dueling = dueling
        self.conv1 = nn.Conv2d(input, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, output)
        if self.dueling:
            self.v_fc1 = nn.Linear(7 * 7 * 64, 512)
            self.v_fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = x / 255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        adv = F.relu(self.fc1(x))
        adv = self.fc2(adv)
        if not self.dueling:
            return adv

        v = F.relu(self.v_fc1(x))
        v = self.v_fc2(v)
        return v + adv - adv.mean(dim=1, keepdim=True)


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


@hydra.main(config_name='config/atari_dqn_config.yaml')
def main(cfg):
    set_seed(cfg.seed)

    cwd = hydra.utils.get_original_cwd()
    mlflow.set_tracking_uri(os.path.join(cwd, 'mlruns'))
    mlflow.set_experiment('atari_dqn')

    with mlflow.start_run(run_name=cfg.name):
        mlflow.log_param('seed', cfg.seed)
        mlflow.log_param('gamma', cfg.gamma)
        mlflow.log_param('replay', cfg.replay_capacity)
        mlflow.log_param('dueling', cfg.dueling)
        mlflow.log_param('prioritized', cfg.prioritized)
        mlflow.set_tag('env', cfg.env)

        if not cfg.device:
            cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        is_ram = '-ram' in cfg.env
        if is_ram:
            env = make_atari_ram(cfg.env)
            eval_env = make_atari_ram(cfg.env, clip_rewards=False)
        else:
            env = make_atari(cfg.env)
            eval_env = make_atari(cfg.env, clip_rewards=False)


        env.seed(cfg.seed)
        eval_env.seed(cfg.seed)

        dim_state = env.observation_space.shape[0]
        dim_action = env.action_space.n

        if is_ram:
            q_func = RamNet(dim_state, dim_action)
        else:
            q_func = Net(dim_state, dim_action, dueling=cfg.dueling)

        if cfg.load:
            q_func.load_state_dict(torch.load(os.path.join(cwd, cfg.load), map_location=cfg.device))

        optimizer = optim.RMSprop(
            q_func.parameters(), lr=0.00025, alpha=0.95, eps=1e-2)

        if cfg.prioritized:
            memory = PrioritizedReplayBuffer(capacity=cfg.replay_capacity, beta_steps=cfg.steps)
        else:
            memory = ReplayBuffer(capacity=cfg.replay_capacity)

        score_steps = []
        scores = []

        explorer = epsilon_greedy.LinearDecayEpsilonGreedy(
            start_eps=1.0, end_eps=0.1, decay_steps=1e6)

        agent = dqn.DoubleDQN(q_func, optimizer, memory, cfg.gamma,
                              explorer, cfg.device, batch_size=32, target_update_interval=10000)

        if cfg.demo:
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

                        # print(reward)
                        total_reward += reward

                    if eval_env.was_real_done:
                        break

                print('Episode:', x, 'Score:', total_reward)

            exit(0)

        next_eval_cnt = 1
        episode_cnt = 0

        train_start_time = time.time()

        log_file_name = 'scores.txt'
        best_score = -1e10

        while agent.total_steps < cfg.steps:
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

                next_valid = 1 if step == env.spec.max_episode_steps else float(
                    not done)
                agent.update(obs, action, next_obs, reward, next_valid)

                obs = next_obs

            if agent.total_steps > next_eval_cnt * cfg.eval_interval:
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
                    model_name = f'{agent.total_steps}.model'
                    torch.save(q_func.state_dict(), model_name)
                    best_score = total_reward

                now = time.time()
                elapsed = now - train_start_time

                log = f'{agent.total_steps} {total_reward} {elapsed:.1f}\n'
                print(log, end='')
                mlflow.log_metric('reward', total_reward, step=agent.total_steps)

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

        model_name = f'final.model'
        torch.save(q_func.state_dict(), model_name)

        now = time.time()
        elapsed = now - train_start_time

        log = f'{agent.total_steps} {total_reward} {elapsed:.1f}\n'
        print(log, end='')
        mlflow.log_metric('reward', total_reward, step=agent.total_steps)

        with open(log_file_name, 'a') as f:
            f.write(log)

        print('Complete')
        env.close()


if __name__ == '__main__':
    main()
