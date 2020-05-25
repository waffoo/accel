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

from accel.explorers import epsilon_greedy
from accel.replay_buffers.replay_buffer import Transition, ReplayBuffer
from accel.agents import dqn

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


class Net(nn.Module):
    def __init__(self, input, output, hidden):
        super().__init__()
        self.l1 = nn.Linear(input, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, output)

    def forward(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


def plot_scores(episodes, scores):
    plt.figure(2)
    plt.clf()
    scores = torch.tensor(scores, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(episodes, scores)
    # Take 100 episode averages and plot them too
    window_size = 5
    if len(scores) >= window_size:
        means = scores.unfold(0, window_size, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(window_size - 1), means))
        plt.plot(episodes, means)

    plt.pause(0.001)  # pause a bit so that plots are updated


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

env = gym.make('CartPole-v0').unwrapped
env.seed(seed)
dim_state = len(env.observation_space.low)
dim_action = env.action_space.n

num_episodes = 250
dim_hidden = 32
GAMMA = 0.95
max_episode_len = 200


q_func = Net(dim_state, dim_action, dim_hidden)
optimizer = optim.Adam(q_func.parameters())
memory = ReplayBuffer(capacity=50000)

score_episodes = []
scores = []

explorer = epsilon_greedy.LinearDecayEpsilonGreedy(
    start_eps=0.9, end_eps=0.1, decay_steps=20000)

agent = dqn.DoubleDQN(q_func, optimizer, memory, GAMMA, explorer, device)

for i in range(num_episodes):
    obs = env.reset()
    done = False
    total_reward = 0
    step = 0

    while not done and step < max_episode_len:
        # env.render()
        action = agent.act(obs)
        next_obs, reward, done, _ = env.step(action)
        agent.update(obs, action, next_obs, reward, done)

        total_reward += reward
        step += 1

        obs = next_obs

    if i % 5 == 0:
        obs = env.reset()
        done = False
        total_reward = 0
        step = 0

        while not done and step < max_episode_len:
            action = agent.act(obs, eval=True)
            obs, reward, done, _ = env.step(action)

            total_reward += reward
            step += 1

        score_episodes.append(i)
        scores.append(total_reward)
        plot_scores(score_episodes, scores)
        print(i, agent.total_steps, total_reward)


print('Complete')
env.render()
env.close()
