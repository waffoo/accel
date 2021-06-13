import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from subprocess import call
from os import system


class Net(nn.Module):
    def __init__(self, input, output, high_reso=False):
        super().__init__()
        self.conv1 = nn.Conv2d(input, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        linear_size = 7 * 7 * 64 if not high_reso else 12 * 12 * 64
        self.fc1 = nn.Linear(linear_size, 512)
        self.fc2 = nn.Linear(512, output)

    def forward(self, x):
        x = x / 255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)

        adv = F.relu(self.fc1(x))
        adv = self.fc2(adv)
        return F.softmax(adv, dim=-1)

from accel.utils.atari_wrappers import make_atari

eval_env = make_atari(
    'PongNoFrameskip-v4', clip_rewards=False, color=True, image_size=128, frame_stack=False)
dim_state = eval_env.observation_space.shape[0]
dim_action = eval_env.action_space.n

import os
import numpy as np
data_path = os.path.join('dataset', 'pong-1s', '9.npz')
print(f'open {data_path}...')
dataset = np.load(data_path)

observations, actions = dataset['state'], dataset['action']

observations = torch.tensor(observations)
actions = torch.tensor(actions)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

obs_train, obs_test, action_train, action_test = train_test_split(observations, actions,
                                                                  test_size=0.2, random_state=0)

batch_size = 64
epochs = 20

trainset = TensorDataset(obs_train, action_train)
testset = TensorDataset(obs_test, action_test)

train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

net = Net(dim_state, dim_action, high_reso=True).to(device)
print(net)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

for epoch in range(epochs):
    total_loss = 0
    net.train()
    print('train', epoch)

    for i, (obs, action) in enumerate(train_loader):
        obs, action = obs.to(device), action.long().to(device)
        pred = net(obs)
        loss = F.cross_entropy(pred, action)

        # loss = F.mse_loss(pred, action)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    net.eval()
    total_test_loss = 0
    for i, (obs, action) in enumerate(test_loader):
        obs, action = obs.to(device), action.long().to(device)
        with torch.no_grad():
            pred = net(obs)
        loss = F.cross_entropy(pred, action)
        total_test_loss += loss.item()

    print(f'epoch {epoch} / train_loss: {total_loss / len(train_loader)}')
    print(f'epoch {epoch} / test_loss: {total_test_loss / len(test_loader)}')

    print('policy eval')
    total_reward = 0
    obs = eval_env.reset()
    obs = torch.tensor(obs, dtype=torch.float32)[None]

    done = False
    t = 0

    while not done:
        obs = obs.to(device)
        t += 1
        with torch.no_grad():
            action = net(obs).detach().cpu().numpy().argmax()
        from PIL import Image
        arr = eval_env.render(mode='rgb_array')
        img = Image.fromarray(arr)
        img.save(f'gym-results/{t:04d}.png')

        obs, reward, done, _ = eval_env.step(action)
        obs = torch.tensor(obs, dtype=torch.float32)[None]
        total_reward += reward

    system(f'ffmpeg -hide_banner -loglevel panic -f image2 -r 30 -y -i '
           f'gym-results/%04d.png -an -vcodec libx264 -pix_fmt yuv420p gym-results/pong{epoch}.mp4')
    system(f'rm gym-results/*.png')

    print(f'epoch {epoch} / reward: {total_reward}')


