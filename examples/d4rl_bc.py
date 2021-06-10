import gym
import d4rl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from subprocess import call
from os import system


class BCNet(nn.Module):
    def __init__(self, insize, outsize):
        super().__init__()
        mid1 = 400
        mid2 = 300
        self.l1 = nn.Linear(insize, mid1)
        self.l2 = nn.Linear(mid1, mid2)
        self.l3 = nn.Linear(mid2, outsize)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)


env = gym.make('walker2d-medium-expert-v0')
dataset = env.get_dataset()
#actions, observations, rewards, terminals, timeouts
# d4rl.qlearning_dataset(env)

observations, actions = dataset['observations'], dataset['actions']

observations = torch.tensor(observations)
actions = torch.tensor(actions)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

obs_train, obs_test, action_train, action_test = train_test_split(observations, actions,
                                                                  test_size=0.2, random_state=0)

batch_size = 64
epochs = 100 // 20

trainset = TensorDataset(obs_train, action_train)
testset = TensorDataset(obs_test, action_test)

train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

net = BCNet(obs_train.size(1), action_train.size(1)).to(device)
print(net)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

for epoch in range(epochs):
    total_loss = 0
    net.train()
    print('train', epoch)

    for i, (obs, action) in enumerate(train_loader):
        obs, action = obs.to(device), action.to(device)
        pred = net(obs)
        loss = F.mse_loss(pred, action)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    net.eval()
    total_test_loss = 0
    for i, (obs, action) in enumerate(test_loader):
        obs, action = obs.to(device), action.to(device)
        with torch.no_grad():
            pred = net(obs)
        loss = F.mse_loss(pred, action)
        total_test_loss += loss.item()

    print(f'epoch {epoch} / train_loss: {total_loss / len(train_loader)}')
    print(f'epoch {epoch} / test_loss: {total_test_loss / len(test_loader)}')

    print('policy eval')
    total_reward = 0
    obs = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32)[None]

    done = False
    t = 0

    while not done:
        obs = obs.to(device)
        t += 1
        with torch.no_grad():
            action = net(obs).detach().cpu().numpy()
        from PIL import Image
        arr = env.render(mode='rgb_array')
        img = Image.fromarray(arr)
        img.save(f'gym-results/{t:04d}.png')

        obs, reward, done, _ = env.step(action)
        obs = torch.tensor(obs, dtype=torch.float32)[None]
        total_reward += reward

    system(f'ffmpeg -hide_banner -loglevel panic -f image2 -r 30 -y -i '
           f'gym-results/%04d.png -an -vcodec libx264 -pix_fmt yuv420p gym-results/out{epoch}.mp4')
    system(f'rm gym-results/*.png')

    print(f'epoch {epoch} / reward: {total_reward}')


