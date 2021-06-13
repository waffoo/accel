# Accel
[![build](https://img.shields.io/circleci/build/github/waffoo/accel?logo=circleci)](https://app.circleci.com/pipelines/github/waffoo/accel)

Accel is an open source library for reinforcement learning.

## Install
You can install Accel via pip (>=19.0) by entering the following command:
```
$ pip install git+https://github.com/waffoo/accel
```

You can also install it from the source code:
```
$ pip install .
```

Some examples are constructed on top of [D4RL](https://github.com/rail-berkeley/d4rl). Please install it if necessary.

## Algorithms
Following algorithms have been implemented:
- [DQN](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf)
  (including [Double DQN](https://arxiv.org/abs/1509.06461), [Dueling DQN](https://arxiv.org/abs/1511.06581), [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952), and Multi-step DQN)
- [Soft Actor-Critic (SAC)](https://arxiv.org/abs/1801.01290)
- [PPO](https://arxiv.org/abs/1707.06347)
- Behavioral Chroning (BC)
  - Only examples are provided
- [Conservative Q-Learning (CQL)](https://arxiv.org/abs/2006.04779) (+ Double DQN, SAC)

## Lisense
[MIT License](LICENSE)
