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

## Algorithms
Following algorithms have been implemented:
- [DQN](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf)
  (including [Double DQN](https://arxiv.org/abs/1509.06461), [Dueling DQN](https://arxiv.org/abs/1511.06581), [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952), and Multi-step DQN)
- [PPO](https://arxiv.org/abs/1707.06347)
- [Soft Actor-Critic](https://arxiv.org/abs/1801.01290)

## Lisense
[MIT License](LICENSE)
