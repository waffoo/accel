import gym
from accel.utils.procgen_wrappers import callable_procgen_wrapper
import torch
import torch.nn as nn
import torch.nn.functional as F
from accel.agents.ppo import PPO
import hydra
import mlflow
from accel.utils.utils import set_seed
import os

def init(module, weight_init, bias_init, gain=1.):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class ActorNet(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))
        self.conv1 = init_(nn.Conv2d(input, 32, kernel_size=8, stride=4))
        self.conv2 = init_(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = init_(nn.Conv2d(64, 64, kernel_size=3, stride=1))

        linear_size = 4 * 4 * 64
        self.fc1 = init_(nn.Linear(linear_size, 512))

        init_ = lambda m: init(
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
    def __init__(self, input):
        super().__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))
        self.conv1 = init_(nn.Conv2d(input, 32, kernel_size=8, stride=4))
        self.conv2 = init_(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = init_(nn.Conv2d(64, 64, kernel_size=3, stride=1))

        linear_size = 4 * 4 * 64
        self.fc1 = init_(nn.Linear(linear_size, 512))

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
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


@hydra.main(config_name='config/procgen_ppo_config.yaml')
def main(cfg):
    set_seed(cfg.seed)

    cwd = hydra.utils.get_original_cwd()
    if cfg.load:
        cfg.load = os.path.join(cwd, cfg.load)

    if not cfg.device:
        cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    mlflow.set_tracking_uri(os.path.join(cwd, 'mlruns'))
    mlflow.set_experiment('procgen')

    if cfg.demo:
        wrapper = callable_procgen_wrapper(frame_stack=not cfg.no_stack, color=cfg.color)
        eval_wrapper = callable_procgen_wrapper(frame_stack=not cfg.no_stack, color=cfg.color, clip_rewards=False)
        envs = gym.vector.make(cfg.env, cfg.parallel, start_level=0, num_levels=cfg.num_levels,
                               distribution_mode='easy', wrappers=wrapper)

        eval_env = gym.make(cfg.env, start_level=cfg.num_levels, num_levels=100_000, distribution_mode='easy',
                            render_mode='human' if not cfg.image_demo else 'rgb_array')
        eval_env = eval_wrapper(eval_env)

        dim_state = envs.observation_space.shape[1]
        dim_action = envs.action_space[0].n

        actor = ActorNet(dim_state, dim_action)
        critic = CriticNet(dim_state)

        agent = PPO(envs, eval_env, steps=cfg.steps, actor=actor, critic=critic, lmd=0.9,
                    gamma=cfg.gamma, device=cfg.device,
                    batch_size=cfg.batch_size, load=cfg.load, eval_interval=cfg.eval_interval, clip_eps=0.1,
                    mlflow=True, value_loss_coef=cfg.value_loss_coef, value_clipping=True,
                    epoch_per_eval=cfg.epoch_per_eval, horizon=cfg.horizon)
        agent.demo(image_demo=cfg.image_demo, steps=50000, episodes=None, out_dir=cfg.demo_outdir)
        exit(0)


    with mlflow.start_run(run_name=cfg.name):
        mlflow.log_param('seed', cfg.seed)
        mlflow.log_param('gamma', cfg.gamma)
        mlflow.log_param('parallel', cfg.parallel)
        mlflow.log_param('color', cfg.color)
        mlflow.log_param('no_stack', cfg.no_stack)
        mlflow.log_param('batch_size', cfg.batch_size)
        mlflow.log_param('horizon', cfg.horizon)
        mlflow.log_param('num_levels', cfg.num_levels)
        mlflow.set_tag('env', cfg.env)
        mlflow.set_tag('hostname', os.uname()[1])

        wrapper = callable_procgen_wrapper(frame_stack=not cfg.no_stack, color=cfg.color)
        eval_wrapper = callable_procgen_wrapper(frame_stack=not cfg.no_stack, color=cfg.color, clip_rewards=False)
        envs = gym.vector.make(cfg.env, cfg.parallel, start_level=0, num_levels=cfg.num_levels,
                               distribution_mode='easy', wrappers=wrapper)

        eval_env = gym.make(cfg.env, start_level=cfg.num_levels, num_levels=100_000, distribution_mode='easy')
        eval_env = eval_wrapper(eval_env)


        mlflow.set_tag('device', cfg.device)

        '''
        obs = envs.reset()
        obs, reward, done, info = envs.step(envs.action_space.sample())
        for i in range(8):
            print(info[i]['level_seed'])
        '''

        dim_state = envs.observation_space.shape[1]
        dim_action = envs.action_space[0].n

        actor = ActorNet(dim_state, dim_action)
        critic = CriticNet(dim_state)

        agent = PPO(envs, eval_env, steps=cfg.steps, actor=actor, critic=critic, lmd=0.9,
                    gamma=cfg.gamma, device=cfg.device,
                    batch_size=cfg.batch_size, load=cfg.load, eval_interval=cfg.eval_interval, clip_eps=0.1,
                    mlflow=True, value_loss_coef=cfg.value_loss_coef, value_clipping=True,
                    epoch_per_eval=cfg.epoch_per_eval, horizon=cfg.horizon)
        agent.run()


if __name__ == '__main__':
    main()