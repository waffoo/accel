import gym
import torch
from accel.agents.sac_cql import SAC_CQL
import argparse
from accel.replay_buffers.replay_buffer import ReplayBuffer
import d4rl


parser = argparse.ArgumentParser()
parser.add_argument('--env', default='HumanoidBulletEnv-v0',
                    help='name of environment')
parser.add_argument('--load', default=None,
                    help='model path')
parser.add_argument('--demo', action='store_true',
                    help='demo flag')
args = parser.parse_args()

env = gym.make('hopper-expert-v0')
dataset = env.get_dataset()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

memory = ReplayBuffer(capacity=10**6)  # assert nstep=1
agent = SAC_CQL(eval_env=env, outdir='gym-results', cql_weight=5., policy_eval_start=0,
                device=device, observation_space=env.observation_space, action_space=env.action_space,
                gamma=.99, replay_buffer=memory, update_interval=1, load=args.load)
agent.set_dataset(dataset)

num_steps = 5 * 10**6
eval_interval = 5 * 10**3

agent.fit(num_steps, eval_interval)

agent.eval(render=True)
