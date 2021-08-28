import os
from logging import DEBUG, getLogger
from time import time

from comet_ml import Experiment  # isort: split
import gym
import hydra
import numpy as np
import pybullet_envs
import torch

from accel.agents.sac import SAC
from accel.replay_buffers.replay_buffer import ReplayBuffer
from accel.utils.utils import save_as_video, set_seed
from accel.utils.wrappers import RewardScaler

logger = getLogger(__name__)
logger.setLevel(DEBUG)


@hydra.main(config_path='config', config_name='bullet_sac')
def main(cfg):
    set_seed(cfg.seed)
    cwd = hydra.utils.get_original_cwd()

    if cfg.comet:
        comet_username = os.environ['COMET_USERNAME']
        comet_api_token = os.environ['COMET_API_TOKEN']
        logger.debug(f'Comet username: {comet_username}')

    env = RewardScaler(gym.make(cfg.env), scale=cfg.reward_scale)
    eval_env = gym.make(cfg.env)

    if not cfg.device:
        cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    memory = ReplayBuffer(capacity=cfg.replay_capacity, record_size=cfg.record_size,
                          record_outdir=os.path.join(
                              cwd, cfg.record_outdir, cfg.name)
                          )  # TODO assert nstep=1

    agent = SAC(device=cfg.device, observation_space=env.observation_space,
                action_space=env.action_space, gamma=cfg.gamma,
                replay_buffer=memory, update_interval=1, load=cfg.load,
                bullet=True)

    if cfg.demo:
        agent.eval(eval_env, n_epis=10, render=True)
        exit(0)

    next_eval_cnt = 1
    episode_cnt = 0

    score_steps = []
    scores = []

    log_file_name = f'scores.txt'
    best_score = -1e10

    train_start_time = time()

    train_rewards = []

    if cfg.comet:
        comet_exp = Experiment(project_name='accel',
                               api_key=comet_api_token,
                               workspace=comet_username)
        comet_exp.add_tag('bullet_sac')
        comet_exp.add_tag(cfg.env)
        comet_exp.set_name(cfg.name)

        comet_params = {
            'seed': cfg.seed,
            'gamma': cfg.gamma,
            'replay': cfg.replay_capacity,
            'nstep': cfg.nstep,
            'eval_times': cfg.eval_times,
            'env': cfg.env,
        }
        comet_exp.log_parameters(comet_params)

    while agent.total_steps < cfg.steps:
        episode_cnt += 1

        obs = env.reset()
        done = False
        total_reward = 0
        step = 0

        while not done:
            if agent.total_steps > cfg.initial_random_steps:
                action = agent.act(obs)
            else:
                action = env.action_space.sample()

            next_obs, reward, done, _ = env.step(action)
            total_reward += reward
            step += 1

            next_valid = 1 if step == env.spec.max_episode_steps else float(
                not done)
            agent.update(obs, action, next_obs, reward, next_valid)

            obs = next_obs

        logger.info(f'Train episode: {episode_cnt} '
                    f'steps: {step} '
                    f'total_steps:{agent.total_steps} '
                    f'score:{total_reward:.2f} '
                    )
        train_rewards.append(total_reward)

        final_flag = not (agent.total_steps < cfg.steps)
        if agent.total_steps >= next_eval_cnt * cfg.eval_interval:
            gif_flag = next_eval_cnt % cfg.gif_eval_ratio == 0

            next_eval_cnt += 1
            ave_r, rewards, frames = agent.eval(eval_env,
                                                n_epis=cfg.eval_times,
                                                record_n_epis=1)

            score_steps.append(agent.total_steps)
            scores.append(ave_r)
            ave_train_r = np.array(train_rewards).mean()
            train_rewards = []

            if gif_flag:
                gifname = f'eval{agent.total_steps}.gif'
                save_as_video(gifname, frames)
                if cfg.comet:
                    comet_exp.log_image(
                        gifname, name='eval_agent', step=agent.total_steps)
                logger.debug(f'save {gifname}')

            elapsed = time() - train_start_time
            logger.info(
                f'Eval result | total_step: {agent.total_steps} '
                f'score: {ave_r:.1f} train_score: {ave_train_r:.1f}'
                f'  elapsed: {elapsed:.1f}')
            comet_exp.log_metric('reward', ave_r, step=agent.total_steps)

            if ave_r > best_score:
                best_score = ave_r

                model_name = 'best_q1.model'
                torch.save(agent.critic1.state_dict(), model_name)
                if cfg.comet:
                    comet_exp.log_model('best_q1', model_name, overwrite=True)

                model_name = 'best_q2.model'
                torch.save(agent.critic2.state_dict(), model_name)
                if cfg.comet:
                    comet_exp.log_model('best_q2', model_name, overwrite=True)

                model_name = 'best_pi.model'
                torch.save(agent.actor.state_dict(), model_name)
                if cfg.comet:
                    comet_exp.log_model('best_pi', model_name, overwrite=True)
                    comet_exp.log_metric('best_timestep', agent.total_steps)
                    comet_exp.log_metric('best_score', best_score)

                logger.info('save model')

            log = f'{agent.total_steps} {ave_r} {elapsed:.1f}\n'
            with open(log_file_name, 'a') as f:
                f.write(log)

            if final_flag:
                model_name = 'final_q1.model'
                torch.save(agent.critic1.state_dict(), model_name)
                if cfg.comet:
                    comet_exp.log_model('final_q1', model_name)

                model_name = 'final_q2.model'
                torch.save(agent.critic2.state_dict(), model_name)
                if cfg.comet:
                    comet_exp.log_model('final_q2', model_name)

                model_name = 'final_pi.model'
                torch.save(agent.actor.state_dict(), model_name)
                if cfg.comet:
                    comet_exp.log_model('final_pi', model_name)

                logger.info('save final model')

    duration = np.round(elapsed / 60 / 60, 2)
    if cfg.comet:
        comet_exp.log_metric('duration', duration)
        comet_exp.log_artifact(log_file_name)

    print('Complete')
    env.close()


if __name__ == '__main__':
    main()
