import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import torch
from tqdm import *

import gym
import gym_oscillator

from algo.PPO_torch.ppo2_torch import PPOTorch

from options.base_options import TrainOptions, TestOptions
from utils.env_utils import ParamGenerator, EnvProcessor
from utils.normal_utils import samples_reshape

from utils.plot_utils import plot_train_rewards, plot_loss
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def train():
    opt = TrainOptions().parse()

    num_iteration = opt.num_iteration
    batch_size = opt.batch_size
    num_env = opt.num_env
    num_epoch = opt.num_epoch

    agent = PPOTorch(opt)

    param_generator = ParamGenerator(opt)

    record_rewards = []
    best_iteration = 0
    record_pg_loss = []
    record_vf_loss = []

    for iteration in tqdm(range(num_iteration)):
        params = param_generator.generate(num_env)
        envs = [gym.make('oscillator-v0', **params[i]) for i in range(num_env)]

        env_processor = EnvProcessor(envs)

        samples = {'tra_states': [], 'tra_actions': [], 'tra_values': [], 'tra_rewards': [], 'tra_neglogpacs': []}

        for sample_id in range(batch_size):
            observations = env_processor.get_obs()  # y_state
            states = torch.Tensor(observations)  # contrastive
            actions, values, neglogpacs = agent.predict(states)
            rewards = env_processor.step(actions)
            samples['tra_states'].append(states)
            samples['tra_actions'].append(actions)
            samples['tra_values'].append(values)
            samples['tra_rewards'].append(rewards)
            samples['tra_neglogpacs'].append(neglogpacs)

        observations = env_processor.get_obs()
        states = torch.Tensor(observations)
        values = agent.estimate_value(states)
        samples['tra_values'].append(values)

        env_processor.close()
        samples = samples_reshape(samples)  # dim*num* to num*dim*

        avg_reward = torch.mean(samples['tra_rewards'])
        record_rewards.append(avg_reward)
        if iteration == 0 or avg_reward > record_rewards[best_iteration]:
            best_iteration = iteration
            agent.save_networks()

        pg_loss, vf_loss = agent.update(samples)
        record_pg_loss.append(pg_loss.cpu().detach().numpy())
        record_vf_loss.append(vf_loss.cpu().detach().numpy())
        break
        #for epoch in range(num_epoch):
        #    agent.update(samples)

    plot_train_rewards(opt, record_rewards, best_iteration)
    plot_loss(opt, record_pg_loss, record_vf_loss, best_iteration)

    #agent.save_networks()


if __name__ == '__main__':
    train()
