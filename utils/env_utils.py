import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import torch

import gym
import gym_oscillator
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import DummyVecEnv,SubprocVecEnv,VecNormalize, VecEnv

from algo.PPO.ppo2 import PPO2
from algo.PPO.ppo2_policy import MlpPolicy

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


class Parameter:
    def __init__(self, name, lower_bound, upper_bound=0, random=True):
        self.name = name
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.random = random

    def generate(self, mode, num_param):
        if not self.random:
            return [self.lower_bound for _ in range(num_param)]
        if mode == 'uniform':
            params = np.random.random(num_param)
            params = params * (self.upper_bound - self.lower_bound) + self.lower_bound
            return params


class ParamGenerator:
    def __init__(self, opt):
        self.mode = opt.generator_mode
        self.opt = opt

        self.parameters = []

        self.initiate()

    def initiate(self):
        # initialize parameters
        random_param_ids = self.opt.random_params.split(',')
        if '0' in random_param_ids:
            amplitude_rate = Parameter('amplitude_rate', self.opt.range_ar_low, self.opt.range_ar_up)
            self.parameters.append(amplitude_rate)
        if '1' in random_param_ids:
            frequency_rate = Parameter('frequency_rate', self.opt.range_fr_low, self.opt.range_fr_up)
            self.parameters.append(frequency_rate)

        stable_param_ids = self.opt.stable_params.split(',')
        if '0' in stable_param_ids:
            assert '0' not in random_param_ids, 'randomization of amplitude_rate misused'
            amplitude_rate = Parameter('amplitude_rate', self.opt.amplitude_rate, random=False)
            self.parameters.append(amplitude_rate)
        if '1' in stable_param_ids:
            assert '1' not in random_param_ids, 'randomization of frequency_rate misused'
            frequency_rate = Parameter('frequency_rate', self.opt.frequency_rate, random=False)
            self.parameters.append(frequency_rate)
        if '2' in stable_param_ids:
            len_state = Parameter('len_state', self.opt.len_state, random=False)
            self.parameters.append(len_state)

    def generate(self, num_env):
        params = [{} for _ in range(num_env)]
        for parameter in self.parameters:
            param = parameter.generate(self.mode, num_env)
            for i in range(num_env):
                params[i][parameter.name] = param[i]
        return params


class EnvProcessor:
    def __init__(self, envs):
        self.envs = envs

    def get_obs(self):
        obs = [np.array(env.y_state) for env in self.envs]
        return obs

    def step(self, actions):
        rewards = []
        for env_id in range(len(self.envs)):
            env = self.envs[env_id]
            action = actions[env_id]
            _, reward, _, _ = env.step(action)
            rewards.append(torch.Tensor([reward]))
        rewards = torch.stack(rewards)        
        return rewards

    def close(self):
        for env in self.envs:
            env.close()

    def get_x(self):
        x_vals = []
        for env in self.envs:
            x_vals.append(env.x_val)
        return x_vals
