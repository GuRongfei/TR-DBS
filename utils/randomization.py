import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np

import gym
import gym_oscillator
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import DummyVecEnv,SubprocVecEnv,VecNormalize, VecEnv

from algo.PPO.ppo2 import PPO2
from algo.PPO.ppo2_policy import MlpPolicy

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from utils.execute import Executor


class Randomizer:
    def __init__(self, ar_range, fr_range):
        self.ar_range = ar_range
        self.fr_range = fr_range

        self.env_params = []

        self.setup()

    def setup(self):
        for ar in self.ar_range:
            for fr in self.fr_range:
                self.env_params.append({'amplitude_rate': ar, 'frequency_rate': fr})

    def domain_randomization_set(self, tar_no):
        dr_set = []
        env_num = len(self.env_params)
        accident_cnt = 0
        while len(dr_set) < 5:
            i = np.random.randint(env_num)
            if (i != tar_no) and (not i in dr_set):
                dr_set.append(i)
        for i in range(5):
            dr_set[i] = self.env_params[dr_set[i]]
            accident_cnt += 1
            assert accident_cnt < 15, "loop broken accidentally"
        return dr_set

    def random_execute(self, env_id, algo_id, algo_para, train_timestep, test_timestep):
        for tar_no in range(len(self.env_params)):
            dr_set = self.domain_randomization_set(tar_no)
            executor = Executor(env_id, self.env_params[tar_no], algo_id, algo_para, randomization=dr_set)
            executor.train_policy(train_timestep)
            executor.test_model(test_timestep)
