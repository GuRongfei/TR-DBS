import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import gym
import gym_oscillator
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import DummyVecEnv,SubprocVecEnv,VecNormalize, VecEnv


from utils.execute import Executor

from algo.PPO.ppo2 import PPO2
from algo.PPO.ppo2_policy import MlpPolicy

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


class Validator:
    def __init__(self, ar_range, fr_range, env_id='oscillator-v0', algo_id='ppo', test_timestep=5000):
        self.ar_range = ar_range
        self.fr_range = fr_range
        self.env_id = env_id
        self.algo_id = algo_id
        self.test_timestep = test_timestep

        self.env_params = []

        self.setup()

    def setup(self):
        for ar in self.ar_range:
            for fr in self.fr_range:
                self.env_params.append({'amplitude_rate': ar, 'frequency_rate': fr})

    def cross_test(self, mode="zero-shot"):
        if mode == "zero-shot":
            file_name = 'zero1.txt'
            finetune = False
        elif mode == "few-shot":
            file_name = 'few.txt'
            finetune = True
        else:
            return
        save_paths = []

        num_envs = len(self.env_params)
        avg_rwd = np.zeros((num_envs, num_envs), dtype=np.float32)
        spr_rate = np.zeros((num_envs, num_envs), dtype=np.float32)
        for train_env_no in range(num_envs):
            train_env_para = self.env_params[train_env_no]
            executor = Executor(self.env_id, train_env_para, self.algo_id, {}, new_executor=False)

            save_path_load = False
            for test_env_no in range(num_envs):
                if test_env_no == train_env_no:
                    continue

                test_env_para = self.env_params[test_env_no]
                print("processing... model:%s/%s | test:%s/%s" % (train_env_para['amplitude_rate'],
                                                                  train_env_para['frequency_rate'],
                                                                  test_env_para['amplitude_rate'],
                                                                  test_env_para['frequency_rate']))
                executor.load(test_env_para, finetune)
                if not save_path_load:
                    save_paths.append(executor.model_path)
                    save_path_load = True
                if finetune:
                    executor.finetune(2048)

                avg_rwd_, spr_rate_ = 0., 0.
                for _ in range(2):
                    rwd, spr = executor.test_model(self.test_timestep, finetune)
                    avg_rwd_ += rwd/2.
                    spr_rate_ += spr/2.
                avg_rwd[test_env_no][train_env_no] = avg_rwd_
                spr_rate[test_env_no][train_env_no] = spr_rate_

        for no in range(num_envs):
            no_rwd = avg_rwd[no].reshape((1, -1))
            no_spr = spr_rate[no].reshape((1, -1))
            val = np.concatenate((no_rwd, no_spr), axis=0)

            np.savetxt('%s/%s' % (save_paths[no], file_name), val, fmt='%0.4f')

    def baseline_test(self):
        for env in self.env_params:
            print("processing env... ar:%s, fr:%s" % (str(env['amplitude_rate']), str(env['frequency_rate'])))
            executor = Executor(self.env_id, env, self.algo_id, {}, new_executor=False)
            executor.load(env)
            iter_num = 10
            avr_rwd = 0.
            spr_rate = 0.
            for _ in range(iter_num):
                rwd, spr = executor.test_model(self.test_timestep)
                avr_rwd += rwd
                spr_rate += spr
            avr_rwd /= iter_num
            spr_rate /= iter_num

            avr_rwd = round(avr_rwd, 3)
            spr_rate = round(spr_rate, 3)

            save_path = executor.save_path

            print("saving baseline in %s" % save_path)
            with open("%s/baseline.txt" % save_path, 'w') as f:
                f.write("------------baseline--------------\n")
                f.write("|reward         | %-15s|\n" % avr_rwd)
                f.write("|spr rate       | %-15s|\n" % spr_rate)