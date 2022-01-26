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
    def __init__(self, ar_range, fr_range, env_id='oscillator-v0', algo_id='ppo', test_timestep=5000, manual_define=None, tst_folder='baselines'):
        self.ar_range = ar_range
        self.fr_range = fr_range
        self.env_id = env_id
        self.algo_id = algo_id
        self.test_timestep = test_timestep
        self.manual_define = manual_define
        self.tst_folder = tst_folder

        self.train_no = []
        self.test_no = []

        self.env_params = []

        self.setup()

    def setup(self):
        cnt = 0
        for ar in self.ar_range:
            for fr in self.fr_range:
                self.env_params.append({'amplitude_rate': ar, 'frequency_rate': fr})
                self.train_no.append(cnt)
                self.test_no.append(cnt)
                cnt += 1
        self.env_params.append({'amplitude_rate': 2., 'frequency_rate': 2.})
        self.env_params.append({'amplitude_rate': 0.5, 'frequency_rate': 0.5})
        if self.manual_define:
            self.train_no = self.manual_define['train']
            self.test_no = self.manual_define['test']

    def cross_test(self, mode="zero-shot"):
        if mode == "zero-shot":
            file_name = 'zero.txt'
            finetune = False
        elif mode == "few-shot":
            file_name = 'few.txt'
            finetune = True
        else:
            return

        num_envs = len(self.env_params)
        save_paths = ['' for _ in range(num_envs)]
        avg_rwd = np.zeros((num_envs, num_envs), dtype=np.float32)
        spr_rate = np.zeros((num_envs, num_envs), dtype=np.float32)
        std_dev = np.zeros((num_envs, num_envs), dtype=np.float32)
        for train_env_no in self.train_no:
            train_env_para = self.env_params[train_env_no]
            executor = Executor(self.env_id, train_env_para, self.algo_id, {}, new_executor=False, tst_folder=self.tst_folder)

            for test_env_no in self.test_no:
                if test_env_no == train_env_no:
                    continue

                test_env_para = self.env_params[test_env_no]
                print("processing... model:%s/%s | test:%s/%s" % (train_env_para['amplitude_rate'],
                                                                  train_env_para['frequency_rate'],
                                                                  test_env_para['amplitude_rate'],
                                                                  test_env_para['frequency_rate']))
                executor.load(test_env_para, finetune)
                if save_paths[test_env_no] == '':
                    save_paths[test_env_no] = executor.save_path
                if finetune:
                    executor.finetune(10240)
                    print('finetuned')

                iter_num = 10
                rwds, sprs = [], []
                avg_rwd_, spr_rate_ = 0., 0.
                for _ in range(iter_num):
                    rwd, spr = executor.test_model(self.test_timestep, finetune)
                    rwds.append(rwd)
                    sprs.append(spr)
                    avg_rwd_ += rwd/5.
                    spr_rate_ += spr/5.
                std_dev_ = np.std(rwds)
                avg_rwd[test_env_no][train_env_no] = avg_rwd_
                std_dev[test_env_no][train_env_no] = std_dev_
                spr_rate[test_env_no][train_env_no] = spr_rate_

        if self.manual_define:
            for te in self.test_no:
                manual_path = save_paths[te]
                print("saving manual in %s" % manual_path)
                with open("%s/%s" % (manual_path, file_name), 'w') as f:
                    f.write("-------------manual---------------\n")
                    for tr in self.train_no:
                        train_env_para = self.env_params[tr]
                        f.write("----------------------------------\n")
                        f.write("|amplitude      | %-15s|\n" % train_env_para['amplitude_rate'])
                        f.write("|frequency      | %-15s|\n" % train_env_para['frequency_rate'])
                        f.write("|reward         | %-15s|\n" % avg_rwd[te][tr])
                        f.write("|std dev        | %-15s|\n" % std_dev[te][tr])
                        f.write("|spr rate       | %-15s|\n" % spr_rate[te][tr])

            for no1 in self.train_no:
                for no2 in self.test_no:
                    print("train: ", self.env_params[no1])
                    print("test: ", self.env_params[no2])
                    print("rwd: ", avg_rwd[no2][no1])
                    print("spr: ", spr_rate[no2][no1])
            return

        for no in range(num_envs):
            no_rwd = avg_rwd[no].reshape((1, -1))
            no_std = std_dev[no].reshape((1, -1))
            no_spr = spr_rate[no].reshape((1, -1))
            val = np.concatenate((no_rwd, no_std, no_spr), axis=0)
            np.savetxt('%s/%s' % (save_paths[no], file_name), val, fmt='%0.4f')

    def baseline_test(self):
        for env_no in self.test_no:
            env = self.env_params[env_no]
            print("processing env... ar:%s, fr:%s" % (str(env['amplitude_rate']), str(env['frequency_rate'])))
            executor = Executor(self.env_id, env, self.algo_id, {}, new_executor=False, tst_folder=self.tst_folder)
            executor.load(env)
            iter_num = 10
            rwds = []
            avr_rwd = 0.
            spr_rate = 0.
            for _ in range(iter_num):
                rwd, spr = executor.test_model(self.test_timestep)
                rwds.append(rwd)
                avr_rwd += rwd
                spr_rate += spr
            avr_rwd /= iter_num
            std_dev = float(np.std(rwds))
            spr_rate /= iter_num

            avr_rwd = round(avr_rwd, 3)
            std_dev = round(std_dev, 3)
            spr_rate = round(spr_rate, 3)

            save_path = executor.save_path

            print("saving baseline in %s" % save_path)
            with open("%s/baseline.txt" % save_path, 'w') as f:
                f.write("------------baseline--------------\n")
                f.write("|reward         | %-15s|\n" % avr_rwd)
                f.write("|std dev        | %-15s|\n" % std_dev)
                f.write("|spr rate       | %-15s|\n" % spr_rate)
