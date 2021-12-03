import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from scipy import stats

import gym
import gym_oscillator
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import DummyVecEnv,SubprocVecEnv,VecNormalize, VecEnv

from algo.PPO.ppo2 import PPO2
from algo.PPO.ppo2_policy import MlpPolicy

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


class FileReader:
    def __init__(self, ar_range, fr_range):
        self.ar_range = ar_range
        self.fr_range = fr_range

        self.env_params = []
        self.packages = []

        self.setup()

    def setup(self):
        val_to_str = ['09', '095', '1', '105', '11']
        for ar in self.ar_range:
            for fr in self.fr_range:
                self.env_params.append({'amplitude_rate': ar, 'frequency_rate': fr})
                p1 = val_to_str[int(0.5 + 20 * (ar - 0.9))]
                p2 = val_to_str[int(0.5 + 20 * (fr - 0.9))]
                package_name = p1 + '-' + p2
                self.packages.append(package_name)

    def read_baseline(self, package_name):
        with open("./result/direct_tr/baselines/%s/baseline.txt" % package_name, 'r') as f:
            f.readline()
            rwd_line = f.readline()
            spr_line = f.readline()
            rwd = rwd_line[18:-2].strip()
            spr = spr_line[18:-2].strip()
        return rwd, spr
            #f.write("|reward         | %-15s|\n" % avr_rwd)
            #f.write("|spr rate       | %-15s|\n" % spr_rate)

    def read_cross(self, package_name, mode='zero-shot'):
        if mode == 'zero-shot':
            f = np.loadtxt('./result/direct_tr/baselines/%s/zero.txt' % package_name)
        else:
            f = np.loadtxt('./result/direct_tr/baselines/%s/few.txt' % package_name)
        avr_rwd = np.sum(f[0])/(len(f[0])-1.)
        avr_rwd = round(avr_rwd, 3)
        avr_spr = np.sum(f[1])/(len(f[1])-1.)
        avr_spr = round(avr_spr, 3)

        max_rwd = -1.
        max_rwd_spr = 0
        max_spr = 0.1
        max_spr_rwd = 0
        for i in range(len(f[0])):
            if f[0][i] != 0 and f[0][i] > max_rwd:
                max_rwd = f[0][i]
                max_rwd_spr = f[1][i]
            if f[1][i] != 0 and f[1][i] > max_spr:
                max_spr = f[1][i]
                max_spr_rwd = f[0][i]

        return avr_rwd, avr_spr, max_rwd, max_rwd_spr, max_spr, max_spr_rwd

    def rearrange(self):
        baseline = []
        baseline_spr = []
        zero_shot = []
        few_shot = []
        index = []
        with open('./result/direct_tr/baselines/result.txt', 'w') as f:
            for package_name in self.packages:
                rwd, spr = self.read_baseline(package_name)
                zero = self.read_cross(package_name)
                few = self.read_cross(package_name, 'few-shot')
                baseline.append(float(rwd))
                baseline_spr.append(float(spr))
                zero_shot.append(float(zero[0]))
                few_shot.append(float(few[0]))
                index.append(package_name)
                #avr_rwd, avr_spr, max_rwd, max_rwd_spr, max_spr, max_spr_rwd = self.read_zero(package_name)
                #avr_rwd, avr_spr, max_rwd, max_rwd_spr, max_spr, max_spr_rwd = self.read_zero(package_name, 'few-shot')

                f.write("||env para       | %-26s||\n" % package_name)
                f.write("||baseline       | rwd: %-7s| spr: %-7s||\n" % (rwd, spr))
                f.write("||zero avr       | rwd: %-7s| spr: %-7s||\n" % (zero[0], zero[1]))
                f.write("||zero max rwd   | rwd: %-7s| spr: %-7s||\n" % (zero[2], zero[3]))
                f.write("||zero max spr   | rwd: %-7s| spr: %-7s||\n" % (zero[5], zero[4]))
                f.write("||few  avr       | rwd: %-7s| spr: %-7s||\n" % (few[0], few[1]))
                f.write("||few  max rwd   | rwd: %-7s| spr: %-7s||\n" % (few[2], few[3]))
                f.write("||few  max spr   | rwd: %-7s| spr: %-7s||\n" % (few[5], few[4]))
                f.write('\n\n')

        fig = plt.figure(figsize=(25, 10))
        ax = fig.add_subplot(111)
        #ax.tick_params(labelsize=30, pad=10)
        ax.plot(index, baseline, '-', linewidth=3, c='steelblue', label='baseline')
        #ax.set_xticklabels(index, rotation=30)
        ax.plot(zero_shot, '-', linewidth=3, c='lightcoral', label='zero shot')
        ax.plot(few_shot, '-', linewidth=3, c='olivedrab', label='few_shot')
        ax.legend(bbox_to_anchor=(0.84, 1.), fontsize=25)
        ax.grid()
        #ax.set_xlabel("TimeStep", fontsize=45, labelpad=20)
        ax.set_ylabel("Reward", fontsize=45, labelpad=20)
        ax.set_title("Rewards", fontsize=55, pad=20)
        plt.tight_layout()
        plt.savefig('./result/direct_tr/baselines/result.png')

        b = np.asarray(baseline)
        bs = np.asarray(baseline_spr)
        z = np.asarray(zero_shot)
        f = np.asarray(few_shot)
        r1 = stats.levene(b, z)
        r2 = stats.levene(b, f)
        print("r1: ", r1)
        print("r2: ", r2)
        ind_1 = stats.ttest_ind(b, z)
        ind_2 = stats.ttest_ind(b, f)
        print("ind_1: ", ind_1)
        print("ind_2: ", ind_2)

        coef = np.corrcoef(b, bs)
        print("coef: ", coef)
