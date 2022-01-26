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


def make_env(env_id, rank,seed=0, **env_para ):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :param s_i: (bool) reward form, only one can be true
    """

    def _init():
        env = gym.make(env_id, **env_para)
        #print(env.reset().shape)
        return env

    set_global_seeds(seed)
    return _init


class Executor:
    def __init__(self, env_id, env_para, algo_id, algo_para, new_executor=True, use_multi_env=False, randomization=None,
                 tst_folder='baselines'):
        self.env_id = env_id
        self.env_para = env_para
        self.algo_id = algo_id
        self.algo_para = algo_para
        self.new_executor = new_executor
        self.use_multi_env = use_multi_env
        self.randomization = randomization
        self.tst_folder = tst_folder

        self.folder_num = None
        self.train_env = None
        self.test_env = None
        self.test_obs = None
        self.algo_model = None
        self.finetune_model = None

        self.modepath = None
        self.save_path = None

        self.train_timestep = None

        if self.new_executor:
            self.setup()

    #setup
    def setup(self):
        self.setup_env()
        self.setup_algo_model()

        root = 'executor'
        if self.randomization:
            root = 'dr_executor'

        self.folder_num = sum([1 for _ in os.listdir('./result/%s/%s/%s/' % (root, self.env_id, self.algo_id))])
        if self.folder_num < 10:
            folder_name = '0' + str(self.folder_num)
        else:
            folder_name = str(self.folder_num)
        self.save_path = './result/%s/%s/%s/%s' % (root, self.env_id, self.algo_id, folder_name)
        os.mkdir(self.save_path)

    def setup_env(self):
        if self.env_id == 'oscillator-v0':
            if self.use_multi_env:
                self.train_env = SubprocVecEnv([make_env(self.env_id, i, **self.env_para) for i in range(8)])
            elif self.randomization:
                env_num = len(self.randomization)
                train_env_list = []
                for i in range(env_num):
                    tmp_env_para = self.randomization[i]
                    tmp_env = make_env(self.env_id, i, **tmp_env_para)
                    train_env_list.append(tmp_env)
                self.train_env = SubprocVecEnv(train_env_list)
            else:
                self.train_env = gym.make(self.env_id, **self.env_para)

            if 'ep_length' in self.env_para.keys():
                tmp = self.env_para['ep_length']
            else:
                tmp = 10000
            self.env_para['ep_length'] = 50000
            self.test_env = gym.make(self.env_id, **self.env_para)
            self.env_para['ep_length'] = tmp
        else:
            self.train_env = gym.make(self.env_id)
            self.test_env = gym.make(self.env_id)

        self.test_obs = self.test_env.reset()

    def setup_algo_model(self, finetune=False):
        if finetune:
            self.finetune_model = PPO2(MlpPolicy, self.finetune_env, **self.algo_para)
            self.finetune_model = self.finetune_model.load('%s/model.pkl' % self.model_path)
            self.finetune_model.set_env(self.finetune_env)
        elif self.algo_id == 'ppo':
            if not self.new_executor:
                self.algo_model = PPO2(MlpPolicy, self.test_env, **self.algo_para)
                self.algo_model = self.algo_model.load('%s/model.pkl' % self.model_path)
            else:
                self.algo_model = PPO2(MlpPolicy, self.train_env, **self.algo_para)
        else:
            pass

    #load
    def load(self, test_env_para, finetune=False):
        val_to_str = ['09', '095', '1', '105', '11']
        ar_train = self.env_para['amplitude_rate']
        fr_train = self.env_para['frequency_rate']
        if ar_train == 0.5:
            ar = '05'
        elif ar_train == 2.:
            ar = '2'
        else:
            ar = val_to_str[int(0.5+20*(ar_train-0.9))]
        if fr_train == 0.5:
            fr = '05'
        elif fr_train == 2.:
            fr = '2'
        else:
            fr = val_to_str[int(0.5+20*(fr_train-0.9))]
        ar_test = test_env_para['amplitude_rate']
        fr_test = test_env_para['frequency_rate']
        if ar_test == 0.5:
            ar_tst = '05'
        elif ar_test == 2.:
            ar_tst = '2'
        else:
            ar_tst = val_to_str[int(0.5+20*(ar_test-0.9))]
        if fr_test == 0.5:
            fr_tst = '05'
        elif fr_test == 2.:
            fr_tst = '2'
        else:
            fr_tst = val_to_str[int(0.5+20*(fr_test-0.9))]

        self.finetune_env = SubprocVecEnv([make_env(self.env_id, i, **test_env_para) for i in range(8)])
        test_env_para['ep_length'] = 50000
        self.test_env = gym.make(self.env_id, **test_env_para)
        self.test_obs = self.test_env.reset()

        self.model_path = './result/tr/%s/%s-%s' % (self.tst_folder, ar, fr)
        self.save_path = './result/tr/%s/%s-%s' % (self.tst_folder, ar_tst, fr_tst)
        self.setup_algo_model(finetune)
        #self.algo_model = self.algo_model.load('%s/model.pkl' % self.model_path)

    # train
    def finetune(self, finetune_step):
        self.finetune_model.learn(finetune_step)

    def train_policy(self, train_timestep, save=False):
        self.algo_model.learn(train_timestep)
        self.train_timestep = train_timestep

        rwds = self.algo_model.reward_log
        new_rwds = []
        for i in range(0, len(rwds), 100):
            if i+100 < len(rwds):
                new_rwds.append(np.mean(rwds[i:i+100]))

        fig = plt.figure(figsize=(25, 5))
        ax = fig.add_subplot(111)
        ax.plot(new_rwds, '-', label='State')
        ax.legend(loc=0)
        ax.grid()
        ax.set_xlabel("TimeStep(1024*100)", fontsize=20)
        ax.set_ylabel("reward", fontsize=20)
        ax.set_title("Train Reward", fontsize=25)
        plt.savefig('%s/trainReward.png' % self.save_path)

        if save:
            self.algo_model.save('%s/model.pkl' % self.save_path)

    #test
    def test_model(self, test_timestep, finetune=False):
        if self.env_id == 'oscillator-v0':
            avr_rwd, spr_rate = self.test_osc(test_timestep, finetune)
        else:
            avr_rwd, spr_rate = 0, 0
        avr_rwd = round(avr_rwd, 3)
        spr_rate = round(spr_rate, 3)
        if self.new_executor:
            self.savedata(avr_rwd, spr_rate)
        return avr_rwd, spr_rate

    def test_osc(self, test_timestep, finetune=False):
        test_obs, states_x, test_act, test_rwd = [], [], [], []
        for test_step in range(7000+test_timestep):
            if test_step < 5000 or test_step >= (5000+test_timestep):
                action = [0]
            else:
                action = self.model_pred(finetune)
            test_obs.append(self.test_obs)
            states_x.append(self.test_env.x_val)
            test_act.append(action)
            self.test_obs, rewards, dones, info = self.test_env.step(action)
            test_rwd.append(rewards)

        if self.new_executor:
            fig = plt.figure(figsize=(25, 10))
            ax = fig.add_subplot(111)
            ax.tick_params(labelsize=30, pad=10)
            ax2 = ax.twinx()
            ax2.tick_params(labelsize=30, pad=10)
            ax2.plot(test_act, '-', linewidth=3, c='lightcoral', label='Action')
            ax.plot(states_x, '-', linewidth=3, c='steelblue', label='State ')
            ax.legend(bbox_to_anchor=(0.88, 1.12), fontsize=25)
            ax.grid()
            ax.set_xlabel("TimeStep", fontsize=45, labelpad=20)
            ax.set_ylabel("States", fontsize=45, labelpad=20)
            ax2.set_ylabel("Actions", fontsize=45, labelpad=20)
            ax2.legend(bbox_to_anchor=(1, 1.12), fontsize=25)
            ax.set_title("State & Action", fontsize=55, pad=20)
            plt.tight_layout()
            plt.savefig('%s/SA.png' % self.save_path)

        if test_timestep == 0:
            return 0
        else:
            avr_rwd = sum(test_rwd[5000:-2000]) / float(test_timestep)
            spr_rate = np.std(states_x[0:5000]) / np.std(states_x[5000:-2000])
            return avr_rwd, spr_rate

    def model_pred(self, finetune=False):
        if self.algo_id == 'ppo':
            if finetune:
                action, _, _, _ = self.finetune_model.step([self.test_obs])
            else:
                action, _, _, _ = self.algo_model.step([self.test_obs])
        else:
            if finetune:
                action = self.finetune_model.step(self.test_obs)
            else:
                action = self.algo_model.step(self.test_obs)
        return action

    #record
    def savedata(self, reward, spr_rate):
        print("saving para in %s" % self.save_path)
        with open("%s/param.txt" % self. save_path, 'w+') as f:
            f.write("##################################\n")
            f.write("--------------info----------------\n")
            f.write('|environment    | %-15s|\n' % self.env_id)
            f.write('|algorithm      | %-15s|\n' % self.algo_id)
            if self.randomization:
                f.write("##################################\n")
                f.write("----------randomization-----------\n")
                for i in range(len(self.randomization)):
                    f.write("-------------env %s----------------\n" % str(i))
                    for k, v in self.randomization[i].items():
                        f.write("|%-15s| %-15s|\n" % (k, v))
            f.write("##################################\n")
            f.write("-------------env para-------------\n")
            for k, v in self.env_para.items():
                f.write("|%-15s| %-15s|\n" % (k, v))
            f.write("##################################\n")
            f.write("------------algo para-------------\n")
            f.write("|train timesteps| %-15s|\n" % self.train_timestep)
            for k, v in self.algo_para.items():
                f.write("|%-15s| %-15s|\n" % (k, v))
            f.write("##################################\n")
            f.write("--------------rslt----------------\n")
            f.write("|reward         | %-15s|\n" % reward)
            f.write("|spr rate       | %-15s|" % spr_rate)
