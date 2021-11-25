import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

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
    def __init__(self, env_id, env_para, algo_id, algo_para, new_executor=True, folder_num=-1, use_multi_env=False):
        self.env_id = env_id
        self.env_para = env_para
        self.algo_id = algo_id
        self.algo_para = algo_para
        self.new_executor = new_executor
        self.folder_num = folder_num
        self.use_multi_env = use_multi_env

        self.train_env = None
        self.test_env = None
        self.test_obs = None
        self.algo_model = None

        self.save_path = None

        self.train_timestep = None

        self.setup()

    def setup(self):
        self.setup_env()
        self.setup_algo_model()

        if self.new_executor:
            self.folder_num = sum([1 for _ in os.listdir('./result/executor/%s/%s/' % (self.env_id, self.algo_id))])
            self.save_path = './result/executor/%s/%s/%s' % (self.env_id, self.algo_id, str(self.folder_num))
            os.mkdir(self.save_path)
        else:
            self.save_path = './result/executor/%s/%s/%s' % (self.env_id, self.algo_id, str(self.folder_num))
            self.algo_model = self.algo_model.load('%s/model.pkl' % self.save_path)

    def setup_env(self):
        if self.env_id == 'oscillator-v0':
            if self.use_multi_env:
                self.train_env = SubprocVecEnv([make_env(self.env_id, i, **self.env_para) for i in range(8)])
            else:
                self.train_env = gym.make(self.env_id, **self.env_para)
            self.env_para['ep_length'] = 20000
            self.test_env = gym.make(self.env_id, **self.env_para)
        else:
            self.train_env = gym.make(self.env_id)
            self.test_env = gym.make(self.env_id)

        self.test_obs = self.test_env.reset()

    def setup_algo_model(self):
        if self.algo_id == 'ppo':
            self.algo_model = PPO2(MlpPolicy, self.train_env, **self.algo_para)
        else:
            pass

    def train_policy(self, train_timestep, save=False):
        self.algo_model.learn(train_timestep)
        self.train_timestep = train_timestep

        """fig = plt.figure(figsize=(25, 5))
        ax = fig.add_subplot(111)
        ax.plot(self.algo_model.episodes_reward, '-', label='State')
        ax.legend(loc=0)
        ax.grid()
        ax.set_xlabel("TimeStep(1024*)", fontsize=20)
        ax.set_ylabel("reward", fontsize=20)
        ax.set_title("Train Reward", fontsize=25)
        plt.savefig('%s/trainReward.png' % self.save_path)"""

        if save:
            self.algo_model.save('%s/model.pkl' % self.save_path)

    def test_model(self, test_timestep):
        if self.env_id == 'oscillator-v0':
            avr_rwd = self.test_osc(test_timestep)
        else:
            avr_rwd = 0
        if self.new_executor:
            self.savedata(round(avr_rwd, 3))

    def test_osc(self, test_timestep):
        test_obs, states_x, test_act, test_rwd = [], [], [], []
        for test_step in range(7000+test_timestep):
            if test_step < 5000 or test_step >= (5000+test_timestep):
                action = [0]
            else:
                action = self.model_pred()
            test_obs.append(self.test_obs)
            states_x.append(self.test_env.x_val)
            test_act.append(action)
            self.test_obs, rewards, dones, info = self.test_env.step(action)
            test_rwd.append(rewards)

        print(states_x)

        fig = plt.figure(figsize=(25, 10))
        ax = fig.add_subplot(111)
        ax.tick_params(labelsize=25)
        ax.plot(test_act, '-', c='lightcoral', label='Action')
        ax2 = ax.twinx()
        ax2.tick_params(labelsize=25)
        ax2.plot(states_x, '-r', c='steelblue', label='State ')
        ax.legend(bbox_to_anchor=(1, 1), fontsize=25)
        ax.grid()
        ax.set_xlabel("TimeStep", fontsize=45)
        ax.set_ylabel("Actions", fontsize=45)
        ax2.set_ylabel("States", fontsize=45)
        ax2.legend(bbox_to_anchor=(1, 0.9), fontsize=25)
        ax.set_title("State length", fontsize=55)
        plt.savefig('%s/SA.png' % self.save_path)

        if test_timestep == 0:
            return 0
        else:
            avr_rwd = sum(test_rwd[1000:-200]) / float(test_timestep)
            return avr_rwd

    def model_pred(self):
        if self.algo_id == 'ppo':
            action, _, _, _ = self.algo_model.step([self.test_obs])
        else:
            action = self.algo_model.step(self.test_obs)
        return action

    def savedata(self, reward):
        print("saving para in %s" % self.save_path)
        with open("%s/param.txt" % self. save_path, 'w+') as f:
            f.write("--------------info----------------\n")
            f.write('|environment    | %-15s|\n' % self.env_id)
            f.write('|algorithm      | %-15s|\n' % self.algo_id)
            f.write("-------------env para-------------\n")
            for k, v in self.env_para.items():
                f.write("|%-15s| %-15s|\n" % (k, v))
            f.write("------------algo para-------------\n")
            f.write("|train timesteps| %-15s|\n" % self.train_timestep)
            for k, v in self.algo_para.items():
                f.write("|%-15s| %-15s|\n" % (k, v))
            f.write("--------------rslt----------------\n")
            f.write("|reward         | %-15s|" % reward)
