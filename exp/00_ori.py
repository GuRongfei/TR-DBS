import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import gym
import gym_oscillator
import oscillator_cpp
from stable_baselines.common import set_global_seeds

from algo.PPO.ppo2_policy import MlpPolicy,MlpLnLstmPolicy,FeedForwardPolicy
from stable_baselines.common.vec_env import DummyVecEnv,SubprocVecEnv,VecNormalize, VecEnv
from algo.PPO.ppo2 import PPO2
from stable_baselines.common.vec_env import VecEnv

import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def make_env(env_id, rank, seed=0, ):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :param s_i: (bool) reward form, only one can be true
    """

    def _init():
        env = gym.make(env_id)
        print(env.reset().shape)
        return env

    set_global_seeds(seed)
    return _init


if __name__=='__main__':
    env_id = 'oscillator-v0'
    time_steps = int(10e6)
    #Number of cpus
    num_cpu = 8
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    model = PPO2(MlpPolicy, env, verbose=1)#,tensorboard_log="MLP/")
    model.learn(time_steps)
    os.mkdir('./result/executor/oscillator-v0/ppo/ori1')
    model.save('./result/executor/oscillator-v0/ppo/ori1/model.pkl')

    env = gym.make(env_id)

    rews_, obs_, acs_, states_x, states_y = [], [], [], [], []
    obs = env.reset()

    # Initial, non-suppresssion
    for i in range(25000):
        obs, rewards, dones, info = env.step([0])
        states_x.append(env.x_val)
        states_y.append(env.y_val)
        obs_.append(obs[0])
        acs_.append(0)
        rews_.append(rewards)

    # Suppression stage
    for i in range(25000):
        action, _states = model.predict(obs)

        obs, rewards, dones, info = env.step(action)

        states_x.append(env.x_val)
        states_y.append(env.y_val)
        obs_.append(obs[0])
        acs_.append(action)
        rews_.append(rewards)

    # Final relaxation
    for i in range(5000):
        obs, rewards, dones, info = env.step([0])
        states_x.append(env.x_val)
        states_y.append(env.y_val)
        obs_.append(obs[0])
        acs_.append(0)
        rews_.append(rewards)

    fig = plt.figure(figsize=(25, 10))
    ax = fig.add_subplot(111)
    ax.tick_params(labelsize=25)
    ax.plot(acs_, '-', c='lightcoral', label='Action')
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
    plt.savefig('./result/executor/oscillator-v0/ppo/ori1/SA.png')