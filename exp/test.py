import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import torch

import gym
import gym_oscillator

from algo.PPO_torch.ppo2_torch import PPOTorch

from options.base_options import TrainOptions, TestOptions
from utils.env_utils import ParamGenerator, EnvProcessor
from utils.normal_utils import samples_reshape

from utils.plot_utils import plot_train_rewards, plot_test_result
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def test():
    opt = TestOptions().parse()

    param_generator = ParamGenerator(opt)
    test_param = param_generator.generate(1)[0]
    test_env = gym.make('oscillator-v0', **test_param)
    env_processor = EnvProcessor([test_env])

    agent = PPOTorch(opt, False)

    actions = []
    states_x = []

    for step in range(opt.init_timestep+opt.stimulation_timestep+opt.rest_timestep):
        observation = env_processor.get_obs()
        state = torch.Tensor(observation)
        if step < opt.init_timestep or step >= (opt.init_timestep+opt.stimulation_timestep):
            action = [[0]]
            _ = env_processor.step(action)
        else:
            action, _, _ = agent.predict(state)
            _ = env_processor.step(action)
            action = action.cpu()
        state_x = env_processor.get_x()
        actions.append(action[0][0])
        states_x.append(state_x[0])
        #print('state_x: ', state_x)
    plot_test_result(opt, states_x, actions)


if __name__ == '__main__':
    test()
