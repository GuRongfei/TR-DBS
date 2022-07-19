import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


class Parameter:
    def __init__(self, name, num, lower_bound, upper_bound=0, mode="Stable"):
        self.name = name
        self.num = num
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.mode = mode

    def generate(self):
        if self.mode == "Stable":
            return [self.lower_bound for _ in range(self.num)]
        elif self.mode == 'Uniform':
            params = np.random.random(self.num)
            params = params * (self.upper_bound - self.lower_bound) + self.lower_bound
            return params


class ParamGenerator:
    def __init__(self, env_params):
        self.env_params = env_params
        self.parameters = []
        self.num = None

        self.initiate()

    def initiate(self):
        for env_param in self.env_params:
            self.parameters.append(Parameter(**env_param))
        self.num = self.parameters[0].num

    def generate(self):
        params = [{} for _ in range(self.num)]
        for parameter in self.parameters:
            param = parameter.generate()
            for i in range(self.num):
                params[i][parameter.name] = param[i]
        return params


class EnvProcessor:
    def __init__(self, envs):
        self.envs = envs

    def get_obs(self):
        obs = [np.array(env.y_state) for env in self.envs]
        return obs

    def reset(self):
        obs = [torch.Tensor(env.reset()) for env in self.envs]
        return torch.stack(obs)

    def step(self, actions):
        obs_, reward_, done_, info_ = [], [], [], []
        for env_id in range(len(self.envs)):
            env = self.envs[env_id]
            action = actions[env_id]
            obs, reward, done, info = env.step(action.tolist()[0])
            obs_.append(torch.Tensor(obs))
            reward_.append(torch.Tensor([reward]))
            done_.append(torch.Tensor([done]))
            info_.append(info)
        obs_ = torch.stack(obs_)
        reward_ = torch.stack(reward_)
        done_ = torch.stack(done_)
        return obs_, reward_, done_, info_

    def close(self):
        for env in self.envs:
            env.close()

    def get_x(self):
        x_vals = []
        for env in self.envs:
            x_vals.append(env.x_val)
        return x_vals

    def clip(self, action):
        action_space = self.envs[0].action_space
        action = torch.clamp(action, action_space.low[0], action_space.high[0])
        return action


def write_env_info(env_params, file_dir, mode):
    message = '---------- %5s_env_info  ------------\n' % mode
    keys = env_params[0].keys()
    for key in keys:
        message += "|%-15s" % key
    message += "\n"

    for env_param in env_params:
        for key in keys:
            message += "|%-15.3f" % env_param[key]
        message += "\n"
    message += '------------------End--------------------\n'
    file_name = file_dir + '/' + 'options.txt'
    with open(file_name, 'a') as f:
        f.write(message)
        f.write('\n')
