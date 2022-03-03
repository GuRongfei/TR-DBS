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

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def samples_reshape(samples):
    for key, values in samples.items():
        values = torch.stack(values)
        #if samples[key].ndim == 2:
        #    samples[key] = samples[key].T
        #else:
        #    samples[key] = samples[key].transpose((1, 0, 2))
        samples[key] = values.transpose(0, 1)
        #print(key, ' : ', samples[key][0])
    return samples

