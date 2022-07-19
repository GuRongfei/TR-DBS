import time
from collections import deque

import gym
import gym_oscillator
import numpy as np
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate
from utils import env_utils, plot_utils
from data_augmentation.oned_aug import OneDAug
from encoder.cntra_model import CU


def test(id, test_env, agent, encoder, file_dir, device):
    observation = test_env.reset()
    if encoder:
        observation = encoder.encode_obs(torch.Tensor(observation))
    recurrent_hidden_states = torch.zeros(agent.recurrent_hidden_state_size, device=device)
    masks = torch.zeros(1, device=device)

    actions, states_x = [], []
    action_cumu = 0
    for step in range(2000 + 5000 + 500):
        if step < 2000 or step >= (5000 + 2000):
            action = 0
            observation, reward, done, info = test_env.step(action)
        else:
            value, action, action_log_prob, recurrent_hidden_states = agent.act(torch.Tensor(observation),
                                                                                       recurrent_hidden_states, masks)
            #value, action, action_log_prob, recurrent_hidden_states = agent.act(torch.Tensor(observation),
                                                                                       #recurrent_hidden_states, masks)
            #action = env_processor.clip(action)
            action = action.numpy().tolist()[0][0][0][0]
            observation, reward, done, info = test_env.step(action)
        if encoder:
            observation = encoder.encode_obs(torch.Tensor(observation))
        #print("tst_rwd: ", reward)
        actions.append(action)
        action_cumu += abs(action)
        states_x.append(test_env.x_val)
    test_env.close()

    action_cumu /= 5000
    suppression_r = torch.std(torch.Tensor(states_x[500:2000])) / torch.std(torch.Tensor(states_x[2500:5000]))
    #dict = {'action_cumu': action_cumu, 'suppression_r': suppression_r}
    #np.save("%s/test/%d.npy" % (file_dir, id), dict)

    #load_dict = np.load("%s/test/%d.npy" % (file_dir, id), allow_pickle=True).item()
    #print(load_dict['action_cumu'])
    #print(load_dict['suppression_r'])

    #if suppression_r > 4:
    plot_utils.plot_overlap(states_x, actions, "state_action", file_dir)
    return action_cumu, suppression_r


