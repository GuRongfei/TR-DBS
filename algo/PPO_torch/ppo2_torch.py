import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

from options.base_options import BaseOptions
from utils.env_utils import ParamGenerator, EnvProcessor

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


class PPOTorch:
    def __init__(self, opt):
        self.opt = opt
        self.gamma = 0.99
        self.lambda_ = 0.95

        self.device = None
        self.save_path = None

        self.actor = None
        self.critic = None
        self.actor_optimizer = None
        self.critic_optimizer = None

        self.vf_loss = None
        self.vf_coef = 0.5
        self.pg_loss = None

        self.clip_range = 0.2

        self.initiate()

    def initiate(self):
        self.device = torch.device('cuda:{}'.format(self.opt.gpu_ids[0])) if self.opt.gpu_ids else torch.device('cpu')
        print(torch.cuda.is_available())
        self.save_path = self.opt.root_path + '/' + self.opt.package_name + '/'
        
        self.actor = MLP(self.opt.len_state, 1)
        self.critic = MLP(self.opt.len_state, 1)

        self.actor = self.actor.to(self.opt.gpu_ids[0])
        self.actor = torch.nn.DataParallel(self.actor, self.opt.gpu_ids)
        self.critic = self.critic.to(self.opt.gpu_ids[0])
        self.critic = torch.nn.DataParallel(self.critic, self.opt.gpu_ids)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.opt.policy_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.opt.value_lr)

    def predict(self, states):
        mean = self.actor(states)
        #std = torch.ones(mean.size())*0.1
        std = torch.Tensor([[0.1]])
        std = std.to(self.device)
        dist = MultivariateNormal(mean, std)
        actions = dist.sample()
        neglogpacs = -dist.log_prob(actions)
        #clipped action
        #return [[0] for state in states], [1 for state in states]#action/neglogpac
        return actions, self.estimate_value(states), neglogpacs

    def estimate_value(self, states):
        return self.critic(states)
        #return [0 for state in states]

    def get_neglogpac(self, states, actions):
        mean = self.actor(states)
        std = torch.Tensor([[0.1]])
        std = std.to(self.device)
        dist = MultivariateNormal(mean, std)
        neglogpacs = -dist.log_prob(actions)
        return neglogpacs

    def update(self, samples):
        tra_states = samples['tra_states'].to(self.device)#env*step*dim
        tra_actions = samples['tra_actions'].to(self.device)
        tra_values = samples['tra_values'].to(self.device)
        tra_rewards = samples['tra_rewards'].to(self.device)
        tra_neglogpacs = samples['tra_neglogpacs'].to(self.device)
        tra_advantages = torch.zeros(tra_rewards.shape).to(self.device)
        last_adv = torch.zeros(tra_rewards[:, 0].shape).to(self.device)
        for step in reversed(range(self.opt.batch_size)):
            delta = tra_rewards[:, step] + self.gamma * tra_values[:, step+1] - tra_values[:, step]
            tra_advantages[:, step] = last_adv = delta + self.gamma * self.lambda_ * last_adv
        tra_returns = tra_advantages + tra_values[:, :-1]

        vpred = self.estimate_value(tra_states)
        #vf_losses = np.square(vpred - tra_returns)
        #self.vf_loss = .5 * torch.mean(vf_losses)
        self.vf_loss = F.mse_loss(vpred, tra_returns)

        neglogpacs = self.get_neglogpac(tra_states, tra_actions)
        #self.entropy = tf.reduce_mean(train_model.proba_distribution.entropy())

        print("tra_neglogpacs: ", tra_neglogpacs)
        print("neglogpacs: ", neglogpacs)
        ratio = torch.exp(tra_neglogpacs - neglogpacs)
        print("tra_advantages: ", tra_advantages)
        print("ratio: ", ratio)
        tra_advantages = torch.reshape(tra_advantages, ((tra_advantages.shape[0], -1)))
        print("tra_advantages: ", tra_advantages)

        pg_losses = -tra_advantages * ratio
        pg_losses2 = -tra_advantages * torch.clip(ratio, 1.0 - self.clip_range, 1.0 +
                                                      self.clip_range)
        self.pg_loss = torch.mean(torch.maximum(pg_losses, pg_losses2))
        loss = self.pg_loss + self.vf_loss * self.vf_coef

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()
  
    def save_networks(self):
        print("saving models in: ", self.save_path)
        if len(self.opt.gpu_ids)> 0 and torch.cuda.is_available():
            torch.save(self.actor.module.cpu().state_dict(), self.save_path+'actor.pth')
            self.actor.cuda(self.opt.gpu_ids[0])
            torch.save(self.critic.module.cpu().state_dict(), self.save_path+'critic.pth')
            self.critic.cuda(self.opt.gpu_ids[0])
        else:
            torch.save(self.actor.cpu().state_dict(), self.save_path+'actor.pth')
            torch.save(self.critic.cpu().state_dict(), self.save_path+'critic.pth')

    def load_networks(self):
        print("loading models in: ", self.save_path)
        actor_dict = torch.load(self.save_path+'actor.pth', map_location=str(self.device))
        self.actor.load_state_dict(actor_dict, strict=True)
        critic_dict = torch.load(self.save_path+'critic.pth', map_location=str(self.device))
        self.critic.load_state_dict(critic_dict, strict=True)
        

class MLP(nn.Module):
    def __init__(self, input_size, output_size, act_fun=None):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.act_fun = act_fun

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, output_size)

    def forward(self, input):
        if isinstance(input, np.ndarray):
            input = torch.tensor(input, dtype=torch.float)

        output = F.tanh(self.fc1(input))
        output = F.tanh(self.fc2(output))
        output = F.tanh(self.fc3(output))
        output = self.output(output)
        if self.act_fun == 'tanh':
            output = F.tanh(output)
        return output
   
