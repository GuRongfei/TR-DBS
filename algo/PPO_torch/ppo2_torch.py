import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

from options.base_options import TrainOptions, TestOptions
from utils.env_utils import ParamGenerator, EnvProcessor

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


class PPOTorch:
    def __init__(self, opt, train_mode=True):
        self.opt = opt
        self.gamma = 0.99
        self.lambda_ = 0.95

        self.device = None
        self.train_path = None
        self.test_path = None

        self.actor = None
        self.critic = None
        self.actor_optimizer = None
        self.critic_optimizer = None

        self.vf_loss = None
        self.vf_coef = 0.5
        self.pg_loss = None

        self.clip_range = 0.2

        self.initiate(train_mode)

    def initiate(self, train_mode):
        self.device = torch.device('cuda:{}'.format(self.opt.gpu_ids[0])) if self.opt.gpu_ids else torch.device('cpu')
        print(torch.cuda.is_available())
        self.train_path = self.opt.train_root_path + '/' + self.opt.train_package_name + '/'
        
        self.actor = MLP(self.opt.len_state, 1)
        self.critic = MLP(self.opt.len_state, 1)
        if not train_mode:
            self.load_networks()

        self.actor = self.actor.to(self.opt.gpu_ids[0])
        self.actor = torch.nn.DataParallel(self.actor, self.opt.gpu_ids)
        self.critic = self.critic.to(self.opt.gpu_ids[0])
        self.critic = torch.nn.DataParallel(self.critic, self.opt.gpu_ids)

        if train_mode:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.opt.policy_lr)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.opt.value_lr)

    def predict(self, states):
        mean = self.actor(states)
        #std = torch.ones(mean.size())*0.1
        std = torch.Tensor([[0.1]])
        std = std.to(self.device)
        dist = MultivariateNormal(mean, std)
        actions = dist.sample()
        actions = torch.clip(actions, -1., 1.)
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

        #tra_states, tra_returns, tra_actions, tra_neglogpacs, tra_advantages
        num_samples = self.opt.num_env * self.opt.batch_size
        tra_states = torch.reshape(tra_states, (num_samples, -1))
        tra_returns = torch.reshape(tra_returns, (num_samples, -1))
        print('before_action: ', tra_actions)
        tra_actions = torch.reshape(tra_actions, (num_samples, -1))
        print('after_action: ', tra_actions)
        print('before: ', tra_neglogpacs)
        tra_neglogpacs = torch.reshape(tra_neglogpacs, (num_samples, -1))
        print('after: ', tra_neglogpacs)
        tra_advantages = torch.reshape(tra_advantages, (num_samples, -1))

        record_pg_loss, record_vf_loss = [], []
        self.pg_loss, self.vf_loss = self.train(tra_states, tra_returns, tra_actions, tra_neglogpacs, tra_advantages)
        record_pg_loss.append(self.pg_loss)
        record_vf_loss.append(self.vf_loss)

        """index = np.arange(num_samples)
        for epoch_num in range(self.opt.num_epoch):
            np.random.shuffle(index)
            for start in range(0, num_samples, self.opt.batch_size):
                end = start + self.opt.batch_size
                batch_index = index[start:end]
                batch_states = tra_states[batch_index]
                batch_returns = tra_returns[batch_index]
                batch_actions = tra_actions[batch_index]
                batch_neglogpacs = tra_neglogpacs[batch_index]
                batch_advantages = tra_advantages[batch_index]
                self.pg_loss, self.vf_loss = self.train(batch_states, batch_returns, batch_actions,
                                                        batch_neglogpacs, batch_advantages)
                record_pg_loss.append(self.pg_loss)
                record_vf_loss.append(self.vf_loss)
                break
            break"""
        return torch.mean(torch.stack(record_pg_loss)), torch.mean(torch.stack(record_vf_loss))

    def train(self, batch_states, batch_returns, batch_actions, batch_neglogpacs, batch_advantages):
        vpred = self.estimate_value(batch_states)
        vf_loss = F.mse_loss(vpred, batch_returns)

        neglogpacs = self.get_neglogpac(batch_states, batch_actions)
        # self.entropy = tf.reduce_mean(train_model.proba_distribution.entropy())

        ratio = torch.exp(batch_neglogpacs - neglogpacs)
        batch_advantages = torch.reshape(batch_advantages, ((batch_advantages.shape[0], -1)))

        pg_losses = -batch_advantages * ratio
        pg_losses2 = -batch_advantages * torch.clip(ratio, 1.0 - self.clip_range, 1.0 +
                                                    self.clip_range)
        pg_loss = torch.mean(torch.maximum(pg_losses, pg_losses2))
        loss = pg_loss + vf_loss * self.vf_coef

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        #loss1 = loss.detach_().requires_grad_(True)
        #loss1.backward()
        loss.backward(retain_graph=True)
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        return pg_loss, vf_loss

    def save_networks(self):
        print("saving models in: ", self.train_path)
        if len(self.opt.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(self.actor.module.cpu().state_dict(), self.train_path+'actor.pth')
            self.actor.cuda(self.opt.gpu_ids[0])
            torch.save(self.critic.module.cpu().state_dict(), self.train_path+'critic.pth')
            self.critic.cuda(self.opt.gpu_ids[0])
        else:
            torch.save(self.actor.cpu().state_dict(), self.train_path+'actor.pth')
            torch.save(self.critic.cpu().state_dict(), self.train_path+'critic.pth')

    def load_networks(self):
        print("loading models from: ", self.train_path)
        actor_dict = torch.load(self.train_path+'actor.pth', map_location=str(self.device))
        self.actor.load_state_dict(actor_dict, strict=True)

        critic_dict = torch.load(self.train_path+'critic.pth', map_location=str(self.device))
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
   
