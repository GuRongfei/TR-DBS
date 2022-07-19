import gym
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
#plt.rc('font', family='Times New Roman')

import gym_oscillator


class OneDAug:
    def __init__(self, magnitude=0., sampling=0., noise=0., timeshift=0, permutation=1):
        self.magnitude = magnitude
        self.sampling = sampling
        self.noise = noise
        self.timeshift = timeshift
        self.permutation = permutation

        self.obs = None
        self.aug_obs = None
        self.obs_shape = None

    def gen_aug(self, obs):
        self.obs = obs
        self.aug_obs = obs.clone()
        self.obs_shape = obs.shape[1]

        ma_ratio = (torch.rand(1)-0.5)*self.magnitude + 1
        self.magnitude_adj(ma_ratio)

        sample_rate = (torch.rand(1)-0.5)*self.sampling + 1
        if sample_rate > 1:
            self.upsampling(ratio=sample_rate)
        elif sample_rate < 1:
            self.downsampling(ratio=sample_rate)

        std = torch.var(obs, dim=1)
        std = torch.mean(std)*self.noise
        self.guassian_noise(ratio=std)

        shift = torch.randint(0, self.timeshift+1, (1,))[0]
        self.time_shift(deviation=shift)

        self.p_shuffle(self.permutation)

        #print("mag: ", self.magnitude)
        #print("sam: ", self.sampling)
        #print("noi: ", self.noise)
        #print("tim: ", self.timeshift)
        #print("per: ", self.permutation)

        return self.aug_obs

    def guassian_noise(self, ratio):
        noise = torch.randn(self.obs_shape)*ratio
        self.aug_obs += noise

    def magnitude_adj(self, ratio):
        self.aug_obs *= ratio

    def upsampling(self, ratio):
        upsample = torch.nn.Upsample(scale_factor=ratio,  mode='linear')
        self.aug_obs = upsample(torch.stack([self.aug_obs]))[0][:, -self.obs_shape:]

    def downsampling(self, ratio):
        upsample = torch.nn.Upsample(scale_factor=ratio,  mode='linear')
        self.aug_obs = upsample(torch.stack([self.aug_obs]))[0]
        self.aug_obs = torch.cat((self.aug_obs, self.aug_obs), dim=1)
        self.aug_obs = self.aug_obs[:, -self.obs_shape:]

    def time_shift(self, deviation):
        obs_1 = self.aug_obs[:, -deviation:]
        obs_2 = self.aug_obs[:, :-deviation]
        self.aug_obs = torch.cat((obs_1, obs_2), dim=1)

    def p_shuffle(self, piece):
        assert piece > 0, "invalid number of pieces for permutation"
        order = [i for i in range(piece)]
        p_len = self.obs_shape // piece
        end = [(i+1)*p_len for i in range(piece)]
        end[-1] = self.obs_shape

        obs_pieces = []
        for i in range(piece):
            obs_pieces.append(self.aug_obs[:, i*p_len:end[i]])

        while order == [0, 1, 2]:
            random.shuffle(order)

        self.aug_obs = obs_pieces[order[0]]
        for i in range(1, piece):
            self.aug_obs = torch.cat((self.aug_obs, obs_pieces[order[i]]), dim=1)


if __name__ == "__main__":
    env = gym.make('oscillator-v0', len_state=250)
    #obs = torch.stack([torch.Tensor(env.reset())])
    env.reset()
    obs = torch.stack([torch.Tensor(env.x_state)])

    method = "Downsampling"
    onedaug = OneDAug(sampling=0.5)
    new_obs = onedaug.gen_aug(obs)
    print(obs[:, :10])
    print(new_obs[:, :10])


    fig = plt.figure(figsize=(20, 7))
    ax = fig.add_subplot(111)
    ax.tick_params(labelsize=50, pad=10)
    ax.plot(obs[0], '-', linewidth=7, c='steelblue', label='Obs    ', alpha=0.7)
    ax.plot(new_obs[0], '-', linewidth=7, c='orange', label='New obs', alpha=0.7)
    ax.grid()
    ax.set_xlabel("Time slice", fontsize=55, labelpad=20)
    ax.set_ylabel("Amplitude", fontsize=55, labelpad=20)
    ax.set_title(method, fontsize=55, pad=20)

    plt.tight_layout()
    plt.savefig('./data_augmentation/%s.png' % method)

