import time
from collections import deque

import gym
import gym_oscillator
import numpy as np
import torch
from tqdm import tqdm
import os

import matplotlib.pyplot as plt

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate
from utils import env_utils, plot_utils, exp_utils
from data_augmentation.oned_aug import OneDAug
from encoder.cntra_model import CU


def train_agent(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0" if args.cuda else "cpu")

    file_dir = args.save_dir + '/' + args.file_name

    # initialize training environment
    env_param_generator = env_utils.ParamGenerator(args.env_params)
    env_params = env_param_generator.generate()
    print(env_params)
    env_utils.write_env_info(env_params, file_dir, "train")
    envs = [gym.make(args.env_name, **env_param) for env_param in env_params]
    env_processor = env_utils.EnvProcessor(envs)

    if args.use_cl:
        obs_dim = (args.shallow_dim,)
    else:
        obs_dim = envs[0].observation_space.shape

    # initialize agent
    actor_critic = Policy(obs_dim, envs[0].action_space,
                          base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)

    # initialize sample storage
    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              obs_dim, envs[0].action_space,
                              actor_critic.recurrent_hidden_state_size)

    # initialize augmentor
    if args.use_aug or args.use_cl:
        augmentor = OneDAug(args.magnitude, args.sampling, args.noise, args.timeshift, args.permutation)
        rollouts_aug = RolloutStorage(args.num_steps, args.num_processes,
                                      obs_dim, envs[0].action_space,
                                      actor_critic.recurrent_hidden_state_size)

    obs = env_processor.reset()
    if args.use_cl:
        contrastive_encoder = CU(envs[0].observation_space.shape[0], args.shallow_dim, args.deep_dim)
        #contrastive_encoder.load_state_dict("./result/cl/msa/0602_3/encoder.pt")
        feature = contrastive_encoder.encode_obs(obs)
        rollouts.obs[0].copy_(feature)
        rollouts_aug.obs[0].copy_(feature)
    else:
        rollouts.obs[0].copy_(obs)
        if args.use_aug:
            rollouts_aug.obs[0].copy_(obs)

    rollouts.to(device)
    if args.use_aug or args.use_cl:
        rollouts_aug.to(device)

    # initialize training settings
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    vf_loss, pg_loss, rewards_rcd = [], [], []
    vf_loss_aug, pg_loss_aug = [], []

    for j in tqdm(range(num_updates)):
        ori_obs = []
        aug_obs = []
        avg_rwd = 0.
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                # print("feature: ", rollouts.obs[step])
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])
            # Obser reward and next obs
            obs, reward, done, infos = env_processor.step(action)
            # print("obs: ", obs)
            ori_obs.append(obs)
            aug_obs_ = augmentor.gen_aug(obs)
            aug_obs.append(aug_obs_)
            avg_rwd += reward.mean()

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
            if args.use_cl:
                feature = contrastive_encoder.encode_obs(obs)
                rollouts.insert(feature, recurrent_hidden_states, action, action_log_prob, value, reward, masks,
                                bad_masks)
                aug_feature = contrastive_encoder.encode_obs(aug_obs_)
                rollouts_aug.insert(aug_feature, recurrent_hidden_states, action, action_log_prob, value, reward, masks,
                                bad_masks)
            else:
                rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks, bad_masks)

            if args.use_aug:
                rollouts_aug.insert(aug_obs_, recurrent_hidden_states, action, action_log_prob, value, reward, masks,
                                    bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        if args.use_cl:
            if j % 2 == 0 and j < 0.7*num_updates:
                x = 1
                contrastive_encoder.train(augmentor, ori_obs, aug_obs, 10, args)
            #tensor_obs = torch.stack(ori_obs)
            #encoded_obs = contrastive_encoder.encode_obs(tensor_obs)
            #rollouts.obs[1:].copy_(encoded_obs)

        if args.use_aug or args.use_cl:
            rollouts_aug.compute_returns(next_value, args.use_gae, args.gamma,
                                         args.gae_lambda, args.use_proper_time_limits)
            value_loss_aug, action_loss_aug, dist_entropy_aug = agent.update(rollouts_aug)
            vf_loss_aug.append(value_loss_aug)
            pg_loss_aug.append(action_loss_aug)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        vf_loss.append(value_loss)
        pg_loss.append(action_loss)
        rewards_rcd.append(avg_rwd / args.num_steps)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0 or j == num_updates - 1):
            torch.save(actor_critic.state_dict(), file_dir + "/actor_critic.pt")
            if args.use_cl:
                torch.save(contrastive_encoder.state_dict(), file_dir + "/encoder.pt")

    if args.use_aug or args.use_cl:
        plot_utils.plot_double(pg_loss_aug, vf_loss_aug, "train_loss_aug", file_dir)
    plot_utils.plot_single(rewards_rcd, "train_reward", file_dir)
    plot_utils.plot_double(pg_loss, vf_loss, "train_loss", file_dir)


def test(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0" if args.cuda else "cpu")

    file_dir = args.save_dir + '/' + args.file_name

    env_param_generator = env_utils.ParamGenerator(args.env_params)
    env_param = env_param_generator.generate()[0]
    env = gym.make(args.env_name, **env_param)

    if args.use_cl:
        obs_dim = (args.shallow_dim,)
    else:
        obs_dim = (250,)

    # initialize agent
    actor_critic = Policy(obs_dim, env.action_space,
                          base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)
    actor_critic.load_state_dict(torch.load(file_dir + "/actor_critic.pt"))

    contrastive_encoder = None
    if args.use_cl:
        contrastive_encoder = CU(250, args.shallow_dim, args.deep_dim)
        contrastive_encoder.to(device)
        contrastive_encoder.load_state_dict(file_dir + "/encoder.pt")

    #if not os.path.exists(file_dir + "/test"):
    #    os.mkdir(file_dir + "/test")
    action_cumus, suppression_rs = [], []
    for i in tqdm(range(10)):
        env_param = env_param_generator.generate()[0]
        env_utils.write_env_info([env_param], file_dir, "test")
        env = gym.make(args.env_name, **env_param)
        action_cumu, suppression_r = exp_utils.test(i, env, actor_critic, contrastive_encoder, file_dir, device)
        action_cumus.append(action_cumu)
        suppression_rs.append(suppression_r)
    print(np.mean(suppression_rs))
    print(np.mean(action_cumus))
    #print(suppression_rs)

    #dict = {'action_cumu': action_cumus, 'suppression_r': suppression_rs}
    #np.save("%s/test_metric.npy" % file_dir, dict)


if __name__ == "__main__":
    args = get_args()

    if args.mode == "train":
        train_agent(args)
    else:
        test(args)
