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
from utils.env_utils import EnvProcessor
from data_augmentation.oned_aug import OneDAug

if __name__ == "__main__":
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    file_dir = args.save_dir + '/' + args.file_name

    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = [gym.make(args.env_name) for i in range(args.num_processes)]

    env_processor = EnvProcessor(envs)

    actor_critic = Policy(
        envs[0].observation_space.shape,
        envs[0].action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
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
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs[0].observation_space.shape, envs[0].action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = env_processor.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    #print("num_updates: ", num_updates)
    #print("args.num_env_steps: ", args.num_env_steps)
    #print("args.num_steps: ", args.num_steps)
    #print("args.num_processes: ", args.num_processes)

    vf_loss, pg_loss, rewards_rcd = [], [], []
    for j in tqdm(range(num_updates)):
        #break

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        avg_rwd = 0.
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                #print("obs: ", rollouts.obs[step])
                #print("rec_states: ", rollouts.recurrent_hidden_states[step])
                #print("masks: ", rollouts.masks[step])
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            #action = env_processor.clip(action)

            # Obser reward and next obs
            obs, reward, done, infos = env_processor.step(action)
            #print("action: ", action[-1])
            #print("obs: ", obs[-1])
            #print("reward: ", reward[-1])
            avg_rwd += reward.mean()

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)
            #one_d_aug = OneDAug(obs)
            #aug_obs = one_d_aug.gen_aug()

            #rollouts.insert(aug_obs, recurrent_hidden_states, action,
                            #action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        vf_loss.append(value_loss)
        pg_loss.append(action_loss)
        rewards_rcd.append(avg_rwd/args.num_steps)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0 or j == num_updates - 1):
            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
            ], file_dir + "/model.pt")

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            obs_rms = utils.get_vec_normalize(envs).obs_rms
            evaluate(actor_critic, obs_rms, args.env_name, args.seed,
                     args.num_processes, file_dir, device)

    fig = plt.figure(figsize=(25, 10))
    ax = fig.add_subplot(111)
    ax.tick_params(labelsize=30, pad=10)
    ax.plot(rewards_rcd, '-', linewidth=3, c='yellowgreen', label='Train Rewards')
    ax.grid()
    ax.set_xlabel("Updates", fontsize=45, labelpad=20)
    ax.set_ylabel("Rewards", fontsize=45, labelpad=20)
    ax.set_title("Train Rewards", fontsize=55, pad=20)
    plt.tight_layout()
    plt.savefig('%s/TrainReward.png' % file_dir)

    fig = plt.figure(figsize=(25, 20))
    ax = fig.add_subplot(211)
    ax.tick_params(labelsize=30, pad=10)
    ax.plot(pg_loss, '-', linewidth=3, c='darkseagreen', label='Policy Loss')
    ax.grid()
    ax.set_xlabel("Updates", fontsize=45, labelpad=20)
    ax.set_ylabel("Loss", fontsize=45, labelpad=20)
    ax.set_title("Policy Loss", fontsize=55, pad=20)

    ax2 = fig.add_subplot(212)
    ax2.tick_params(labelsize=30, pad=10)
    ax2.plot(vf_loss, '-', linewidth=3, c='teal', label='Value Loss')
    ax2.grid()
    ax2.set_xlabel("Updates", fontsize=45, labelpad=20)
    ax2.set_ylabel("Loss", fontsize=45, labelpad=20)
    ax2.set_title("Value Loss", fontsize=55, pad=20)

    plt.tight_layout()
    plt.savefig('%s/TrainLoss.png' % file_dir)


    test_env = gym.make(args.env_name)
    observation = test_env.reset()
    recurrent_hidden_states = torch.zeros(actor_critic.recurrent_hidden_state_size, device=device)
    masks = torch.zeros(1, device=device)
    """frames = []
    cnt = 0
    for step in range(1000):
        #frames.append(test_env.render(mode='rgb_array'))
        value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(torch.Tensor(observation), recurrent_hidden_states, masks)

        action = env_processor.clip(action)
        observation, reward, done, info = test_env.step(action.numpy().tolist()[0][0][0][0])
        #print("reward: ", reward)
        if done:
            observation = test_env.reset()"""

    actions = []
    states_x = []
    for step in range(2000 + 2000 + 500):
        if step < 2000 or step >= (2000 + 2000):
            action = 0
            observation, reward, done, info = test_env.step(action)
        else:
            value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(torch.Tensor(observation),
                                                                                       recurrent_hidden_states, masks)
            #action = env_processor.clip(action)
            action = action.numpy().tolist()[0][0][0][0]
            observation, reward, done, info = test_env.step(action)
        #print("tst_rwd: ", reward)
        actions.append(action)
        states_x.append(test_env.x_val)

    fig = plt.figure(figsize=(25, 10))
    ax = fig.add_subplot(111)
    ax.tick_params(labelsize=30, pad=10)
    ax2 = ax.twinx()
    ax2.tick_params(labelsize=30, pad=10)
    ax2.plot(actions, '-', linewidth=3, c='lightcoral', label='Action', alpha=0.7)
    ax.plot(states_x, '-', linewidth=3, c='steelblue', label='State ')
    ax.legend(bbox_to_anchor=(0.88, 1.12), fontsize=25)
    ax.grid()
    ax.set_xlabel("TimeStep", fontsize=45, labelpad=20)
    ax.set_ylabel("States", fontsize=45, labelpad=20)
    ax2.set_ylabel("Actions", fontsize=45, labelpad=20)
    ax2.legend(bbox_to_anchor=(1, 1.12), fontsize=25)
    ax.set_title("State & Action", fontsize=55, pad=20)
    plt.tight_layout()
    plt.savefig('%s/StateAction.png' % file_dir)

    """patch = plt.imshow(frames[0])
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])
    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=100)
    anim.save('./demo.gif', writer='pillow', fps=30)"""

    test_env.close()

