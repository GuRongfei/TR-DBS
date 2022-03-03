import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
from collections import namedtuple


def plot_train_rewards(opt, rewards, best_iteration):
    train_path = opt.train_root_path + '/' + opt.train_package_name + '/'

    fig = plt.figure(figsize=(25, 10))
    ax = fig.add_subplot(111)
    ax.tick_params(labelsize=30, pad=10)
    ax.plot(rewards, '-', linewidth=3, c='yellowgreen', label='Train Rewards')
    ax.grid()
    ax.set_xlabel("Updates", fontsize=45, labelpad=20)
    ax.set_ylabel("Rewards", fontsize=45, labelpad=20)
    ax.set_title("Train Rewards", fontsize=55, pad=20)
    plt.axvline(best_iteration, linewidth=3, c='lightcoral')
    plt.tight_layout()
    plt.savefig('%sTrainReward.png' % train_path)


def plot_loss(opt, pg_loss, vf_loss, best_iteration):
    train_path = opt.train_root_path + '/' + opt.train_package_name + '/'

    fig = plt.figure(figsize=(25, 20))
    ax = fig.add_subplot(211)
    ax.tick_params(labelsize=30, pad=10)
    ax.plot(pg_loss, '-', linewidth=3, c='darkseagreen', label='Policy Loss')
    ax.grid()
    ax.set_xlabel("Updates", fontsize=45, labelpad=20)
    ax.set_ylabel("Loss", fontsize=45, labelpad=20)
    ax.set_title("Policy Loss", fontsize=55, pad=20)
    ax.axvline(best_iteration, linewidth=3, c='lightcoral')

    ax2 = fig.add_subplot(212)
    ax2.tick_params(labelsize=30, pad=10)
    ax2.plot(vf_loss, '-', linewidth=3, c='teal', label='Value Loss')
    ax2.grid()
    ax2.set_xlabel("Updates", fontsize=45, labelpad=20)
    ax2.set_ylabel("Loss", fontsize=45, labelpad=20)
    ax2.set_title("Value Loss", fontsize=55, pad=20)
    ax2.axvline(best_iteration, linewidth=3, c='lightcoral')

    plt.tight_layout()
    plt.savefig('%sTrainLoss.png' % train_path)


def plot_test_result(opt, states_x, actions):
    test_path = opt.test_root_path + '/' + opt.test_package_name + '/'

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
    plt.savefig('%sStateAction.png' % test_path)
