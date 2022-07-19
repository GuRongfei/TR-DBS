import numpy as np
import torch
import matplotlib.pyplot as plt


def plot_single(value, mode, file_dir):
    if mode == "train_reward":
        color = "yellowgreen"
        x_label = "Updates"
        y_label = "Train Rewards"
        file_name = "TrainReward"
    else:
        return

    fig = plt.figure(figsize=(25, 10))
    ax = fig.add_subplot(111)
    ax.tick_params(labelsize=30, pad=10)
    ax.plot(value, '-', linewidth=3, c=color, label=y_label)
    ax.grid()
    ax.set_xlabel(x_label, fontsize=45, labelpad=20)
    ax.set_ylabel(y_label, fontsize=45, labelpad=20)
    ax.set_title(y_label, fontsize=55, pad=20)
    plt.tight_layout()
    plt.savefig('%s/%s.png' % (file_dir, file_name))


def plot_double(value1, value2, mode, file_dir):
    if mode == "train_loss":
        color1 = "darkseagreen"
        color2 = "teal"
        x_label = "Updates"
        y_label1 = "Policy Loss"
        y_label2 = "Value Loss"
        file_name = "TrainLoss"
    elif mode == "train_loss_aug":
        color1 = "darkseagreen"
        color2 = "teal"
        x_label = "Updates"
        y_label1 = "Policy Loss(Aug)"
        y_label2 = "Value Loss(Aug)"
        file_name = "TrainLoss(Aug)"
    else:
        return
    fig = plt.figure(figsize=(25, 20))
    ax = fig.add_subplot(211)
    ax.tick_params(labelsize=30, pad=10)
    ax.plot(value1, '-', linewidth=3, c=color1, label=y_label1)
    ax.grid()
    ax.set_xlabel(x_label, fontsize=45, labelpad=20)
    ax.set_ylabel(y_label1, fontsize=45, labelpad=20)
    ax.set_title(y_label1, fontsize=55, pad=20)

    ax2 = fig.add_subplot(212)
    ax2.tick_params(labelsize=30, pad=10)
    ax2.plot(value2, '-', linewidth=3, c=color2, label=y_label2)
    ax2.grid()
    ax2.set_xlabel(x_label, fontsize=45, labelpad=20)
    ax2.set_ylabel(y_label2, fontsize=45, labelpad=20)
    ax2.set_title(y_label2, fontsize=55, pad=20)

    plt.tight_layout()
    plt.savefig('%s/%s.png' % (file_dir, file_name))


def plot_overlap(value1, value2, mode, file_dir):
    if mode == "state_action":
        color1 = "steelblue"
        color2 = "lightcoral"
        x_label = "TimeStep"
        y_label1 = "State "
        y_label2 = "Action"
        title = "State & Action"
        file_name = "StateAction"
    else:
        return
    fig = plt.figure(figsize=(25, 10))
    ax = fig.add_subplot(111)
    ax.tick_params(labelsize=30, pad=10)
    ax2 = ax.twinx()
    ax2.tick_params(labelsize=30, pad=10)
    ax.plot(value1, '-', linewidth=3, c=color1, label=y_label1, alpha=0.7)
    ax2.plot(value2, '-', linewidth=3, c=color2, label=y_label2, alpha=0.7)
    ax.legend(bbox_to_anchor=(0.88, 1.12), fontsize=25)
    ax.grid()
    ax.set_xlabel(x_label, fontsize=45, labelpad=20)
    ax.set_ylabel(y_label1, fontsize=45, labelpad=20)
    ax2.set_ylabel(y_label2, fontsize=45, labelpad=20)
    ax2.legend(bbox_to_anchor=(1, 1.12), fontsize=25)
    ax.set_title(title, fontsize=55, pad=20)
    plt.tight_layout()
    plt.savefig('%s/%s.png' % (file_dir, file_name))
