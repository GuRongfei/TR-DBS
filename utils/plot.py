import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
from collections import namedtuple


def plot_bar():
    n_groups = 2

    means_b = [-0.059, -0.041]
    means_dr = [-0.264, -0.116]
    means_dt = [-1.221, -0.175]

    minval = 2
    for i in range(0, 2):
        means_b[i] += minval
        means_dr[i] += minval
        means_dt[i] += minval

    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    bar_width = 0.15

    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    def neg_tick(x, pos):
        return '%.1f' % (x - minval if x != minval else 0)
    formatter = FuncFormatter(neg_tick)
    ax.yaxis.set_major_formatter(formatter)

    rects1 = ax.bar(index, means_b, bar_width,
                    alpha=opacity, color='b', error_kw=error_config,
                    label='baseline')

    rects2 = ax.bar(index + bar_width, means_dr, bar_width,
                    alpha=opacity, color='r', error_kw=error_config,
                    label='domain randomization')

    rects3 = ax.bar(index + 2*bar_width, means_dt, bar_width,
                    alpha=opacity, color='g', error_kw=error_config,
                    label='direct transfer')

    ax.set_xlabel('Group', fontsize=15)
    ax.set_ylabel('Rewards', fontsize=15, labelpad=5)
    ax.set_title('Rewards under diff targets', fontsize=15, pad=5)
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(('2-2', '05-05'))
    ax.legend(bbox_to_anchor=(0.45, 0.75), fontsize=10)

    fig.tight_layout()
    plt.savefig('./result/tr/baselines/result2.png')


if __name__ == '__main__':
    plot_bar()