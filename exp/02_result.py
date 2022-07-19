import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read_data(titles, paths):
    mode_num = len(titles)

    action_cumus = {}
    suppression_rates = {}
    stm_means, stm_stds, sup_means, sup_stds = [], [], [], []

    for i in range(mode_num):
        load_dict = np.load("%s/test_metric.npy" % paths[i], allow_pickle=True).item()
        action_cumu = load_dict['action_cumu']
        suppression_r = load_dict['suppression_r']
        action_cumus[titles[i]] = action_cumu
        suppression_rates[titles[i]] = suppression_r
        stm_means.append(np.mean(action_cumu))
        stm_stds.append(np.std(action_cumu))
        sup_means.append(np.mean(suppression_r))
        sup_stds.append(np.std(suppression_r))

    return stm_means, stm_stds, sup_means, sup_stds


def plot_contr(mode_num, means1, means2, stds1, stds2, titles, label):
    fig = plt.figure(figsize=(8, 3))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    ax1.set_ylabel('Suppression Rate', fontsize=15, labelpad=5)
    ax2.set_ylabel('Stimulation Magnitude', fontsize=15, labelpad=10)
    ax1.set_xticks(range(mode_num), labelpad=5)
    ax1.set_xticklabels(titles, fontsize=15)
    ax1.tick_params(labelsize=15, pad=5)
    ax2.tick_params(labelsize=15, pad=5)

    rect1 = ax1.bar(x=range(mode_num), height=means1, color="steelblue", align="edge", width=-0.4, alpha=0.7)
    rect2 = ax2.bar(x=range(mode_num), height=means2, color="lightcoral", align="edge", width=0.4, alpha=0.7)

    x = np.arange(mode_num)
    for i in range(mode_num):
        z1 = [x[i]-0.2, x[i]-0.2]
        w1 = [means1[i]-stds1[i], means1[i]+stds1[i]]

        z2 = [x[i]+0.2, x[i]+0.2]
        w2 = [means2[i]-stds2[i], means2[i]+stds2[i]]

        ax1.errorbar(x[i]-0.2, means1[i], yerr=stds1[i], ecolor='blue', capsize=4)
        ax2.errorbar(x[i]+0.2, means2[i], yerr=stds2[i], ecolor='red', capsize=4)

        #ax1.plot(z1, w1, color='red', alpha=0.7)
        #ax2.plot(z2, w2, color='blue', alpha=0.7)

    plt.legend((rect1, rect2), ("Suppression Rate", "Stimulation Magnitude"), fontsize=10)
    plt.tight_layout()
    plt.savefig("./result/img/%s.png" % label)
    #ax1.set_ylim(0, 1)
    """rect1 = ax1.bar(x=range(mode_num), height=means, color="steelblue", align="edge", width=0.4, alpha=0.7)
    rect2 = plt.bar(x=range(mode_num), height=means, color="lightcoral", align="edge", width=-0.4, alpha=0.7)

    plt.ylabel(u'%s' % label)

    plt.xticks(range(mode_num), titles)#, FontSize=6)

    plt.legend((rect1, rect2), (u'Mean',u'Mean2'))

    x = np.arange(mode_num)

    for i in range(mode_num):
        z1 = [x[i]-0.2, x[i]-0.2]
        w1 = [means[i]-stds[i], means[i]+stds[i]]

        z2 = [x[i]+0.2, x[i]+0.2]
        w2 = [means[i]-stds[i], means[i]+stds[i]]

        plt.plot(z1, w1, color='red', alpha=0.7)
        plt.plot(z2, w2, color='blue', alpha=0.7)

    for x, y in enumerate(means):
        plt.text(x, y+0.001, '%.3f' % y, ha='center')

    plt.savefig("./result/img/%s.png" % label)"""


if __name__ == "__main__":
    title_path = [[0, "Baseline", "./result/normal/0526_7"],
                  [1, "DR", "./result/dr_result/0525_5"],
                  [2, "Amp", "./result/aug_result/magnitude/0526_3"],
                  [3, "Samp", "./result/aug_result/sampling/0526_8"],
                  [4, "N", "./result/aug_result/noise/0526_9"],
                  [5, "Perm", "./result/aug_result/permutation/0526_5"],
                  [6, "Shft", "./result/aug_result/shift/0526_1"],
                  [7, "A-Sa", "./result/aug_result/m_sa/0526_1"],
                  [8, "A-N", "./result/aug_result/mn/0602_10"],
                  [9, "Sa-N", "./result/aug_result/san/0602_3"],
                  [10, "DAT(A-Sa)", "./result/cl/msa/0602_3"],
                  [11, "New Mag", "./result/aug_result/magnitude/0614_06"],
                  [12, "New Samp", "./result/aug_result/sampling/0614_05"]]
    """titles = ["Normal", "Domain Rand",
              "Magnitude", "Sampling", "Noise", "Shift", "Permutation",
              "M-Sa", "Mn", "San",
              "CL_msa"]
    paths = ["./result/normal/0526_7", "./result/dr_result/0525_5",
             "./result/aug_result/magnitude/0526_3",
             "./result/aug_result/sampling/0526_8", "./result/aug_result/noise/0526_9",
             "./result/aug_result/shift/0526_1", "./result/aug_result/permutation/0526_5",
             "./result/aug_result/m_sa/0526_1", "./result/aug_result/mn/0602_10", "./result/aug_result/san/0602_3",
             "./result/cl/0602_3"]"""

    picked = [0, 2, 3, 11, 12]
    titles, paths = [], []
    for i in picked:
        titles.append(title_path[i][1])
        paths.append(title_path[i][2])


    stm_means, stm_stds, sup_means, sup_stds = read_data(titles, paths)
    for i in range(len(titles)):
        print("--------------------------")
        print(" ", titles[i])
        print(" stim_q:  %-5.5f | %-5.5f" % (stm_means[i], stm_stds[i]))
        print(" supr_r:  %-5.5f | %-5.5f" % (sup_means[i], sup_stds[i]))
    #plot_contr(len(titles), stm_means, stm_stds, titles, "Stimulation Magnitude")
    plot_contr(len(titles), sup_means, stm_means, sup_stds, stm_stds, titles, "Combined")
