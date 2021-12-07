from matplotlib import pyplot as plt
import numpy as np
from matplotlib import animation
import matplotlib
import torch


def plot_emg(emg, mov_str='some', rep_str='plot'):
    plt.figure()

    for m in range(emg.shape[1]):
        '''
        plt.rcParams['xtick.bottom'] = False
        plt.rcParams['xtick.labelbottom'] = False
        plt.xticks(False)
        if m == 11:
            plt.rcParams['xtick.bottom'] = True
            plt.rcParams['xtick.labelbottom'] = True
            plt.xticks(True)
        '''
        plt.subplot(emg.shape[1], 1, m + 1)
        plt.yticks(fontsize=6)
        plt.plot(emg[:, m])

    plt.suptitle(mov_str + '_' + rep_str)
    # plt.show()

    return plt.gcf()


def plot_emg_tensor(emg, mov_str='some', rep_str='plot'):
    plt.figure()

    for m in range(emg.shape[2]):
        plt.subplot(emg.shape[1], 1, m + 1)
        plt.yticks(fontsize=6)
        plt.plot(emg[0, :, m])

    plt.suptitle(mov_str + '_' + rep_str)
    # plt.show()

    return plt.gcf()


def plot_glove(glove: np.array, mov_str: str = 'some', rep_str: str = 'plot'):
    clrs_list = ['b', 'lawngreen', 'r', 'magenta', 'yellow', 'k']  # list of basic colors
    styl_list = ['-', '--', '-.', ':']  # list of basic linestyles

    targets = []
    for j in range(1, glove.shape[1] + 1):
        targets.append(j)

    plt.figure()

    for j in range(glove.shape[1]):
        clr = clrs_list[j // 4]
        styl = styl_list[j % 4]
        plt.plot(glove[:, j], label=targets[j], color=clr, ls=styl)

    plt.legend(prop={'size': 8})
    plt.title(mov_str + '_' + rep_str)
    # plt.show()

    return plt.gcf()


def plot_glove_win(window: np.array, mov_str: str = 'some', rep_str: str = 'plot'):
    glove = window.glove
    emg = window.emg

    clrs_list = ['b', 'lawngreen', 'r', 'magenta', 'yellow', 'k']  # list of basic colors
    styl_list = ['-', '--', '-.', ':']  # list of basic linestyles

    targets = []
    for j in range(1, glove.shape[0] + 1):
        targets.append(j)

    AVG_plot = np.array([np.array(glove) for _ in range(emg.shape[0])])

    plt.figure()

    for j in range(glove.shape[0]):
        clr = clrs_list[j // 4]
        styl = styl_list[j % 4]
        plt.plot(AVG_plot[:, j], label=targets[j], color=clr, ls=styl)

    plt.legend(prop={'size': 8})
    plt.title(mov_str + '_' + rep_str)
    # plt.show()

    return plt.gcf()


def plot_glove_avg(glove: np.array, window_size: int, mov_str: str = 'some', rep_str: str = 'plot'):
    clrs_list = ['b', 'lawngreen', 'r', 'magenta', 'yellow', 'k']  # list of basic colors
    styl_list = ['-', '--', '-.', ':']  # list of basic linestyles

    targets = []
    for j in range(1, glove.shape[0] + 1):
        targets.append(j)

    AVG_plot = np.array([np.array(glove) for _ in range(window_size)])

    plt.figure()

    for j in range(glove.shape[0]):
        clr = clrs_list[j // 4]
        styl = styl_list[j % 4]
        plt.plot(AVG_plot[:, j], label=targets[j], color=clr, ls=styl)

    plt.legend(prop={'size': 8})
    plt.title(mov_str + '_' + rep_str)
    # plt.show()

    return plt.gcf()


def plot_glove_whole_movement_windows(glove_windows, window_size: int, mov_str: str = 'some', rep_str: str = 'plot'):
    clrs_list = ['b', 'lawngreen', 'r', 'magenta', 'yellow', 'k']  # list of basic colors
    styl_list = ['-', '--', '-.', ':']  # list of basic linestyles

    targets = []
    for j in range(1, glove_windows.shape[1] + 1):
        targets.append(j)

    length = window_size * glove_windows.shape[0]
    y = np.linspace(0, length, glove_windows.shape[0])

    plt.figure()

    for j in range(glove_windows.shape[1]):
        clr = clrs_list[j // 4]
        styl = styl_list[j % 4]
        plt.plot(y, glove_windows[:, j], label=targets[j], color=clr, ls=styl)

    plt.legend(prop={'size': 8})
    plt.title(mov_str + '_' + rep_str)
    # plt.show()

    return plt.gcf()


def save_plot(plot: plt.figure, file_name: str = 'some_plot', is_emg: bool = False):
    if len(plot.axes) > 0:
        filename = plot.axes[0].get_title()
    else:
        filename = file_name
    if is_emg:
        path = '/emg_plots/' + filename
    else:
        path = '/glove_plots/' + filename

    plot.savefig('../plots' + path)


def show_plot(plot: plt.figure):
    plot.show()
    plt.close(plot)
