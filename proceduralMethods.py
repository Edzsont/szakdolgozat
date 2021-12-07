import datetime

import numpy as np
import scipy.io as spio

import torch

from MemoryGloveNet import GloveNet
from LightningGeneralFunctions import preprocess, denorm
from plotter import plot_glove_whole_movement_windows, show_plot, save_plot


def rare_glove_general(glove: np.array, joints: tuple):
    array = glove[:, joints[0] - 1:joints[0]]
    for idx in joints:
        if idx != joints[0]:
            array = np.hstack((array, glove[:, idx - 1:idx]))

    return array


def rare_glove_v01(glove: np.array):  # flexion/extension, ring and little finger moves together + wrist
    return rare_glove_general(glove, (1, 3, 4, 5, 7, 8, 9, 10, 12, 21, 22))


def rare_glove_v02(glove: np.array):  # flexion/extension + wrist
    return rare_glove_general(glove, (1, 3, 4, 5, 7, 8, 9, 10, 12, 14, 16, 21, 22))


def rare_glove_v03(glove: np.array):  # everything (except noisy 6)
    return rare_glove_general(glove, (1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22))


def emg_process_8_loop(emg: np.array):
    return np.hstack((emg[:, 7:8], emg[:, 0:8], emg[:, 0:1]))


def emg_process_10_loop(emg: np.array):
    return np.hstack((emg[:, 7:8], emg[:, 0:8], emg[:, 0:1], emg[:, 8:9], emg[:, 9:10]))


def emg_process_10(emg: np.array):
    return np.hstack((emg[:, 7:8], emg[:, 0:8], emg[:, 0:1])), emg[:, 8:9], emg[:, 9:10]


def load_subject(database: int, subject: int, exercise: int,
                 glove_rarefier: callable = None, process_emg: callable = None):
    path = '../data/DB' + str(database) + '/S' + str(subject) + '_'
    if database == 1:
        path += 'A1_E' + str(exercise)
    else:
        path += 'E' + str(exercise) + '_A1'
    path += '.mat'

    glove_path = '../CALIBRATED_KINEMATIC_DB_v2/s_' + str(subject + 27) + '_angles' + \
                 '/S' + str(subject + 27) + '_E' + str(exercise) + '_A1.mat'

    lib = spio.loadmat(path)
    emg = lib['emg']
    movement = lib['restimulus']
    repetition = lib['rerepetition']
    # glove = lib['glove']
    lib = spio.loadmat(glove_path)
    glove = lib['angles']

    emg_select = []
    glove_select = []
    rep_data = []
    movement_data = []
    current_mov = 1
    current_rep = repetition[0]  # 1

    # emg_select = np.array([np.array(x) for x in emg if x[0] != 0])
    for i in range(emg.shape[0]):
        if movement[i] != 0:
            if current_mov != movement[i]:  # new movement (also new repetition)
                if glove_rarefier:
                    glove_select = glove_rarefier(np.array(glove_select))
                if process_emg:
                    emg_select = process_emg(np.array(emg_select))
                rep_data.append((np.array(emg_select), np.array(glove_select)))
                movement_data.append(rep_data)
                rep_data = []
                emg_select = []
                glove_select = []
                current_mov = movement[i]
                current_rep = repetition[i]
            else:
                if current_rep != repetition[i]:  # new repetition (no new movement)
                    if glove_rarefier:
                        glove_select = glove_rarefier(np.array(glove_select))
                    if process_emg:
                        emg_select = process_emg(np.array(emg_select))
                    rep_data.append((np.array(emg_select), np.array(glove_select)))
                    emg_select = []
                    glove_select = []
                    current_rep = repetition[i]

            emg_select.append(emg[i])
            glove_select.append(glove[i])
    if glove_rarefier:
        glove_select = glove_rarefier(np.array(glove_select))
    if process_emg:
        emg_select = process_emg(np.array(emg_select))
    rep_data.append((np.array(emg_select), np.array(glove_select)))
    movement_data.append(rep_data)

    return movement_data


def add_subject(subjects: list, database: int, subject: int, exercise: int,
                glove_rarefier: callable = None, process_emg: callable = None):
    movement_data = load_subject(database=database, subject=subject, exercise=exercise,
                                 glove_rarefier=glove_rarefier, process_emg=process_emg)

    new_subjects = [x for x in subjects]
    new_subjects.append(movement_data)

    return new_subjects


def load_exercise_all_subjects(database: int = 2, exercise: int = 1, num_subjects: int = 40,
                               glove_rarefier: callable = None, process_emg: callable = None):
    subjects = []
    for i in range(1, num_subjects + 1):
        subjects = add_subject(subjects=subjects, database=database, subject=i, exercise=exercise,
                               glove_rarefier=glove_rarefier, process_emg=process_emg)
        print('\tsubject ' + str(i) + ' added')

    return subjects


def glove_label(glove: np.array):
    return glove[-1]


def glove_previous_label(glove: np.array):
    return glove[0]


def window_repetition(repetition: tuple, window_size: int):
    emg = repetition[0]
    glove = repetition[1]
    emg_window = []
    glove_window = []
    windows = []

    if emg.shape[0] != glove.shape[0]:
        print('ERROR IN DATA')
        print('emg length: ', emg.shape[0])
        print('glove length: ', glove.shape[0])
    else:
        sum_data = emg.shape[0] - (emg.shape[0] % window_size)
        #  print(emg.shape[0], ' - ', (emg.shape[0] % window_size), ' = ', sum_data)
        #  last window is lost (would contain less data than window_size)
        for i in range(sum_data):
            if i > 0:
                if i % window_size == 0:
                    #  closing a window
                    #  print('new window: ', len(emg_window), len(glove_window))
                    windows.append((np.array(emg_window), glove_label(np.array(glove_window)),
                                    glove_previous_label(np.array(glove_window))))
                    emg_window = []
                    glove_window = []

            emg_window.append(emg[i, :])
            glove_window.append(glove[i, :])

        #  print('emg: ', emg.shape[0], ', glove: ', glove.shape[0], ', window size: ', window_size,
        #        ', number of windows:', sum_data / window_size, ', sum data in windows: ', sum_data)

        return windows


def window_movement(movement: list, window_size: int):
    windows = []
    for idx, rep in enumerate(movement):
        windows += window_repetition(repetition=rep, window_size=window_size)
        movement[idx] = None
        del rep

    return windows


def window_subject(subject: list, window_size: int):
    windows = []
    for idx, mov in enumerate(subject):
        windows += window_movement(movement=mov, window_size=window_size)
        subject[idx] = None

    return windows


def window_all_subjects(subjects: list, window_size: int):
    windows = []
    for idx, sub in enumerate(subjects):
        windows += window_subject(subject=sub, window_size=window_size)
        subjects[idx] = None
        print('\tsubject ' + str(idx) + ' windowed')

    return windows


def k_fold_separate(windows: list, k: int = 5):
    folds = []
    for _k in range(1, k + 1):
        folds.append([])

    num_wins = len(windows)
    for idx in range(num_wins):
        fold = idx % k
        folds[fold].append(windows[idx])

    return folds


def window_list_to_tensor(windows: list):
    emg_data = []
    glove_data = []
    previous = []
    for idx, x in enumerate(windows):
        emg_data.append(x[0])
        glove_data.append(x[1])
        previous.append(x[2])
        windows[idx] = None

    emg_tensor = torch.tensor(emg_data)
    glove_tensor = torch.tensor(glove_data)
    previous_tensor = torch.tensor(previous)

    return tuple((emg_tensor, glove_tensor, previous_tensor))


def save_window_list_to_pt(windows: list, path: str):
    data = window_list_to_tensor(windows)
    torch.save(data, path)


def save_folds(folds: list, path: str):
    path = '../own_data/' + path
    for i in range(len(folds)):
        save_window_list_to_pt(folds[i], path + '/' + str(i) + '.pt')
        folds[i] = None


def full_load_save(window_size: int, path: str, k: int = 5, database: int = 2, exercise: int = 1,
                   num_subjects: int = 40, glove_rarefier: callable = None,
                   process_emg: callable = None):
    print('STARTING PROCESS')
    print('LOADING DATA')
    subjects = load_exercise_all_subjects(database=database, exercise=exercise, num_subjects=num_subjects,
                                          glove_rarefier=glove_rarefier, process_emg=process_emg)
    print('LOADING COMPLETE')
    print('WINDOWING DATA')
    windowed = window_all_subjects(subjects=subjects, window_size=window_size)
    print('WINDOWING DATA')
    print('K-FOLDING DATA')
    k_fold = k_fold_separate(windows=windowed, k=k)
    print('K-FOLDING COMPLETE')
    print('SAVING DATA')
    save_folds(k_fold, path + '/' + str(window_size))
    print('SAVING COMPLETE')
    print('PROCESS COMPLETE')


def save_emg_folds(process_emg: callable = None, window_size: int = 200, k: int = 5, path: str = ''):
    print('LOADING DATA')
    subjects = load_exercise_all_subjects(process_emg=process_emg)

    print('WINDOWING DATA')
    windows = window_all_subjects(subjects=subjects, window_size=window_size)
    windows = [x[0] for x in windows]
    del subjects

    fold_save_general(windows, path, window_size, k)


def save_glove_folds(glove_rarefier: callable = None, window_size: int = 200, k: int = 5, path: str = ''):
    print('LOADING DATA')
    subjects = load_exercise_all_subjects(glove_rarefier=glove_rarefier)

    print('WINDOWING DATA')
    windows = window_all_subjects(subjects, window_size=window_size)
    windows = [x[1] for x in windows]
    del subjects

    fold_save_general(windows, path, window_size, k)


def save_previous_folds(glove_rarefier: callable = None, window_size: int = 200, k: int = 5, path: str = ''):
    print('LOADING DATA')
    subjects = load_exercise_all_subjects(glove_rarefier=glove_rarefier)

    print('WINDOWING DATA')
    windows = window_all_subjects(subjects, window_size=window_size)
    windows_2 = [x[2] for x in windows]
    del windows
    del subjects

    fold_save_general(windows_2, path, window_size, k)


def fold_save_general(windows: list, path: str, window_size: int = 200, k: int = 5):
    print('FOLDING')
    folds = k_fold_separate(windows, k)
    del windows

    print('SAVING DATA')
    for idx, fold in enumerate(folds):
        tensor = torch.tensor(fold)
        folds[idx] = None
        torch.save(tensor, '../own_data/' + path + '/' + str(window_size) + '/' + str(idx) + '.pt')
        print('\t' + str(idx) + 'th fold saved')

    print('PROCESS COMPLETE')

# x, y =load_batch(glove_rarefier=rare_glove_v01, process_emg=emg_process_8_loop,
# window_size=200, subject=20, movement=4, repetition=2)


def load_batch(database: int = 2, subject: int = 1, exercise: int = 1,
               glove_rarefier: callable = None,
               process_emg: callable = None,
               movement: int = 1, repetition: int = 1, window_size: int = 128):
    subject = load_subject(database=database, subject=subject, exercise=exercise,
                           glove_rarefier=glove_rarefier, process_emg=process_emg)
    mov = subject[movement]
    rep = mov[repetition]

    windows = window_repetition(rep, window_size)
    glove = torch.tensor([x[1] for x in windows])
    emg = torch.tensor([x[0] for x in windows])
    emg, glove = preprocess(emg, glove)

    return emg, glove


def load_memo_batch(database: int = 2, subject: int = 1, exercise: int = 1,
                    glove_rarefier: callable = None,
                    process_emg: callable = None,
                    movement: int = 1, repetition: int = 1, window_size: int = 128):
    subject = load_subject(database=database, subject=subject, exercise=exercise,
                           glove_rarefier=glove_rarefier, process_emg=process_emg)
    mov = subject[movement]
    rep = mov[repetition]
    windows = window_repetition(rep, window_size)

    glove = []
    emg = []
    previous = []
    for win in windows:
        emg.append(win[0])
        glove.append(win[1])
        previous.append(win[2])

    emg = torch.tensor(emg)
    glove = torch.tensor(glove)
    previous = torch.tensor(previous)

    return tuple((emg, glove, previous))

# x, y, z = load_memo_batch(glove_rarefier=rare_glove_v01,
# process_emg=emg_process_8_loop,window_size=200, subject=20, movement=4, repetition=2)


def evaluate_network(emg: torch.tensor, glove: torch.tensor, window_size: int = 128,
                     path: str = 'version_153/checkpoints/epoch=14-step=107459.ckpt'):
    path = '../ph_signal1/lightning_logs/' + path
    model = GloveNet.load_from_checkpoint(path)

    results = []

    for window in emg:
        # print(window.shape)
        data = window.view((1, 1, window.shape[1], window.shape[2]))
        results.append(torch.Tensor.detach(model(data)))

    results = torch.stack(results).view((len(results), -1))

    save_plot(plot_glove_whole_movement_windows(np.array(glove), window_size, 'expected', 'results'))
    show_plot(plot_glove_whole_movement_windows(np.array(glove), window_size, 'expected', 'results'))
    save_plot(plot_glove_whole_movement_windows(np.array(results), window_size, 'network', 'results'))
    show_plot(plot_glove_whole_movement_windows(np.array(results), window_size, 'network', 'results'))

# evaluate_network(x, y, window_size=200, path='version_165/checkpoints/epoch=9-step=34009.ckpt')


def evaluate_memo_network(emg: torch.tensor, glove: torch.tensor, previous: torch.tensor, window_size: int = 200,
                     path: str = 'version_153/checkpoints/epoch=14-step=107459.ckpt'):
    path = '../ph_signal1/lightning_logs/' + path
    model = GloveNet.load_from_checkpoint(path)

    # result = np.array(torch.Tensor.detach(model(emg)))

    results_given = []
    results_comp = []

    save_plot(plot_glove_whole_movement_windows(np.array(glove), window_size, 'expected', 'results'))
    show_plot(plot_glove_whole_movement_windows(np.array(glove), window_size, 'expected', 'results'))

    emg, glove, previous = preprocess(emg, glove, previous)
    prev = None

    for idx in range(emg.shape[0]):
        # print(window.shape)
        data = emg[idx].view((1, 1, emg[idx].shape[1], emg[idx].shape[2]))
        extra_given = previous[idx].view((1, previous[idx].shape[0]))
        if prev is None:
            extra_comp = previous[idx].view((1, previous[idx].shape[0]))
        else:
            extra_comp = prev
        res_given = torch.Tensor.detach(model(data, extra_given))
        res_comp = torch.Tensor.detach(model(data, extra_comp))
        results_given.append(res_given)
        results_comp.append(res_comp)
        prev = res_comp

    results_given = torch.stack(results_given).view((len(results_given), -1))
    results_comp = torch.stack(results_comp).view((len(results_comp), -1))

    show_plot(plot_glove_whole_movement_windows(np.array(results_given), window_size, 'network', 'results_norm_given_labels'))
    results_given = denorm(results_given)
    show_plot(plot_glove_whole_movement_windows(np.array(results_comp), window_size, 'network', 'results_norm_computed_labels'))
    results_comp = denorm(results_comp)

    save_plot(plot_glove_whole_movement_windows(np.array(results_given), window_size, 'network', 'results_given_labels'))
    save_plot(plot_glove_whole_movement_windows(np.array(results_comp), window_size, 'network', 'results_computed_labels'))
    show_plot(plot_glove_whole_movement_windows(np.array(results_given), window_size, 'network', 'results_denorm_given'))
    show_plot(plot_glove_whole_movement_windows(np.array(results_comp), window_size, 'network', 'results_denorm_computed'))

# evaluate_memo_network(x, y, z, path='version_163/checkpoints/epoch=10-step=35246.ckpt')


def acc_function(x):
    if x <= 2:
        return 1 - (1 - 1/np.square(np.e))*x/2
    else:
        return np.power(np.e, (np.square(np.e) - 3) - ((np.square(np.e)-1)/2)*x)


def accuracy(expected, logits):
    loss = np.abs(expected-logits)
    for i in range(len(loss)):
        loss[i] = acc_function(loss[i])

    return loss


def runtime(some_function: callable):
    start_time = datetime.datetime.now()

    some_function()

    end_time = datetime.datetime.now()

    return (end_time - start_time).total_seconds() * 1000
