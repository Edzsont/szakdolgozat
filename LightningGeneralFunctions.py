import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader

from GloveDataset import GloveDataset
from Losses import my_loss
from Normalization import feature_scale, feature_standardize, standardize, inv_standardize, \
    inv_feature_standardize, inv_minmax

v01 = [0, 2, 3, 4, 5, 6, 7, 8, 10, 19, 20]
v02 = [0, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 19, 20]
v03 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
CURR_VER = v03

MEANS = [
    12.138158714161262,
    5.93387337235693,
    5.431552533282192,
    14.903740115968391,
    17.87936973056078,
    25.771064213465777,
    27.604223044440804,
    -1.753795442937972,
    30.985272342698185,
    -8.797083099133754,
    9.652247119211628,
    -1.4104840949593271,
    46.18962489089875,
    -4.971801077626776,
    11.212300931134605,
    10.07536285718291,
    -17.987046089661735,
    -16.208736629467793,
    -5.28857567045762,
    -11.714271026653554,
    -1.2581744150771286
]

STDS = [
    18.240672570417996,
    7.727349361953584,
    10.836088654104785,
    24.052309760214946,
    23.82805454692234,
    39.73015210128544,
    38.86597932222063,
    18.505788783223064,
    36.7815847148946,
    21.68844435714726,
    41.441675231385375,
    3.3590718783654485,
    53.51521666523088,
    12.41523994671092,
    22.112596246240575,
    31.191181108622036,
    36.83553659677831,
    55.750114264889056,
    22.009606619581753,
    26.236476491446762,
    12.344851229456957
]

x_min = -0.009
x_max = 0.0099
x_mean = 7.31e-08
x_std = 6.80e-05
y_min = -196.9576
y_max = 301.9159
y_mean = 11.80
# y_mean = 7.09
y_std = 31.23
# y_std = 34


def loss_function(logits, labels):
    loss1 = nn.L1Loss(reduction='mean')
    loss2 = nn.L1Loss(reduction='sum')
    loss_matrix = nn.L1Loss(reduction='none')

    matrix = loss_matrix(logits, labels)
    zeros = torch.zeros(matrix.shape[1], device='cuda:0')
    loss_column = torch.zeros(matrix.shape[0], device='cuda:0')

    for i in range(matrix.shape[0]):
        loss_column[i] = loss2(matrix[i], zeros)

    zeros = torch.zeros_like(loss_column)
    loss = loss1(loss_column, zeros)

    return loss


def preprocess(x, y, z=None):
    x = x.view([x.shape[0], 1, x.shape[1], x.shape[2]])

    # x = standardize(x, x_mean, x_std)
    # y = standardize(x=y, mean=y_mean, std=y_std)
    # x = feature_scale(x, x_min, x_max)
    # y = feature_scale(y, y_min, y_max)
    # y = feature_standardize(y=y, version=CURR_VER, MEANS=MEANS, STDS=STDS)
    if z is not None:
        z = feature_standardize(z, CURR_VER, MEANS, STDS)

        return x, y, z

    return x, y


def denorm(x):
    x = inv_feature_standardize(y=x, version=CURR_VER, MEANS=MEANS, STDS=STDS)

    return x


def general_train_step(net, batch):
    x, y = batch
    x, y = preprocess(x, y)

    logits = net.forward(x)
    loss = loss_function(logits.float(), y.float())
    net.log('train_loss', loss, prog_bar=True)

    return loss


def general_validation_step(net, batch):
    x, y = batch
    x, y = preprocess(x, y)

    logits = net.forward(x)
    loss = loss_function(logits, y)
    net.log('validation_loss', loss, prog_bar=True)

    return loss


def general_test_step(net, batch):
    x, y = batch
    x, y = preprocess(x, y)

    logits = net.forward(x)
    loss = loss_function(logits, y)
    own_loss = my_loss(logits, y)
    net.log('test_loss', loss, prog_bar=True)
    net.log('my_loss', own_loss)
    label_norm = False
    if label_norm:
        logits = inv_feature_standardize(logits, CURR_VER, MEANS, STDS)
        y = inv_feature_standardize(y, CURR_VER, MEANS, STDS)
        # logits = inv_standardize(logits, y_mean, y_std)
        # y = inv_standardize(y, y_mean, y_std)
        # logits = inv_minmax(x=logits, x_min=y_min, x_max=y_max)
        # y = inv_minmax(x=y, x_min=y_min, x_max=y_max)

        denorm_loss = loss_function(logits, y)
        denorm_my_loss = my_loss(logits, y)

        net.log('denorm_test_loss', denorm_loss)
        net.log('denorm_my_loss', denorm_my_loss)

    return loss


def memo_test_step(net, batch):
    x, y, z = batch
    x, y, z = preprocess(x, y, z)

    logits = net.forward(x, z)
    loss = loss_function(logits, y)
    own_loss = my_loss(logits, y)
    net.log('test_loss', loss, prog_bar=True)
    net.log('my_loss', own_loss)

    label_norm = False
    if label_norm:
        # logits = inv_standardize(x=logits, mean=y_mean, std=y_std)
        # y = inv_standardize(x=y, mean=y_mean, std=y_std)
        # logits = inv_minmax(x=logits, x_min=y_min, x_max=y_max)
        # y = inv_minmax(x=y, x_min=y_min, x_max=y_max)

        denorm_loss = loss_function(logits, y)
        denorm_my_loss = my_loss(logits, y)

        net.log('denorm_test_loss', denorm_loss)
        net.log('denorm_my_loss', denorm_my_loss)

    return loss


def memo_time_test(net, batch):
    x, y, z = batch
    x, y, z = preprocess(x, y, z)
    start_time = datetime.datetime.now()
    net.forward(x, z)
    end_time = datetime.datetime.now()

    time = 1000*((end_time-start_time).total_seconds())
    net.log('execution_time', time)

    return torch.tensor(time)


def general_train_loader(emg_folder, glove_folder, window_size, batch_size):
    train_data = GloveDataset(emg_folder, glove_folder, window_size, 1)
    train_data.expand(fold_number=2)
    train_data.expand(fold_number=3)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader


def general_test_loader(emg_folder, glove_folder, window_size, batch_size, fold):
    test_data = GloveDataset(emg_folder, glove_folder, window_size, fold)
    test_loader = DataLoader(test_data, batch_size, shuffle=False, drop_last=True)

    return test_loader
