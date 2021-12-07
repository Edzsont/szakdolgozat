import torch
from torch import nn


def loss_matrix(matrix):
    row_sums = [None] * matrix.shape[0]
    for idx, row in enumerate(matrix):
        row_sum = 0
        for idx_2, value in enumerate(row):
            row_sum += value

        row_sums[idx] = row_sum

    return row_sums


def my_vector_loss(logits, labels):
    loss = torch.abs(logits-labels)

    return loss_matrix(loss)


def my_loss_mean(vector_loss):
    mean = 0
    for idx, row in enumerate(vector_loss):
        mean += row

    mean /= len(vector_loss)

    return mean


def my_loss(logits, labels):
    return my_loss_mean(my_vector_loss(logits, labels))


def lossy_loss(logits, labels):
    loss = nn.L1Loss(reduction='none')
    matrix = loss(logits, labels)

    return loss_matrix(matrix)