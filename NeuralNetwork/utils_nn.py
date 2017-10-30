# -*- coding: utf-8 -*-

import numpy as np


def softmax(a):
    c = np.max(a)
    return np.exp(a - c) / np.sum(np.exp(a - c))


def ReLu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def step_function(x):
    y = x > 0
    return y.astype(np.int)


def mean_squared_error(y, t):
    return (1/2) * np.sum((y - t)**2)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y)) / batch_size