# -*- coding: utf-8 -*-

import numpy as np


def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 값 복원
        it.iternext()

    return grad


def batch_gradient_descent(f, x_init, max_epoch=1000, alpha=0.01, epsilon=1e-6):

    x = x_init
    history_ = {'epoch': [], 'cost': [], 'x': []}

    for epoch in range(max_epoch):
        grad = numerical_gradient(f, x)
        x = x - alpha * grad  # gradient

        if epoch > 10:
            stop_point = np.sum((abs(np.diff(history_['cost'][(epoch-10):epoch])) < epsilon))
            if stop_point == 9:
                print(epoch)
                break

        history_['epoch'].append(epoch)
        history_['cost'].append(f(x))
        history_['x'].append(x)

    return x, history_

# SGD
class SBD(object):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def update(self, params, grads):
        for key in params.key():
            params[key] = params[key] - self.alpha * grads[key]

# Momentum
class Momentum(object):
    def __init__(self, alpha=0.01, gamma=0.9):
        self.alpha = alpha
        self.gamma = gamma
        self.v = None

    def update(self, params, grads):
        for key in params.keys():
            self.v[key] = self.gamma * self.v[key] - self.alpha * grads[key]
            params[key] = params[key] + self.v[key]