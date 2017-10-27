# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import animation, rc
import matplotlib.pylab as plt
# from IPython.display import HTML
rc('animation', html='html5')


def pltcost(history_):
    plt.plot(history_['epoch'][:30], history_['cost'][:30])
    plt.title('Cost', fontsize=20)
    plt.xlabel('epoch')
    plt.ylabel('Cost')
    plt.show()


def pltcost_w(X, y):
    ww = np.vstack((np.ones((100,)) * 50, np.linspace(-10000, 10000, 100))).T
    cost_val = []
    for w in ww:
        cost_w = cost(X, y, w)
        cost_val.append(cost_w)
    plt.scatter(ww[:, 1], cost_val)
    plt.title('cost function')
    plt.show()


class AniPlot(object):
    def __init__(self, title, frames, interval):
        self.fig = plt.figure(figsize=(8, 8))
        self.ax1 = self.fig.add_subplot(1, 1, 1)
        self.line, = self.ax1.plot([], [], 'r-', lw=2)
        self.title = title
        self.frames = frames
        self.interval = interval

    def set_data(self, X, y, history_):
        self.X = X[:, 1]
        self.xx = np.linspace(X.min(), X.max(), y.shape[0])
        self.y = history_['y']
        self.true_y = y

    def ani_init(self):
        self.line.set_data([], [])
        return self.line,

    def ani_update(self, i):
        xs = self.xx
        ys = self.y[i]
        self.line.set_data(xs, ys)
        return self.line,

    def animate(self):
        plt.scatter(self.X, self.true_y, label='dots', alpha=0.3, edgecolors='none')
        plt.title(self.title, fontsize=20)
        plt.xlabel('x')
        plt.ylabel('y')
        anim = animation.FuncAnimation(self.fig, self.ani_update, init_func=self.ani_init,
                                       frames=self.frames, interval=self.interval, blit=True)
        return anim

####################################################################################################################


def cost(X, y, W):
    m = X.shape[0]
    sqr_error = (np.dot(X, W) - y) ** 2
    return sqr_error.sum() / (2 * m)


def d_cost(X, y, W):  # gradient
    m = X.shape[0]
    error = np.dot(X, W) - y
    gradient = np.dot(X.T, error) / m
    return gradient

# batch GD
def batch_gradient_descent(X, y, W0, n=1000, alpha=0.01, epsilon=1e-4):
    xx = np.linspace(X.min(), X.max(), y.shape[0])
    history_ = {'epoch': [], 'cost': [], 'W': [W0], 'y': []}

    for epoch in range(n):
        gradient = d_cost(X, y, history_['W'][-1])
        W = history_['W'][-1] - alpha * gradient
        new_cost = cost(X, y, W)

        #length_of_gradient = np.linalg.norm(gradient, 2)
        #if length_of_gradient < epsilon:
        #    break
        if epoch > 10:
            stop_point = sum((abs(np.diff(history_['cost'][(epoch-10):epoch])) < epsilon))
            if stop_point == 9:
                print(epoch)
                break

        history_['epoch'].append(epoch)
        history_['cost'].append(new_cost)
        history_['W'].append(W)
        history_['y'].append(np.dot(xx, W[1]) + W[0])  # for graph

    frame = history_['epoch'][-1]
    return W, history_, frame

# SGD
def nextbatch(X, y, batch_size):
    for i in range(0, X.shape[0], batch_size):
        yield (X[i:i + batch_size], y[i:i + batch_size])


def shuffle(X, y):
    total = np.hstack([y.reshape(-1, 1), X])
    np.random.shuffle(total)
    X = total[:, 1:]
    y = total[:, 0]
    return X, y


def sgd(X, y, W0, n=1000, alpha=0.01, epsilon=1e-6, batch_size=100):
    X, y = shuffle(X, y)
    xx = np.linspace(X.min(), X.max(), y.shape[0])
    history_ = {'epoch': [], 'cost': [], 'W': [W0], 'y': []}

    for epoch in range(n):
        for (batchX, batchY) in nextbatch(X, y, batch_size):
            gradient = d_cost(batchX, batchY, history_['W'][-1])
            W = history_['W'][-1] - alpha * gradient
            new_cost = cost(batchX, batchY, W)

        length_of_gradient = np.linalg.norm(gradient, 2)
        if length_of_gradient < epsilon:
            print(epoch)
            break

        #if epoch > 10:
        #    stop_point = sum((abs(np.diff(history_['cost'][(epoch - 10):epoch])) < epsilon))
        #    if stop_point == 9:
        #        print(epoch)
        #        break

        history_['epoch'].append(epoch)
        history_['cost'].append(new_cost)
        history_['W'].append(W)
        history_['y'].append(np.dot(xx, W[1]) + W[0])  # for graph

    frame = history_['epoch'][-1]
    return W, history_, frame

# Momentum
def velocity(gradient, alpha, gamma, his_v):
    v = gamma * his_v[-1] + alpha * gradient
    return v

def momentum(X, y, W0, n=1000, alpha=0.01, gamma=0.9, epsilon=1e-6, batch_size=100):
    X, y = shuffle(X, y)
    xx = np.linspace(X.min(), X.max(), y.shape[0])
    init_v = np.zeros(shape=(X.shape[1],))
    history_ = {'epoch': [], 'cost': [], 'W': [W0], 'y': [], 'his_v': [init_v]}

    for epoch in range(n):
        for (batchX, batchY) in nextbatch(X, y, batch_size):
            gradient = d_cost(batchX, batchY, history_['W'][-1])
            v = velocity(gradient, alpha, gamma, his_v=history_['his_v'])
            W = history_['W'][-1] - v

            new_cost = cost(batchX, batchY, W)

        if epoch > 10:
            stop_point = sum((abs(np.diff(history_['cost'][(epoch - 10):epoch])) < epsilon))
            if stop_point == 9:
                print(epoch)
                break

        history_['his_v'].append(v)
        history_['epoch'].append(epoch)
        history_['cost'].append(new_cost)
        history_['W'].append(W)
        history_['y'].append(np.dot(xx, W[1]) + W[0])  # for graph

    frame = history_['epoch'][-1]
    return W, history_, frame


# NAG
def d_cost_NAG(X, y, W, gamma, his_v):
    m = X.shape[0]
    error = np.dot(X, (W - gamma * his_v[-1])) - y
    gradient = np.dot(X.T, error) / m
    return gradient

def NAG(X, y, W0, n=1000, alpha=0.01, gamma=0.9, epsilon=1e-6, batch_size=100):
    X, y = shuffle(X, y)
    xx = np.linspace(X.min(), X.max(), y.shape[0])
    init_v = np.zeros(shape=(X.shape[1],))
    history_ = {'epoch': [], 'cost': [], 'W': [W0], 'y': [], 'his_v': [init_v]}

    for epoch in range(n):
        for (batchX, batchY) in nextbatch(X, y, batch_size):
            gradient = d_cost_NAG(batchX, batchY, history_['W'][-1], gamma, his_v=history_['his_v'])
            v = velocity(gradient, alpha, gamma, his_v=history_['his_v'])
            W = history_['W'][-1] - v

            new_cost = cost(batchX, batchY, W)

        if epoch > 10:
            stop_point = sum((abs(np.diff(history_['cost'][(epoch - 10):epoch])) < epsilon))
            if stop_point == 9:
                print(epoch)
                break

        history_['his_v'].append(v)
        history_['epoch'].append(epoch)
        history_['cost'].append(new_cost)
        history_['W'].append(W)
        history_['y'].append(np.dot(xx, W[1]) + W[0])  # for graph

    frame = history_['epoch'][-1]
    return W, history_, frame


# Adagrad

def Adagrad(X, y, W0, n=1000, alpha=0.01, epsilon=1e-6, batch_size=100):
    X, y = shuffle(X, y)
    xx = np.linspace(X.min(), X.max(), y.shape[0])
    init_v = np.zeros(shape=(X.shape[1],))
    history_ = {'epoch': [], 'cost': [], 'W': [W0], 'y': [], 'his_G': [init_v]}

    for epoch in range(n):
        for (batchX, batchY) in nextbatch(X, y, batch_size):
            # grad parts
            gradient = d_cost(batchX, batchY, history_['W'][-1])
            G = history_['his_G'][-1] + gradient ** 2  # k차원 = 2
            # update parts
            #W = history_['W'][-1] - alpha / np.sqrt(G + epsilon) * gradient
            W = init_v
            for i in range(X.shape[1]):
                W[i] = history_['W'][-1][i] - alpha / np.sqrt(G[i] + epsilon) * gradient[i]

            new_cost = cost(batchX, batchY, W)

        if epoch > 10:
            stop_point = sum((abs(np.diff(history_['cost'][(epoch - 10):epoch])) < epsilon))
            if stop_point == 9:
                print(epoch)
                break

        history_['his_G'].append(G)
        history_['epoch'].append(epoch)
        history_['cost'].append(new_cost)
        history_['W'].append(W)
        history_['y'].append(np.dot(xx, W[1]) + W[0])  # for graph

    frame = history_['epoch'][-1]
    return W, history_, frame

# RMSProp
def RMS(x):
    # x is already squared
    return np.sqrt(np.mean(x))

def RMSProp(X, y, W0, n=1000, alpha=0.01, gamma=0.9, epsilon=1e-6, batch_size=100):
    X, y = shuffle(X, y)
    xx = np.linspace(X.min(), X.max(), y.shape[0])
    init_v = np.zeros(shape=(X.shape[1],))
    history_ = {'epoch': [], 'cost': [], 'W': [W0], 'y': [], 'his_G': [init_v]}

    for epoch in range(n):
        for (batchX, batchY) in nextbatch(X, y, batch_size):
            # grad parts
            gradient = d_cost(batchX, batchY, history_['W'][-1])
            G = history_['his_G'][-1] + gradient**2
            RMS_G = gamma * history_['his_G'][-1].mean() + (1 - gamma) * (gradient**2)

            # update
            # W = history_['W'][-1] - alpha / np.sqrt(RMS_G + epsilon) * gradient
            W = init_v
            for i in range(X.shape[1]):
                W[i] = history_['W'][-1][i] - alpha / np.sqrt(RMS_G[i] + epsilon) * gradient[i]

            new_cost = cost(batchX, batchY, W)

        if epoch > 10:
            stop_point = sum((abs(np.diff(history_['cost'][(epoch - 10):epoch])) < epsilon))
            if stop_point == 9:
                print(epoch)
                break

        history_['his_G'].append(G)
        history_['epoch'].append(epoch)
        history_['cost'].append(new_cost)
        history_['W'].append(W)
        history_['y'].append(np.dot(xx, W[1]) + W[0])  # for graph

    frame = history_['epoch'][-1]
    return W, history_, frame


def AdaDelta(X, y, W0, n=1000, gamma=0.9, epsilon=1e-8, batch_size=100):
    X, y = shuffle(X, y)
    xx = np.linspace(X.min(), X.max(), y.shape[0])
    init_v = np.zeros(shape=(X.shape[1],))
    history_ = {'epoch': [], 'cost': [], 'W': [W0], 'y': [], 'his_G': [init_v], 's': [init_v]}

    for epoch in range(n):
        for (batchX, batchY) in nextbatch(X, y, batch_size):
            # grad parts
            gradient = d_cost(batchX, batchY, history_['W'][-1])
            G = history_['his_G'][-1] + gradient ** 2
            RMS_G = gamma * history_['his_G'][-1].mean() + (1 - gamma) * (gradient ** 2)
            delta = np.sqrt(history_['s'][-1] + epsilon) / np.sqrt(RMS_G + epsilon) * gradient
            s = gamma * history_['s'][-1] + (1 - gamma) * (delta ** 2)
            # update
            # W = history_['W'][-1] - delta
            W = init_v
            for i in range(X.shape[1]):
                W[i] = history_['W'][-1][i] - delta[i]

            new_cost = cost(batchX, batchY, W)

        if epoch > 10:
            stop_point = sum((abs(np.diff(history_['cost'][(epoch - 10):epoch])) < epsilon))
            if stop_point == 9:
                print(epoch)
                break

        history_['his_G'].append(G)
        history_['s'].append(s)
        history_['epoch'].append(epoch)
        history_['cost'].append(new_cost)
        history_['W'].append(W)
        history_['y'].append(np.dot(xx, W[1]) + W[0])  # for graph

    frame = history_['epoch'][-1]
    return W, history_, frame


def Adam(X, y, W0, n=1000, alpha=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, batch_size=100):
    X, y = shuffle(X, y)
    xx = np.linspace(X.min(), X.max(), y.shape[0])
    init_v = np.zeros(shape=(X.shape[1],))
    history_ = {'epoch': [], 'cost': [], 'W': [W0], 'y': [], 'moment1': [init_v], 'moment2': [init_v]}

    for epoch in range(n):
        for (batchX, batchY) in nextbatch(X, y, batch_size):
            # grad parts
            gradient = d_cost(batchX, batchY, history_['W'][-1])
            mo1 = beta1 * history_['moment1'][-1] + (1 - beta1) * gradient
            mo2 = beta2 * history_['moment2'][-1] + (1 - beta2) * (gradient ** 2)
            mo_hat = mo1 / (1 - (beta1 ** (epoch + 1)))
            ve_hat = mo2 / (1 - (beta2 ** (epoch + 1)))
            # update
            W = init_v
            for i in range(X.shape[1]):
                W[i] = history_['W'][-1][i] - alpha / np.sqrt(ve_hat[i] + epsilon) * mo_hat[i]


            new_cost = cost(batchX, batchY, W)


        if epoch > 10:
            stop_point = sum((abs(np.diff(history_['cost'][(epoch - 10):epoch])) < epsilon))
            if stop_point == 9:
                print(epoch)
                break

        history_['moment1'].append(mo1)
        history_['moment2'].append(mo2)
        history_['epoch'].append(epoch)
        history_['cost'].append(new_cost)
        history_['W'].append(W)
        history_['y'].append(np.dot(xx, W[1]) + W[0])  # for graph

    frame = history_['epoch'][-1]
    return W, history_, frame
