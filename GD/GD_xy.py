# -*- coding: utf-8 -*-


import numpy as np
from matplotlib import animation, rc
import matplotlib.pylab as plt
rc('animation', html='html5')

def pltcost(history_):
    plt.plot(history_['epoch'], history_['cost'])
    plt.title('Cost', fontsize=20)
    plt.xlabel('epoch')
    plt.ylabel('Cost')
    plt.show()

class AniPlot(object):
    def __init__(self, title, frames, interval):
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.scat = self.ax.scatter([], [], s=20)
        self.title = title
        self.frames = frames
        self.interval = interval

    def set_data(self, history_):
        xx = np.linspace(-4, 4, 800)
        yy = np.linspace(-3, 3, 600)
        self.X, self.Y = np.meshgrid(xx, yy)
        self.Z = func(self.X, self.Y)
        self.x = history_['x']
        self.y = history_['y']

    def ani_init(self):
        self.scat.set_offsets([])
        return self.scat,

    def ani_update(self, i):
        data = np.hstack((np.array(self.x)[:i, np.newaxis], np.array(self.y)[:i, np.newaxis]))
        self.scat.set_offsets(data)
        return self.scat,

    def animate(self):
        plt.contour(self.X, self.Y, self.Z, colors="gray", levels=[0.4, 3, 15, 50, 150, 500, 1500, 5000])
        plt.plot(1, 1, 'ro', markersize=15)
        plt.title(self.title, fontsize=20)
        plt.xlim(-4, 4)
        plt.ylim(-3, 3)
        plt.xlabel('x')
        plt.ylabel('y')
        anim = animation.FuncAnimation(self.fig, self.ani_update, init_func=self.ani_init,
                                       frames=self.frames, interval=self.interval, blit=True)
        return anim

####################################################################################################################

def func(x, y):
    return (1 - x)**2 + 100.0 * (y - x**2)**2

def d_func(x, y):
    """gradient of function(x, y)"""
    return np.array((2.0 * (x - 1) - 400.0 * x * (y - x**2), 200.0 * (y - x**2)))


#SGD
def sgd(x, y, n=10000, alpha=0.0001, epsilon=1e-6):
    history_ = {'epoch': [], 'cost': [], 'x': [x], 'y': [y]}

    for epoch in range(n):
        gradient = d_func(x, y)
        x = x - alpha * gradient[0]
        y = x - alpha * gradient[1]
        new_cost = func(x, y)


        #length_of_gradient = np.linalg.norm(gradient, 2)
        #if length_of_gradient < epsilon:
        #   print(epoch)
        #   break

        if epoch > 10:
            stop_point = sum((abs(np.diff(history_['cost'][(epoch-10):epoch])) < epsilon))
            if stop_point == 9:
                print(epoch)
                break

        history_['epoch'].append(epoch)
        history_['cost'].append(new_cost)
        history_['x'].append(x)  # for graph
        history_['y'].append(y)  # for graph

    result = (x, y)
    frame = history_['epoch'][-1]
    return result, history_, frame


# Momentum

def velocity(gradient, alpha, gamma, his_v):
    v_x = gamma * his_v[0] + alpha * gradient[0]
    v_y = gamma * his_v[1] + alpha * gradient[1]
    return np.array((v_x, v_y))

def momentum(x, y, n=10000, alpha=0.0001, gamma=0.9, epsilon=1e-6):
    init_v = np.zeros(shape=(2,))
    history_ = {'epoch': [], 'cost': [], 'x': [x], 'y': [y], 'his_v': [init_v]}

    for epoch in range(n):
        gradient = d_func(x, y)
        v = velocity(gradient, alpha, gamma, history_['his_v'][-1])
        x = x - v[0]
        y = y - v[1]

        new_cost = func(x, y)


        if epoch > 10:
            stop_point = sum((abs(np.diff(history_['cost'][(epoch - 10):epoch])) < epsilon))
            if stop_point == 9:
                print(epoch)
                break

        history_['his_v'].append(v)
        history_['epoch'].append(epoch)
        history_['cost'].append(new_cost)
        history_['x'].append(x)  # for graph
        history_['y'].append(y)  # for graph

    result = (x, y)
    frame = history_['epoch'][-1]
    return result, history_, frame


# NAG
def d_func_NAG(x, y, gamma, his_v):
    x = x - gamma * his_v[0]
    y = y - gamma * his_v[1]
    return np.array((2.0 * (x - 1) - 400.0 * x * (y - x**2), 200.0 * (y - x**2)))

def NAG(x, y, n=10000, alpha=0.0001, gamma=0.9, epsilon=1e-6):
    init_v = np.zeros(shape=(2,))
    history_ = {'epoch': [], 'cost': [], 'x': [x], 'y': [y], 'his_v': [init_v]}

    for epoch in range(n):
        gradient = d_func_NAG(x, y, gamma, history_['his_v'][-1])
        v = velocity(gradient, alpha, gamma, history_['his_v'][-1])
        x = x - v[0]
        y = x - v[1]

        new_cost = func(x, y)


        if epoch > 10:
            stop_point = sum((abs(np.diff(history_['cost'][(epoch - 10):epoch])) < epsilon))
            if stop_point == 9:
                print(epoch)
                break

        history_['his_v'].append(v)
        history_['epoch'].append(epoch)
        history_['cost'].append(new_cost)
        history_['x'].append(x)  # for graph
        history_['y'].append(y)  # for graph

    result = (x, y)
    frame = history_['epoch'][-1]
    return result, history_, frame


# Adagrad
def Adagrad(x, y, n=10000, alpha=0.0001, epsilon=1e-6):
    init_v = np.zeros(shape=(2,))
    history_ = {'epoch': [], 'cost': [], 'x': [x], 'y': [y], 'his_G': [init_v]}

    for epoch in range(n):
        # grad parts
        gradient = d_func(x, y)
        G = history_['his_G'][-1] + gradient ** 2

        # update parts
        x = x - (alpha / np.sqrt(G[0] + epsilon)) * gradient[0]
        y = y - (alpha / np.sqrt(G[1] + epsilon)) * gradient[1]

        new_cost = func(x, y)

        if epoch > 10:
            stop_point = sum((abs(np.diff(history_['cost'][(epoch - 10):epoch])) < epsilon))
            if stop_point == 9:
                print(epoch)
                break

        history_['his_G'].append(G)
        history_['epoch'].append(epoch)
        history_['cost'].append(new_cost)
        history_['x'].append(x)  # for graph
        history_['y'].append(y)  # for graph

    result = (x, y)
    frame = history_['epoch'][-1]
    return result, history_, frame

# RMSProp
def RMSProp(x, y, n=10000, alpha=0.0001, gamma=0.9, epsilon=1e-6):
    init_v = np.zeros(shape=(2,))
    history_ = {'epoch': [], 'cost': [], 'x': [x], 'y': [y], 'his_G': [0]}
    G = 0
    for epoch in range(n):
        # grad parts
        gradient = d_func(x, y)
        G += (gradient**2).mean()
        RMS_G = gamma * history_['his_G'][-1] + (1 - gamma) * (gradient**2)

        # update parts
        x = x - (alpha / np.sqrt(RMS_G[0] + epsilon)) * gradient[0]
        y = y - (alpha / np.sqrt(RMS_G[1] + epsilon)) * gradient[1]

        new_cost = func(x, y)

        if epoch > 10:
            stop_point = sum((abs(np.diff(history_['cost'][(epoch - 10):epoch])) < epsilon))
            if stop_point == 9:
                print(epoch)
                break

        history_['his_G'].append(G)
        history_['epoch'].append(epoch)
        history_['cost'].append(new_cost)
        history_['x'].append(x)  # for graph
        history_['y'].append(y)  # for graph

    result = (x, y)
    frame = history_['epoch'][-1]
    return result, history_, frame


# AdaDelta
def AdaDelta(x, y, n=10000, gamma=0.9, epsilon=1e-8):
    min_cost = np.inf
    init_v = np.zeros(shape=(2,))
    history_ = {'epoch': [], 'cost': [], 'x': [x], 'y': [y],
                'his_G': [init_v], 's': [init_v]}

    for epoch in range(n):
        # grad parts
        gradient = d_func(x, y)
        G = history_['his_G'][-1] + gradient ** 2
        RMS_G = gamma * history_['his_G'][-1].mean() + (1 - gamma) * (gradient ** 2)
        x_delta = (np.sqrt(history_['s'][-1][0] + epsilon) / np.sqrt(RMS_G[0] + epsilon)) * gradient[0]
        y_delta = (np.sqrt(history_['s'][-1][1] + epsilon) / np.sqrt(RMS_G[1] + epsilon)) * gradient[1]
        x_s = gamma * history_['s'][-1][0] + (1 - gamma) * (x_delta ** 2)
        y_s = gamma * history_['s'][-1][1] + (1 - gamma) * (y_delta ** 2)
        # update parts
        x = x - x_delta
        y = y - y_delta

        new_cost = func(x, y)
        if new_cost < min_cost:
            min_cost = new_cost

        if epoch > 10:
            stop_point = sum((abs(np.diff(history_['cost'][(epoch - 10):epoch])) < epsilon))
            if stop_point == 9:
                print(epoch)
                break

        history_['his_G'].append(G)
        history_['s'].append(np.array((x_s, y_s)))
        history_['epoch'].append(epoch)
        history_['cost'].append(new_cost)
        history_['x'].append(x)  # for graph
        history_['y'].append(y)  # for graph

    result = (x, y)
    frame = history_['epoch'][-1]
    return result, history_, frame

# Adam
def Adam(x, y, n=10000, alpha=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    min_cost = np.inf
    init_v = np.zeros(shape=(2,))
    history_ = {'epoch': [], 'cost': [], 'x': [x], 'y': [y],
                'moment1': [init_v], 'moment2': [init_v]}

    for epoch in range(n):
        # grad parts
        gradient = d_func(x, y)
        mo1 = beta1 * history_['moment1'][-1] + (1 - beta1) * gradient
        mo2 = beta2 * history_['moment2'][-1] + (1 - beta2) * (gradient ** 2)
        mo_hat = mo1 / (1 - (beta1 ** (epoch + 1)))
        ve_hat = mo2 / (1 - (beta2 ** (epoch + 1)))
        # update
        x = x - (alpha / np.sqrt(ve_hat[0] + epsilon)) * mo_hat[0]
        y = y - (alpha / np.sqrt(ve_hat[1] + epsilon)) * mo_hat[1]

        new_cost = func(x, y)
        if new_cost < min_cost:
            min_cost = new_cost

        if epoch > 10:
            stop_point = sum((abs(np.diff(history_['cost'][(epoch - 10):epoch])) < epsilon))
            if stop_point == 9:
                print(epoch)
                break

        history_['moment1'].append(mo1)
        history_['moment2'].append(mo2)
        history_['epoch'].append(epoch)
        history_['cost'].append(new_cost)
        history_['x'].append(x)  # for graph
        history_['y'].append(y)  # for graph

    result = (x, y)
    frame = history_['epoch'][-1]
    return result, history_, frame
