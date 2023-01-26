import time
import numpy as np
from tqdm import tqdm
from time import sleep
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import math

from mpl_toolkits.mplot3d import Axes3D
import itertools

import torch
from torch import device, nn, reshape, functional, softmax
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torch.autograd import Variable
import os
import scipy.io as io


def marsagliasample(n_point, radius):
    shape = []
    for i in range(n_point):
        while 1:
            u = np.random.uniform(-radius, radius, 1)
            v = np.random.uniform(-radius, radius, 1)
            r = u ** 2 + v ** 2
            if r >= radius:
                continue
            else:
                break

        x = 2 * u * math.sqrt(radius - r)
        y = 2 * v * math.sqrt(radius - r)
        z = radius - 2 * r
        bnd = np.concatenate([x, y, z])
        shape.append(bnd.tolist())

    shape = np.array(shape)
    return shape


def marsagliasample2(n_point, radius):
    shape = []
    for i in range(n_point):
        while 1:
            u = np.random.uniform(-1, 1, 1)
            v = np.random.uniform(-1, 1, 1)
            r = u ** 2 + v ** 2
            if r >= 1:
                continue
            else:
                break

        x = 2 * u * math.sqrt(1 - r)
        y = 2 * v * math.sqrt(1 - r)
        z = 1 - 2 * r

        r2 = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arctan2(np.sqrt(x**2 + y**2), z)
        phi = np.arctan2(y, x)

        x2 = r2*radius*np.sin(theta)*np.cos(phi)
        y2 = r2*radius*np.sin(theta)*np.sin(phi)
        z2 = r2*radius*np.cos(theta)

        bnd = np.concatenate([x2, y2, z2])
        shape.append(bnd.tolist())

    shape = np.array(shape)
    return shape


def blocksample(n_point, lenx, leny, lenz, angle):
    # front and back
    y_fb = np.linspace(-leny / 2, leny / 2, n_point)
    z_fb = np.linspace(-lenz / 2, lenz / 2, n_point)
    grid_fb = np.array(list(itertools.product(y_fb, z_fb)))
    x_front = np.repeat(lenx / 2, n_point**2)
    x_back = np.repeat(-lenx / 2, n_point**2)

    # left and right
    x_lr = np.linspace(-lenx / 2, lenx / 2, n_point)
    z_lr = np.linspace(-lenz / 2, lenz / 2, n_point)
    grid_lr = np.array(list(itertools.product(x_lr, z_lr)))
    y_left = np.repeat(-leny / 2, n_point**2)
    y_right = np.repeat(leny/ 2, n_point**2)

    # up and down
    x_ud = np.linspace(-lenx / 2, lenx / 2, n_point)
    y_ud = np.linspace(-leny / 2, leny / 2, n_point)
    grid_ud = np.array(list(itertools.product(x_ud, y_ud)))
    z_up = np.repeat(lenz / 2, n_point**2)
    z_down = np.repeat(-lenz / 2, n_point**2)

    x_point = []
    y_point = []
    z_point = []

    x_point.append(x_front.tolist())
    x_point.append(x_back.tolist())
    x_point.append(grid_lr.T[0].tolist())
    x_point.append(grid_lr.T[0].tolist())
    x_point.append(grid_ud.T[0].tolist())
    x_point.append(grid_ud.T[0].tolist())

    y_point.append(grid_fb.T[0])
    y_point.append(grid_fb.T[0])
    y_point.append(y_left.tolist())
    y_point.append(y_right.tolist())
    y_point.append(grid_ud.T[1].tolist())
    y_point.append(grid_ud.T[1].tolist())

    z_point.append(grid_fb.T[1])
    z_point.append(grid_fb.T[1])
    z_point.append(grid_lr.T[1])
    z_point.append(grid_lr.T[1])
    z_point.append(z_up.tolist())
    z_point.append(z_down.tolist())

    x_point = np.array(x_point).reshape(-1)
    y_point = np.array(y_point).reshape(-1)
    z_point = np.array(z_point).reshape(-1)
    r = np.sqrt(x_point ** 2 + y_point ** 2)

    phi = np.arctan2(x_point, y_point)
    phi_r = phi - angle * np.pi / 180
    x_point_r = r * np.sin(phi_r)
    y_point_r = r * np.cos(phi_r)

    shape = np.dstack((x_point_r, y_point_r, z_point)).reshape(-1, 3)

    return shape


# sample the blocks uniformly and randomly
def blocksample2(n_point, x, y, z, angle):

    # allocate points for each surface
    n_S1 = np.around(n_point*x*y/(x*y+x*z+y*z)/2)
    n_S2 = np.around(n_point*x*z/(x*y+x*z+y*z)/2)
    n_S3 = n_point/2 - n_S1 - n_S2
    a = []
    b = []
    c = []

    # up
    for i in range(n_S1.astype(int)):
        a.append(np.random.uniform(-x/2, x/2))
        b.append(np.random.uniform(-y/2, y/2))
        c.append(z/2)

    # down
    for i in range(n_S1.astype(int)):
        a.append(np.random.uniform(-x/2, x/2))
        b.append(np.random.uniform(-y/2, y/2))
        c.append(-z/2)

    # front
    for i in range(n_S2.astype(int)):
        a.append(np.random.uniform(-x/2, x/2))
        b.append(y/2)
        c.append(np.random.uniform(-z/2, z/2))

    # back
    for i in range(n_S2.astype(int)):
        a.append(np.random.uniform(-x/2, x/2))
        b.append(-y/2)
        c.append(np.random.uniform(-z/2, z/2))

    # left
    for i in range(n_S3.astype(int)):
        a.append(-x/2)
        b.append(np.random.uniform(-y/2, y/2))
        c.append(np.random.uniform(-z/2, z/2))

    # right
    for i in range(n_S3.astype(int)):
        a.append(x/2)
        b.append(np.random.uniform(-y/2, y/2))
        c.append(np.random.uniform(-z/2, z/2))

    x_point = np.array(a).reshape(-1)
    y_point = np.array(b).reshape(-1)
    z_point = np.array(c).reshape(-1)
    r = np.sqrt(x_point ** 2 + y_point ** 2)

    phi = np.arctan2(x_point, y_point)
    phi_r = phi - angle * np.pi / 180
    x_point_r = r * np.sin(phi_r)
    y_point_r = r * np.cos(phi_r)

    shape = np.dstack((x_point_r, y_point_r, z_point)).reshape(-1, 3)

    return shape


# sample the blocks uniformly and randomly on both surface and edge

def blocksample3(n_point, x, y, z, angle):
    # allocate points for each surface
    n_point_e = np.around(n_point*0.4)
    n_point_s = np.around(n_point*0.6)
    n_E1 = np.around(n_point_e * x / (x + y + z) / 4).astype(int)
    n_E2 = np.around(n_point_e * y / (x + y + z) / 4).astype(int)
    n_E3 = (n_point_e/4 - n_E1 - n_E2).astype(int)
    n_S1 = np.around(n_point_s*x*y/(x*y+x*z+y*z)/2)
    n_S2 = np.around(n_point_s*x*z/(x*y+x*z+y*z)/2)
    n_S3 = n_point_s/2 - n_S1 - n_S2
    a = []
    b = []
    c = []


    l1 = np.linspace(-x / 2, x / 2, n_E1)

    for i in range(n_E1.astype(int)):
        a.append(l1[i])
        b.append(y/2)
        c.append(z/2)
    #
    for i in range(n_E1.astype(int)):
        a.append(l1[i])
        b.append(-y/2)
        c.append(z/2)

    for i in range(n_E1.astype(int)):
        a.append(l1[i])
        b.append(y / 2)
        c.append(-z / 2)

    for i in range(n_E1.astype(int)):
        a.append(l1[i])
        b.append(-y / 2)
        c.append(-z / 2)

    w1 = np.linspace(-y / 2, y / 2, n_E2)

    # width
    for i in range(n_E2.astype(int)):
        a.append(x/2)
        b.append(w1[i])
        c.append(z / 2)

    for i in range(n_E2.astype(int)):
        a.append(-x / 2)
        b.append(w1[i])
        c.append(z / 2)

    for i in range(n_E2.astype(int)):
        a.append(x/2)
        b.append(w1[i])
        c.append(-z / 2)

    for i in range(n_E2.astype(int)):
        a.append(-x/2)
        b.append(w1[i])
        c.append(-z / 2)

    h1 = np.linspace(-z / 2, z / 2, n_E3)

    # hight
    for i in range(n_E3.astype(int)):
        a.append(x / 2)
        b.append(y / 2)
        c.append(h1[i])

    for i in range(n_E3.astype(int)):
        a.append(-x / 2)
        b.append(y / 2)
        c.append(h1[i])

    for i in range(n_E3.astype(int)):
        a.append(x / 2)
        b.append(-y / 2)
        c.append(h1[i])

    for i in range(n_E3.astype(int)):
        a.append(-x / 2)
        b.append(-y / 2)
        c.append(h1[i])

    # up
    for i in range(n_S1.astype(int)):
        a.append(np.random.uniform(-x/2, x/2))
        b.append(np.random.uniform(-y/2, y/2))
        c.append(z/2)

    # down
    for i in range(n_S1.astype(int)):
        a.append(np.random.uniform(-x/2, x/2))
        b.append(np.random.uniform(-y/2, y/2))
        c.append(-z/2)

    # front
    for i in range(n_S2.astype(int)):
        a.append(np.random.uniform(-x/2, x/2))
        b.append(y/2)
        c.append(np.random.uniform(-z/2, z/2))

    # back
    for i in range(n_S2.astype(int)):
        a.append(np.random.uniform(-x/2, x/2))
        b.append(-y/2)
        c.append(np.random.uniform(-z/2, z/2))

    # left
    for i in range(n_S3.astype(int)):
        a.append(-x/2)
        b.append(np.random.uniform(-y/2, y/2))
        c.append(np.random.uniform(-z/2, z/2))

    # right
    for i in range(n_S3.astype(int)):
        a.append(x/2)
        b.append(np.random.uniform(-y/2, y/2))
        c.append(np.random.uniform(-z/2, z/2))

    x_point = np.array(a).reshape(-1)
    y_point = np.array(b).reshape(-1)
    z_point = np.array(c).reshape(-1)
    r = np.sqrt(x_point ** 2 + y_point ** 2)

    phi = np.arctan2(x_point, y_point)
    phi_r = phi - angle * np.pi / 180
    x_point_r = r * np.sin(phi_r)
    y_point_r = r * np.cos(phi_r)

    shape = np.dstack((x_point_r, y_point_r, z_point)).reshape(-1, 3)

    return shape

def getNoise(x_V, SNR):
    tar_V = np.zeros(x_V.shape)
    Noise = np.random.randn(x_V.shape[0], x_V.shape[1])
    Noise = Noise - np.mean(Noise)
    sig_pow = 1/len(x_V)*np.sum(np.multiply(x_V, x_V))
    Noise_vari = sig_pow/(10**(SNR/10))
    Noise = np.sqrt(Noise_vari)/np.std(Noise)*Noise
    tar_V = x_V + Noise
    return tar_V


def plotloss(train_loss_plot, val_loss_plot, xlabel, ylabel, title):
    train_loss_plot = np.reshape(train_loss_plot, [-1, 1])
    val_loss_plot = np.reshape(val_loss_plot, [-1, 1])
    plt.figure()
    plt.plot(train_loss_plot)
    plt.plot(val_loss_plot)
    plt.semilogy()
    plt.xlabel(f'{xlabel}')
    plt.ylabel(f'{ylabel}')
    # plt.ylim([1e-4, 1e-2])
    # plt.xlim([1e-9, 1e-2])
    plt.legend(['train_loss_curve', 'val_loss_curve'])
    plt.title(f'{title}')
    plt.grid(True)
    return

def movemeasurements(x_test, num):
    x_move = np.zeros((208, 1))
    b = x_test.reshape(208,1)
    for i in range(num):
        x_move[num-1-i] = b[-1-i]
    x_move[num:-1] = b[0:-num-1]
    return x_move

def switchmeasurements(x_test):
    x_move = np.zeros((208, 1))
    b = x_test.reshape(208,1)
    x_move[:104] = b[104:]
    x_move[104:] = b[:104]
    return x_move

def extrabnormal(v_exp, thre):
    a = np.where(v_exp >= thre)[0]
    for i in range(len(a)):
        v_exp[a[i]] = 0.5 * (v_exp[a[i] - 1] + v_exp[a[i] + 1])
    return v_exp

def testimg(n_sample, shape, init):
    a = np.random.randint(0, n_sample, 6)
    for i in range(1):
        idx = a[i]
        fig = plt.figure()
        ax = fig.add_subplot(3, 2, 1, projection='3d')
        ax.scatter(shape[idx][0].T[0], shape[idx][0].T[1], shape[idx][0].T[2])
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        ax.set_zlim([-1, 1])

        ax = fig.add_subplot(3, 2, 2, projection='3d')
        ax.scatter(init[idx][0].T[0], init[idx][0].T[1], init[idx][0].T[2])
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        ax.set_zlim([-1, 1])

        ax = fig.add_subplot(3, 2, 3, projection='3d')
        ax.scatter(shape[idx][1].T[0], shape[idx][1].T[1], shape[idx][1].T[2])
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        ax.set_zlim([-1, 1])

        ax = fig.add_subplot(3, 2, 4, projection='3d')
        ax.scatter(init[idx][1].T[0], init[idx][1].T[1], init[idx][1].T[2])
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        ax.set_zlim([-1, 1])

        ax = fig.add_subplot(3, 2, 5, projection='3d')
        ax.scatter(shape[idx][2].T[0], shape[idx][2].T[1], shape[idx][2].T[2])
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        ax.set_zlim([-1, 1])

        ax = fig.add_subplot(3, 2, 6, projection='3d')
        ax.scatter(init[idx][2].T[0], init[idx][2].T[1], init[idx][2].T[2])
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        ax.set_zlim([-1, 1])



# x = 1.4
# y = 1.4
# z = 5.2
# shape1 = blocksample2(1000, x, y, z, 0)
# shape2 = blocksample2(1000, 2.3, 5.2, 1.4, 0)
# shape3 = blocksample2(1000, 3.6, 3.6, 3.6, 0)
# shape4 = blocksample2(1000, 5.2, 1.4, 5.2, 0)
# shape5 = marsagliasample2(1000, 3)
# shape6 = marsagliasample2(1000, 2)
# shape7 = marsagliasample2(1000, 1)
# shape8 = marsagliasample2(1000, 0.8)



# fig = plt.figure(1)
# ax = Axes3D(fig)
# ax.scatter(shape1.T[0], shape1.T[1], shape1.T[2])
# ax.scatter(shape2.T[0]+6, shape2.T[1], shape2.T[2])
# ax.scatter(shape3.T[0], shape3.T[1]+6, shape3.T[2])
# ax.scatter(shape4.T[0]+6, shape4.T[1]+6, shape4.T[2])
# ax.scatter(shape5.T[0], shape5.T[1]+12, shape5.T[2])
# ax.scatter(shape6.T[0]+6, shape6.T[1]+12, shape6.T[2])
# ax.scatter(shape7.T[0]+0, shape7.T[1]+18, shape7.T[2])
# ax.scatter(shape8.T[0]+6, shape8.T[1]+18, shape8.T[2])
# plt.xlim([-5, 20])
# plt.ylim([-5, 20])
# plt.xlabel('x')
# plt.ylabel('y')
# ax.set_zlim([-5, 20])
# ax.title('x='+f'{x}'+'y='+f'{y}'+'z='+f'{z}')

# fig = plt.figure(2)
# ax = Axes3D(fig)
# # ax.scatter(shape1.T[0], shape1.T[1], shape1.T[2])
# # ax.scatter(shape2.T[0]+6, shape2.T[1], shape2.T[2])
# # ax.scatter(shape3.T[0], shape3.T[1]+6, shape3.T[2])
# # ax.scatter(shape4.T[0]+6, shape4.T[1]+6, shape4.T[2])
# ax.scatter(shape5.T[0], shape5.T[1], shape5.T[2])
# ax.scatter(shape6.T[0]+6, shape6.T[1], shape6.T[2])
# ax.scatter(shape7.T[0], shape7.T[1]+6, shape7.T[2])
# ax.scatter(shape8.T[0]+6, shape8.T[1]+6, shape8.T[2])
# plt.xlim([-5, 10])
# plt.ylim([-5, 10])
# plt.xlabel('x')
# plt.ylabel('y')
# ax.set_zlim([-5, 10])
# plt.show()

# np.savetxt('./matlab/sampling_1000.txt', shape)

