import time
import numpy as np
import prompt_toolkit.input.base
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
from function import marsagliasample, blocksample, marsagliasample2, blocksample2


def proidxmix(n_point, path, n_sample, max, min):
    img_position = []
    img_cdt = []
    img_shape = []
    for i in tqdm(range(n_sample)):
        index = f'{i + 1}'
        matr = io.loadmat(path + 'parameters' + index + '.mat')
        position = np.zeros((3, 3))
        cdt = np.zeros(3)
        shape = np.zeros((3, n_point ** 2 * 6, 3))
        # process block
        if matr['BlNum'] != 0:
            sort_blk = np.argsort(matr['BlksSizes'].sum(1))
            n_blk = len(sort_blk)
            for j in range(n_blk):
                position[j] = matr['BlksPositions'][sort_blk[n_blk - 1 - j]]
                cdt[j] = matr['BlksCon'][sort_blk[n_blk - 1 - j]]
                shape[j] = blocksample(n_point, matr['BlksSizes'][sort_blk[n_blk - 1 - j]][0],
                                       matr['BlksSizes'][sort_blk[n_blk - 1 - j]][1],
                                       matr['BlksSizes'][sort_blk[n_blk - 1 - j]][2],
                                       matr['BlksRotations'][sort_blk[n_blk - 1 - j]][2])/(max - min)
        else:
            n_blk = 0

        if matr['SpNum'] != 0:
            sort_sp = np.argsort(matr['SphsRadius'].reshape(-1))
            n_sp = len(sort_sp)
            for p in range(n_sp):
                position[p + n_blk] = matr['SphsPositions'][sort_sp[n_sp - 1 - p]]
                cdt[p + n_blk] = matr['SphsCon'][sort_sp[n_sp - 1 - p]]
                sph = marsagliasample2(n_point ** 2 * 6, matr['SphsRadius'][sort_sp[n_sp - 1 - p]])
                shape[p + n_blk] = sph/(max - min)

        img_position.append(position.tolist())
        img_cdt.append((cdt.tolist()))
        img_shape.append((shape.tolist()))

    img_position = np.array(img_position)
    img_cdt = np.array(img_cdt)
    img_shape = np.array(img_shape)

    return (img_position, img_cdt, img_shape)


# process mixed objects with uniform and random block sampling
def proidxmix2(n_point, path, n_sample, max, min):
    img_position = []
    img_cdt = []
    img_shape = []
    img_initial = []
    for i in tqdm(range(n_sample)):
        index = f'{i + 1}'
        matr = io.loadmat(path + 'parameters' + index + '.mat')
        position = np.zeros((3, 3))
        cdt = np.zeros(3)
        shape = np.zeros((3, n_point, 3))
        ini = np.zeros((3, n_point, 3))
        # process block
        if matr['BlNum'] != 0:
            sort_blk = np.argsort(matr['BlksSizes'].sum(1))
            n_blk = len(sort_blk)
            for j in range(n_blk):
                position[j] = matr['BlksPositions'][sort_blk[n_blk - 1 - j]]
                cdt[j] = matr['BlksCon'][sort_blk[n_blk - 1 - j]]
                shape[j] = blocksample2(n_point, matr['BlksSizes'][sort_blk[n_blk - 1 - j]][0],
                                       matr['BlksSizes'][sort_blk[n_blk - 1 - j]][1],
                                       matr['BlksSizes'][sort_blk[n_blk - 1 - j]][2],
                                       matr['BlksRotations'][sort_blk[n_blk - 1 - j]][2])/(max - min)

                ini[j] = blocksample2(n_point, 0.2, 0.2, 0.2, matr['BlksRotations'][sort_blk[n_blk - 1 - j]][2])
        else:
            n_blk = 0

        if matr['SpNum'] != 0:
            sort_sp = np.argsort(matr['SphsRadius'].reshape(-1))
            n_sp = len(sort_sp)
            for p in range(n_sp):
                position[p + n_blk] = matr['SphsPositions'][sort_sp[n_sp - 1 - p]]
                cdt[p + n_blk] = matr['SphsCon'][sort_sp[n_sp - 1 - p]]
                sph = marsagliasample2(n_point, matr['SphsRadius'][sort_sp[n_sp - 1 - p]])
                shape[p + n_blk] = sph/(max - min)
                ini[p + n_blk] = marsagliasample2(n_point, 0.2)

        img_position.append(position.tolist())
        img_cdt.append((cdt.tolist()))
        img_shape.append((shape.tolist()))
        img_initial.append((ini.tolist()))

    img_position = np.array(img_position)
    img_cdt = np.array(img_cdt)
    img_shape = np.array(img_shape)
    img_initial = np.array(img_initial)

    return (img_position, img_cdt, img_shape, img_initial)


def shapemask(n_point, path, n_sample):
    ones = np.ones((1, n_point, 3), dtype=np.int8)
    origin = np.zeros((3, n_point, 3), dtype=np.int8)
    mask = []
    for i in tqdm(range(n_sample)):
        index = f'{i + 1}'
        matr = io.loadmat(path + 'parameters' + index + '.mat')
        n_ob = matr['BlNum'] + matr['SpNum']
        for i in range(n_ob.item()):
            origin[i] = ones
        mask.append(origin.tolist())

    mask = np.array(mask)
    return mask


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


def saveandtest(name, filename, num, n_point, max, min):
    path_idx = "./raw_data/parameters/idx" + f'{name}' + "/"
    position_idx, cdt_idx, shape_idx, init_idx = proidxmix2(n_point, path_idx, num, max, min)
    mask = shapemask(n_point, path_idx, num)
    np.save('./Processed_data/' + f'{filename}' + '/position_idx' + f'{name}'+'.npy', position_idx)
    np.save('./Processed_data/' + f'{filename}' + '/cdt_idx'+f'{name}'+'.npy', cdt_idx)
    np.save('./Processed_data/' + f'{filename}' + '/shape_idx'+f'{name}'+'.npy', shape_idx)
    np.save('./Processed_data/' + f'{filename}' + '/init_idx' + f'{name}' + '.npy', init_idx)
    np.save('./Processed_data/' + f'{filename}' + '/shapemask_idx' + f'{name}' + '.npy', mask)
    testimg(num, shape_idx, init_idx)


saveandtest('1-2', 'random1000', 3000, 1000, 5.89, -5.89)
# saveandtest('3-4', 'random1000', 3000, 1000, 5.89, -5.89)
# saveandtest('5', 'random1000', 1552, 1000, 5.89, -5.89)
# saveandtest('6', 'random1000', 2001, 1000, 5.89, -5.89)
# saveandtest('7', 'random1000', 2000, 1000, 5.89, -5.89)
# saveandtest('8p1', 'random1000', 2589, 1000, 5.89, -5.89)
# saveandtest('8p2', 'random1000', 1931, 1000, 5.89, -5.89)
# saveandtest('9p1', 'random1000', 518, 1000, 5.89, -5.89)
# saveandtest('9p2', 'random1000', 1041, 1000, 5.89, -5.89)
# saveandtest('9p3', 'random1000', 838, 1000, 5.89, -5.89)
# saveandtest('9p4', 'random1000', 2665, 1000, 5.89, -5.89)
# plt.show()


# create dataset
# load the voltage measurements
# path1 = "D:/OneDrive - University of Edinburgh/2021-2022/BEng Project/3D/3Dmeasurement/"
# V_1 = np.load(path1 + 'v_Blocks_neg.npy')[:1400]    # [1400, 208] idx1
# V_2 = np.load(path1 + 'v_Blocks_neg.npy')[1400:]   # [1600, 208] idx2
# V_3 = np.load(path1 + 'v_Spheres_neg.npy')[:1400]  # [1400, 208] idx3
# V_4 = np.load(path1 + 'v_Spheres_neg.npy')[1400:]  # [1600, 208] idx4
# V_5 = np.load(path1 + 'v_1B1S_neg.npy')           # [1552, 208] idx5
# V_6 = np.load(path1 + 'v_2B1S_neg.npy')        # [2001, 208] idx6
# V_7 = np.load(path1 + 'v_1B2S_neg.npy')       # [2000, 208] idx7
# V_8 = np.load(path1 + 'v_2ob.npy')            # [4520, 208] idx8
# V_9 = np.load(path1 + 'v_3ob.npy')            # [5062, 208] idx9
#
# #
# # # load the shape ground truth
# path2 = "D:/OneDrive - University of Edinburgh/2021-2022/BEng Project/code/ThreeDTransformer/Processed_data/random100/"
# shape_1 = np.load(path2 + 'shape_idx1-2.npy')[:1400]
# shape_2 = np.load(path2 + 'shape_idx1-2.npy')[1400:]
# shape_3 = np.load(path2 + 'shape_idx3-4.npy')[:1400]
# shape_4 = np.load(path2 + 'shape_idx3-4.npy')[1400:]
# shape_5 = np.load(path2 + 'shape_idx5.npy')
# shape_6 = np.load(path2 + 'shape_idx6.npy')
# shape_7 = np.load(path2 + 'shape_idx7.npy')
# shape_8 = np.concatenate((np.load(path2 + 'shape_idx8p1.npy'), np.load(path2 + 'shape_idx8p2.npy')))
# shape_9 = np.concatenate((np.load(path2 + 'shape_idx9p1.npy'), np.load(path2 + 'shape_idx9p2.npy'), np.load(path2
#                                                                                                                +
#                                                                                                         'shape_idx9p3.npy'), np.load(path2 + 'shape_idx9p4.npy')))
#
# # load the shape mask code
# mask_1 = np.load(path2 + 'shapemask_idx1-2.npy')[:1400]
# mask_2 = np.load(path2 + 'shapemask_idx1-2.npy')[1400:]
# mask_3 = np.load(path2 + 'shapemask_idx3-4.npy')[:1400]
# mask_4 = np.load(path2 + 'shapemask_idx3-4.npy')[1400:]
# mask_5 = np.load(path2 + 'shapemask_idx5.npy')
# mask_6 = np.load(path2 + 'shapemask_idx6.npy')
# mask_7 = np.load(path2 + 'shapemask_idx7.npy')
# mask_8 = np.concatenate((np.load(path2 + 'shapemask_idx8p1.npy'), np.load(path2 + 'shapemask_idx8p2.npy')))
# mask_9 = np.concatenate((np.load(path2 + 'shapemask_idx9p1.npy'),  np.load(path2 + 'shapemask_idx9p2.npy'),
#                              np.load(path2 + 'shapemask_idx9p3.npy'),  np.load(path2 + 'shapemask_idx9p4.npy')))
#
#
# # load the shape initial points
# init_1 = np.load(path2 + 'init_idx1-2.npy')[:1400]
# init_2 = np.load(path2 + 'init_idx1-2.npy')[1400:]
# init_3 = np.load(path2 + 'init_idx3-4.npy')[:1400]
# init_4 = np.load(path2 + 'init_idx3-4.npy')[1400:]
# init_5 = np.load(path2 + 'init_idx5.npy')
# init_6 = np.load(path2 + 'init_idx6.npy')
# init_7 = np.load(path2 + 'init_idx7.npy')
# init_8 = np.concatenate((np.load(path2 + 'init_idx8p1.npy'), np.load(path2 + 'init_idx8p2.npy')))
# init_9 = np.concatenate((np.load(path2 + 'init_idx9p1.npy'),  np.load(path2 + 'init_idx9p2.npy'),
#                         np.load(path2 + 'init_idx9p3.npy'),  np.load(path2 + 'init_idx9p4.npy')))
#
# # load the position
# position_1 = np.load(path2 + 'position_idx1-2.npy')[:1400]
# position_2 = np.load(path2 + 'position_idx1-2.npy')[1400:]
# position_3 = np.load(path2 + 'position_idx3-4.npy')[:1400]
# position_4 = np.load(path2 + 'position_idx3-4.npy')[1400:]
# position_5 = np.load(path2 + 'position_idx5.npy')
# position_6 = np.load(path2 + 'position_idx6.npy')
# position_7 = np.load(path2 + 'position_idx7.npy')
# position_8 = np.concatenate((np.load(path2 + 'position_idx8p1.npy'), np.load(path2 + 'position_idx8p2.npy')))
# position_9 = np.concatenate((np.load(path2 + 'position_idx9p1.npy'),  np.load(path2 + 'position_idx9p2.npy'),
#                              np.load(path2 + 'position_idx9p3.npy'),  np.load(path2 + 'position_idx9p4.npy')))
#
# # load the conductivity
# cdt_1 = np.load(path2 + 'cdt_idx1-2.npy')[:1400]
# cdt_2 = np.load(path2 + 'cdt_idx1-2.npy')[1400:]
# cdt_3 = np.load(path2 + 'cdt_idx3-4.npy')[:1400]
# cdt_4 = np.load(path2 + 'cdt_idx3-4.npy')[1400:]
# cdt_5 = np.load(path2 + 'cdt_idx5.npy')
# cdt_6 = np.load(path2 + 'cdt_idx6.npy')
# cdt_7 = np.load(path2 + 'cdt_idx7.npy')
# cdt_8 = np.concatenate((np.load(path2 + 'cdt_idx8p1.npy'), np.load(path2 + 'cdt_idx8p2.npy')))
# cdt_9 = np.concatenate((np.load(path2 + 'cdt_idx9p1.npy'),  np.load(path2 + 'cdt_idx9p2.npy'),
#                              np.load(path2 + 'cdt_idx9p3.npy'),  np.load(path2 + 'cdt_idx9p4.npy')))
#
#
# # build the training and testing dataset
# x_train = np.concatenate((V_1[0:840], V_2[0:960], V_3[0:840], V_4[0:960], V_5[0:930], V_6[0:1200], V_7[0:1200],
#                           V_8[0:2712],
#                           V_9[0:3036]))
# y_train = np.concatenate((shape_1[0:840], shape_2[0:960], shape_3[0:840], shape_4[0:960], shape_5[0:930], shape_6[0:1200], shape_7[0:1200],
#                           shape_8[0:2712],
#                           shape_9[0:3036]))
# mask_train = np.concatenate((mask_1[0:840], mask_2[0:960], mask_3[0:840], mask_4[0:960], mask_5[0:930],
#                              mask_6[0:1200], mask_7[0:1200],
#                           mask_8[0:2712],
#                           mask_9[0:3036]))
# init_train = np.concatenate((init_1[0:840], init_2[0:960], init_3[0:840], init_4[0:960], init_5[0:930],
#                              init_6[0:1200], init_7[0:1200], init_8[0:2712], init_9[0:3036]))
# cdt_train = np.concatenate((cdt_1[0:840], cdt_2[0:960], cdt_3[0:840], cdt_4[0:960], cdt_5[0:930],
#                              cdt_6[0:1200], cdt_7[0:1200], cdt_8[0:2712], cdt_9[0:3036]))
# position_train = np.concatenate((position_1[0:840], position_2[0:960], position_3[0:840], position_4[0:960],
#                                  position_5[0:930], position_6[0:1200], position_7[0:1200], position_8[0:2712],
#                                  position_9[0:3036]))
#
#
# x_val = np.concatenate((V_1[840:1120], V_2[960:1280], V_3[840:1120], V_4[960:1280], V_5[930:1240], V_6[1200:1600],
#                         V_7[1200:1600],
#                         V_8[2712:3616], V_9[3036:4048]))
# y_val = np.concatenate((shape_1[840:1120], shape_2[960:1280], shape_3[840:1120], shape_4[960:1280], shape_5[930:1240], shape_6[1200:1600],
#                         shape_7[1200:1600], shape_8[2712:3616], shape_9[3036:4048]))
# mask_val = np.concatenate((mask_1[840:1120], mask_2[960:1280], mask_3[840:1120], mask_4[960:1280], mask_5[930:1240],
#                            mask_6[1200:1600], mask_7[1200:1600], mask_8[2712:3616], mask_9[3036:4048]))
# init_val = np.concatenate((init_1[840:1120], init_2[960:1280], init_3[840:1120], init_4[960:1280], init_5[930:1240],
#                         init_6[1200:1600], init_7[1200:1600], init_8[2712:3616], init_9[3036:4048]))
# cdt_val = np.concatenate((cdt_1[840:1120], cdt_2[960:1280], cdt_3[840:1120], cdt_4[960:1280], cdt_5[930:1240],
#                         cdt_6[1200:1600], cdt_7[1200:1600], cdt_8[2712:3616], cdt_9[3036:4048]))
# position_val = np.concatenate((position_1[840:1120], position_2[960:1280], position_3[840:1120], position_4[960:1280], position_5[930:1240],
#                         position_6[1200:1600], position_7[1200:1600], position_8[2712:3616], position_9[3036:4048]))
#
#
# x_test = np.concatenate((V_1[1120:], V_2[1280:], V_3[1120:], V_4[1280:], V_5[1240:], V_6[1600:], V_7[1600:], V_8[3616:],
#                          V_9[4048:]))
# y_test = np.concatenate((shape_1[1120:], shape_2[1280:], shape_3[1120:], shape_4[1280:], shape_5[1240:], shape_6[1600:], shape_7[1600:], shape_8[3616:],
#                          shape_9[4048: ]))
# mask_test = np.concatenate((mask_1[1120:], mask_2[1280:], mask_3[1120:], mask_4[1280:], mask_5[1240:], mask_6[1600:], mask_7[1600:],
#                             mask_8[3616:], mask_9[4048: ]))
# init_test = np.concatenate((init_1[1120:], init_2[1280:], init_3[1120:], init_4[1280:], init_5[1240:], init_6[1600:],
#                             init_7[1600:], init_8[3616:], init_9[4048:]))
# cdt_test = np.concatenate((cdt_1[1120:], cdt_2[1280:], cdt_3[1120:], cdt_4[1280:], cdt_5[1240:], cdt_6[1600:],
#                             cdt_7[1600:], cdt_8[3616:], cdt_9[4048:]))
# position_test = np.concatenate((position_1[1120:], position_2[1280:], position_3[1120:], position_4[1280:], position_5[1240:], position_6[1600:],
#                              position_7[1600:], position_8[3616:], position_9[4048:]))
#
#
# np.save('./Processed_data/dataset/random_200/x_train.npy', x_train)
# np.save('./Processed_data/dataset/random_200/y_train.npy', y_train)
# np.save('./Processed_data/dataset/random_200/mask_train.npy', mask_train)
# np.save('./Processed_data/dataset/random_200/init_train.npy', init_train)
# np.save('./Processed_data/dataset/random_200/pos_train.npy', position_train)
# np.save('./Processed_data/dataset/random_200/cdt_train.npy', cdt_train)
#
# np.save('./Processed_data/dataset/random_200/x_val.npy', x_val)
# np.save('./Processed_data/dataset/random_200/y_val.npy', y_val)
# np.save('./Processed_data/dataset/random_200/mask_val.npy', mask_val)
# np.save('./Processed_data/dataset/random_200/init_val.npy', init_val)
# np.save('./Processed_data/dataset/random_200/pos_val.npy', position_val)
# np.save('./Processed_data/dataset/random_200/cdt_val.npy', cdt_val)
#
# np.save('./Processed_data/dataset/random_200/x_test.npy', x_test)
# np.save('./Processed_data/dataset/random_200/y_test.npy', y_test)
# np.save('./Processed_data/dataset/random_200/mask_test.npy', mask_test)
# np.save('./Processed_data/dataset/random_200/init_test.npy', init_test)
# np.save('./Processed_data/dataset/random_200/pos_test.npy', position_test)
# np.save('./Processed_data/dataset/random_200/cdt_test.npy', cdt_test)





