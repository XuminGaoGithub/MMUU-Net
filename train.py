#! /usr/bin/env python
#-*-coding:utf-8-*-

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V
from visdom import Visdom
import cv2
import os
import numpy as np
from metrics import runningScore

#from time import time
import time

from networks.MMUU_Net import MMUU_Net

from framework import MyFrame
from loss import dice_bce_loss
from data import ImageFolder
from train_valid import valid_loss

import re

SHAPE = (512,512)
n_classes=1

#（1）加载训练集合
ROOT = 'dataset/train/'
print(os.listdir(ROOT+'image/'))

trainlist=[]
for filename in os.listdir(ROOT+'image/'):
    trainlist.append((filename.rstrip('.png')))#只保存名字，去除后缀.png
print("trainlist",trainlist)

NAME = 'log01_dink34'
NAME_MODEL='best'
BATCHSIZE_PER_CARD = 1

#网络选择，对应test.py也需要修改

solver = MyFrame(MMUU_Net, dice_bce_loss, 1e-3)

batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD

dataset = ImageFolder(trainlist,ROOT)


data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batchsize,
    #batch_size=4,
    shuffle=True,
    #shuffle=False,
    num_workers=4)

#（2）加载测试集
ROOT_valid = 'dataset/valid/'
print(os.listdir(ROOT_valid+'image/'))

validlist=[]
for filename in os.listdir(ROOT_valid+'image/'):
    validlist.append((filename.rstrip('.png')))#只保存名字，去除后缀.png
print("validlist",validlist)

dataset_valid = ImageFolder(validlist,ROOT_valid)
data_loader_valid = torch.utils.data.DataLoader(
    dataset_valid,
    batch_size=batchsize,
    #batch_size=4,
    shuffle=True,
    #shuffle=False,
    num_workers=4)


mylog = open('logs/'+NAME+'.log','w')
tic = time.time()
no_optim = 0
total_epoch = 2000
train_epoch_best_loss = 100.

# 将窗口类实例化
viz = Visdom()
# 创建窗口并初始化
viz.line([0.], [0], win='train loss', opts=dict(title='train loss'))
viz.line([0.], [0], win='valid loss', opts=dict(title='valid loss'))
viz.line([0.], [0], win='mIoU', opts=dict(title='mIoU'))

grip_epoch=10#没十次绘制一次
running_metrics = runningScore(n_classes) #2类

# (3)开始训练
for epoch in range(1, total_epoch + 1):
    data_loader_iter = iter(data_loader)
    train_epoch_loss = 0

    #--训练--
    b=0
    for img, mask,mask1,mask2,mask3,mask4,mask5 in data_loader_iter:
        solver.set_input(img, mask,mask1,mask2,mask3,mask4,mask5)
        train_loss = solver.optimize_MLOSS()

        train_epoch_loss += train_loss
        b=b+1
        print('batch,train_loss',b,train_loss)
        #print("batch",b)
    train_epoch_loss /= len(data_loader_iter)
    #print('len(data_loader_iter)',len(data_loader_iter))
    print ( mylog, '********')
    print ( mylog, 'epoch:',epoch,'    time:',int(time.time()-tic))

    print ('********')
    print ('epoch:',epoch,'    time:',int(time.time()-tic))
    print ('train_epoch_loss:',train_epoch_loss)
    #print ('SHAPE:',SHAPE)
    solver.save('weights/'+NAME_MODEL+'.pth')

    if epoch%100 == 0:
        solver.save('weights/' + NAME_MODEL + str(epoch) + '.pth')

    if epoch % 100 == 0 and epoch > 0:
        solver.update_lr(5.0, factor=True, mylog=mylog)

    if epoch%grip_epoch == 0:
        # --测试--
        source = 'dataset/valid/'
        valid_process = valid_loss(source, dice_bce_loss)
        valid_epoch_loss, MIoU = valid_process.valid_func()

        viz.line([train_epoch_loss], [epoch], win='train loss', update='append')
        viz.line([valid_epoch_loss], [epoch], win='valid loss', update='append')
        viz.line([MIoU], [epoch], win='mIoU', update='append')
        # time.sleep(0.5)

print ( mylog, 'Finish!')
print ('Finish!')
mylog.close()
