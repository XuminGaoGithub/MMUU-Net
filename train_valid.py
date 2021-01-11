import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

from time import time
import os,shutil
from networks.MMUU_Net import MMUU_Net



from data import ImageFolder
from metrics import runningScore
from framework import MyFrame,MyFrame_valid
from loss import dice_bce_loss
import torch.nn.functional as F

n_classes=2

running_metrics = runningScore(n_classes) #2类

# 创建文件夹
def mkdir(path):
    # 引入模块
    import os

    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False
#python根据开头和结尾字符串获得指定字符串的中间字符串的代码
def GetMiddleStr(content,startStr,endStr):
    startIndex = content.index(startStr)
    if startIndex>=0:
        startIndex += len(startStr)
        endIndex = content.index(endStr)
    return content[startIndex:endIndex]


BATCHSIZE_PER_CARD = 4

class TTAFrame():
    def __init__(self, net):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        
    def test_one_img_from_path(self, path, evalmode = True):
        if evalmode:
            self.net.eval()
        batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
        if batchsize >= 8:
            return self.test_one_img_from_path_1(path)
        elif batchsize >= 4:
            return self.test_one_img_from_path_2(path)
        elif batchsize >= 2:
            return self.test_one_img_from_path_4(path)
        #elif batchsize == 0:
            #return self.test_one_img_from_path_4(path)



    def test_one_img_from_path_8(self, path):
        img = cv2.imread(path)#.transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.array(img1)[:,:,::-1]
        img4 = np.array(img2)[:,:,::-1]
        
        img1 = img1.transpose(0,3,1,2)
        img2 = img2.transpose(0,3,1,2)
        img3 = img3.transpose(0,3,1,2)
        img4 = img4.transpose(0,3,1,2)
        
        img1 = V(torch.Tensor(np.array(img1, np.float32)/255.0 * 3.2 -1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32)/255.0 * 3.2 -1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32)/255.0 * 3.2 -1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32)/255.0 * 3.2 -1.6).cuda())
        
        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()
        
        mask1 = maska + maskb[:,::-1] + maskc[:,:,::-1] + maskd[:,::-1,::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1,::-1]

        mask2_copy = mask1[0] + np.rot90(mask1[1])[::-1,::-1]

        #return mask3, mask3_copy
        
        return mask2,mask2_copy

    def test_one_img_from_path_4(self, path):
        img = cv2.imread(path)#.transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.array(img1)[:,:,::-1]
        img4 = np.array(img2)[:,:,::-1]
        
        img1 = img1.transpose(0,3,1,2)
        img2 = img2.transpose(0,3,1,2)
        img3 = img3.transpose(0,3,1,2)
        img4 = img4.transpose(0,3,1,2)
        
        img1 = V(torch.Tensor(np.array(img1, np.float32)/255.0 * 3.2 -1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32)/255.0 * 3.2 -1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32)/255.0 * 3.2 -1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32)/255.0 * 3.2 -1.6).cuda())
        
        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()
        
        mask1 = maska + maskb[:,::-1] + maskc[:,:,::-1] + maskd[:,::-1,::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1,::-1]
        mask2_copy = mask1[0] + np.rot90(mask1[1])[::-1, ::-1]
        
        return mask2,mask2_copy
    
    def test_one_img_from_path_2(self, path):
        img = cv2.imread(path)#.transpose(2,0,1)[None]

        #img = cv2.resize(img,(1792,1792),interpolation=cv2.INTER_CUBIC)#扩大预测

        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.concatenate([img1,img2])
        img4 = np.array(img3)[:,:,::-1]
        img5 = img3.transpose(0,3,1,2)
        img5 = np.array(img5, np.float32)/255.0 * 3.2 -1.6
        img5 = V(torch.Tensor(img5).cuda())
        img6 = img4.transpose(0,3,1,2)
        img6 = np.array(img6, np.float32)/255.0 * 3.2 -1.6
        img6 = V(torch.Tensor(img6).cuda())
        with torch.no_grad():  # RuntimeError: CUDA out of memory --https://www.jianshu.com/p/4a2be315b32a
                maska= self.net.forward(img5)[5].squeeze().cpu().data.numpy()#.squeeze(1)
                maskb = self.net.forward(img6)[5].squeeze().cpu().data.numpy()

                mask1 = maska + maskb[:,:,::-1]
                mask2 = mask1[:2] + mask1[2:,::-1]
                mask3 = mask2[0] + np.rot90(mask2[1])[::-1,::-1]
                #print('mask2.shape', mask2.shape)
                #print('mask2[0].shape', mask2[0].shape)
                #mask22=np.rot90(mask2[1])[::-1, ::-1]
                #print('mask22.shape', mask22.shape)
                mask3_copy = mask2[0] + np.rot90(mask2[1])[::-1, ::-1]

        return mask3,mask3_copy
    
    def test_one_img_from_path_1(self, path):
        img = cv2.imread(path)#.transpose(2,0,1)[None]
        
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.concatenate([img1,img2])
        img4 = np.array(img3)[:,:,::-1]
        img5 = np.concatenate([img3,img4]).transpose(0,3,1,2)
        img5 = np.array(img5, np.float32)/255.0 * 3.2 -1.6
        img5 = V(torch.Tensor(img5).cuda())
        
        mask = self.net.forward(img5).squeeze().cpu().data.numpy()#.squeeze(1)
        mask1 = mask[:4] + mask[4:,:,::-1]
        mask2 = mask1[:2] + mask1[2:,::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1,::-1]
        
        return mask3

    def test_one_img_from_path_0(self, path):
        self.net.eval()
        img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img = img.transpose(2,0,1)
        img = np.array(img, np.float32) / 255.0 * 3.2 - 1.6
        img = V(torch.Tensor(img).cuda())
        img = torch.unsqueeze(img, 0)#此处由于self.net = torch.nn.DataParallel，所以需要（batch_size,channel,h,w）
        #img=img.numpy()
        #img=np.transpose(0,3,1,2)
        #img=img.cuda().data.cpu().numpy()

        print('img.shape',img.shape)
        mask = self.net.forward(img).squeeze().cpu().data.numpy()  # .squeeze(1)
        #print(mask)
        #print('mask.size()',mask.shape)
        mask_copy = self.net.forward(img).squeeze().cpu().data.numpy()  # .squeeze(1)


        return mask,mask_copy


    def test_one_img_from_path_loss(self, path):
        self.net.eval()
        img = cv2.imread(path)  # .transpose(2,0,1)[None]



        img = img.transpose(2,0,1)
        img = np.array(img, np.float32) / 255.0 * 3.2 - 1.6
        img = V(torch.Tensor(img).cuda())
        img = torch.unsqueeze(img, 0)#此处由于self.net = torch.nn.DataParallel，所以需要（batch_size,channel,h,w）
        #img=img.numpy()
        #img=np.transpose(0,3,1,2)
        #img=img.cuda().data.cpu().numpy()

        #print(img.shape)
        with torch.no_grad():  # RuntimeError: CUDA out of memory --https://www.jianshu.com/p/4a2be315b32a

            start = time()
            mask_loss = self.net.forward(img)
            stop = time()
            print(str(stop - start) + "秒")
            #print('img.shape',img.shape)
            mask = mask_loss[5].squeeze().cpu().data.numpy()#x_out
            mask_copy = mask_loss[5].squeeze().cpu().data.numpy()#x_out


        '''
        #这段代码不严紧，会造成cuda out of memory
        mask = self.net.forward(img).squeeze().cpu().data.numpy()  # .squeeze(1)
        #print(mask)
        #print('mask.shape', mask.shape)
        mask_copy = self.net.forward(img).squeeze().cpu().data.numpy()  # .squeeze(1)
        '''


        return mask,mask_copy,mask_loss


    def load(self, path):
        #model = torch.load(path, map_location='cpu')
        #self.net = model.module  # 才是你的模型
        self.net.load_state_dict(torch.load(path),False)






# --测试(1)--直接测试写成类封装并计算测试loss(用于train阶段调用)--

class valid_loss():

    def __init__(self, path,loss):
        self.path = path
        self.loss = loss()

    def valid_func(self):
        ###(1)加载图片
        source = self.path
        val_img = os.listdir(source + 'image/')
        print("val_img", val_img)
        val_lbl = os.listdir(source + 'label/')
        print("val_lbl", val_lbl)

        ###(2)加载模型

        solver = TTAFrame(MMUU_Net)  

        NAME_MODEL = 'best'
        solver.load('weights/' + NAME_MODEL + '.pth')

        tic = time()
        target = 'submits/log01_dink34/'
        shutil.rmtree(target)  # 删除文件夹
        os.mkdir(target)

        valid_epoch_loss=0

        ###(3)开始测试
        for i, name in enumerate(val_img):

            if i % 10 == 0:
                print(i / 10, '    ', '%.2f' % (time() - tic))
            pred, pred_copy,pred_loss= solver.test_one_img_from_path_loss(source + 'image/' + name)

            #####（3）--1计算准确率
            # print('pred_copy', pred)
            # np.set_printoptions(threshold=np.inf)  # 打印所有数据
            shape = pred_copy.shape
            result = np.zeros(shape)
            for x in range(0, shape[0]):
                for y in range(0, shape[1]):
                    if pred_copy[x, y] > 0.5:
                        result[x, y] = 1
                    else:
                        result[x, y] = 0

            pred_result = np.array(result).astype(np.uint8)
            # print(pred_result)
            #print('img',source+'image/'+name)
            #print('img', source + 'label/' + name)

            #mask = cv2.imread(source + 'label/' + name, cv2.IMREAD_GRAYSCALE)# python根据开头和结尾字符串获得指定字符串的中间字符串的代码

            label_name = GetMiddleStr(name, "", ".")
            mask = cv2.imread(source + 'label/' + label_name+'_label.png', cv2.IMREAD_GRAYSCALE)  # 此时image与label下的 img和lbl命名格式为（x.png,x_label.png）
            #mask = cv2.imread(source + 'label/' + label_name + '.png', cv2.IMREAD_GRAYSCALE)

            mask = mask/float(255.0)
            mask[mask >= 0.5] = 1
            mask[mask <= 0.5] = 0
            mask = np.array(mask).astype(np.uint8)
            # print('mask.shape', mask.shape)
            # print('label', source+'label/'+name)

            output_label = pred_result
            # print('output_label', type(output_label))
            # print('output_label', output_label[100,100])
            gt = mask  # 这里需要乘以4
            # print('gt', type(gt))
            # print('gt', gt[100,100])
            running_metrics.update(gt, output_label)  # gt, output_label需要维度一致，数据类型为int

            pred[pred > 0.5] = 255
            pred[pred <= 0.5] = 0
            pred = np.concatenate([pred[:, :, None], pred[:, :, None], pred[:, :, None]], axis=2)
            # print(pred)
            cv2.imwrite(target + label_name+'.png', pred.astype(np.uint8))

            #####(3)--2计算loss
            label_name = GetMiddleStr(name, "", ".")
            mask_loss = cv2.imread(source + 'label/' + label_name + '_label.png',cv2.IMREAD_GRAYSCALE)  # 此时image与label下的 img和lbl命名格式为（x.png,x_label.png）
            #mask_loss = cv2.imread(source + 'label/' + label_name + '.png', cv2.IMREAD_GRAYSCALE)
            (h, w) = mask_loss.shape[:2]



            #print('mask_loss',mask_loss)
            #print('pred_loss', pred_loss)
            ###(1)mloss
            mask1 = mask_loss
            mask2 = cv2.resize(mask_loss, (h // 2, w // 2), interpolation=cv2.INTER_CUBIC)
            mask3 = cv2.resize(mask_loss, (h // 4, w // 4), interpolation=cv2.INTER_CUBIC)
            mask4 = cv2.resize(mask_loss, (h // 8, w // 8), interpolation=cv2.INTER_CUBIC)
            mask5 = cv2.resize(mask_loss, (h // 16, w // 16), interpolation=cv2.INTER_CUBIC)
            mask1 = np.expand_dims(mask1, axis=2)
            mask1 = np.array(mask1, np.float32).transpose(2, 0, 1) / 255.0
            mask1[mask1 >= 0.5] = 1
            mask1[mask1 <= 0.5] = 0
            mask1 = torch.Tensor(mask1)
            mask1 = V(mask1.cuda(), False)
            mask2 = np.expand_dims(mask2, axis=2)
            mask2 = np.array(mask2, np.float32).transpose(2, 0, 1) / 255.0
            mask2[mask2 >= 0.5] = 1
            mask2[mask2 <= 0.5] = 0
            mask2 = torch.Tensor(mask2)
            mask2 = V(mask2.cuda(), False)
            mask3 = np.expand_dims(mask3, axis=2)
            mask3 = np.array(mask3, np.float32).transpose(2, 0, 1) / 255.0
            mask3[mask3 >= 0.5] = 1
            mask3[mask3 <= 0.5] = 0
            mask3 = torch.Tensor(mask3)
            mask3 = V(mask3.cuda(), False)
            mask4 = np.expand_dims(mask4, axis=2)
            mask4 = np.array(mask4, np.float32).transpose(2, 0, 1) / 255.0
            mask4[mask4 >= 0.5] = 1
            mask4[mask4 <= 0.5] = 0
            mask4 = torch.Tensor(mask4)
            mask4 = V(mask4.cuda(), False)
            mask5 = np.expand_dims(mask5, axis=2)
            mask5 = np.array(mask5, np.float32).transpose(2, 0, 1) / 255.0
            mask5[mask5 >= 0.5] = 1
            mask5[mask5 <= 0.5] = 0
            mask5 = torch.Tensor(mask5)
            mask5 = V(mask5.cuda(), False)
            mask_loss = np.expand_dims(mask_loss, axis=2)
            mask_loss = np.array(mask_loss, np.float32).transpose(2, 0, 1) / 255.0
            mask_loss[mask_loss >= 0.5] = 1
            mask_loss[mask_loss <= 0.5] = 0
            mask_loss = torch.Tensor(mask_loss)
            mask_loss = V(mask_loss.cuda(), False)
            #self.masks=[mask1,mask2,mask3,mask4,mask5,mask_loss]
            self.masks = [mask1, mask2, mask3, mask4, mask5,mask1]
            loss = self.loss(self.masks[0], pred_loss[0])
            loss = loss + self.loss(self.masks[1], pred_loss[1])
            loss = loss + self.loss(self.masks[2], pred_loss[2])
            loss = loss + self.loss(self.masks[3], pred_loss[3])
            loss = loss + self.loss(self.masks[4], pred_loss[4])
            loss = loss + self.loss(self.masks[5], pred_loss[5])
            loss = loss / 6.0
            valid_loss = loss.item()
            valid_epoch_loss += valid_loss
            ###

        valid_epoch_loss /= len(val_img)
        print('valid_loss:', valid_epoch_loss)

        # 计算测试Imou
        score, class_iou = running_metrics.get_scores()
        MIoU = 0.0000
        i = 0
        for k, v in score.items():
            i = i + 1
            print("score.items", k, v)
            if i == 4:
                MIoU = v
        print('Mean IoU', MIoU)

        for i in range(n_classes):
            print("class_iou", i, class_iou[i])

        return valid_epoch_loss, MIoU


# --测试(3)--加入TTA--
'''
source = 'dataset/test/'
val_img = os.listdir(source+'image/')
print("val_img",val_img)
val_lbl = os.listdir(source+'label/')
print("val_lbl",val_lbl)

solver = TTAFrame(MMUU_Net)

NAME_MODEL='best'#有效
solver.load('weights/'+NAME_MODEL+'.pth')
tic = time()
target = 'submits/log01_dink34/'
shutil.rmtree(target)#删除文件夹
os.mkdir(target)

for i,name in enumerate(val_img):


    if i%10 == 0:
        print (i/10, '    ','%.2f'%(time()-tic))
    pred,pred_copy = solver.test_one_img_from_path(source + 'image/' + name)
    #print('pred_copy', pred)
    #np.set_printoptions(threshold=np.inf)  # 打印所有数据
    shape = pred_copy.shape
    result = np.zeros(shape)
    for x in range(0, shape[0]):
        for y in range(0, shape[1]):
            if pred_copy[x, y] > 4:
                result[x, y] = 1
            else:
                result[x, y] = 0

    pred_result = np.array(result).astype(np.uint8)
    #print(pred_result)
    #print('img',source+'image/'+name)

    
    label_name = GetMiddleStr(name, "", ".")
    mask = cv2.imread(source + 'label/' + label_name + '_label.png',cv2.IMREAD_GRAYSCALE)  # 此时image与label下的 img和lbl命名格式为（x.png,x_label.png）
    #mask = cv2.imread(source + 'label/' + label_name + '.png', cv2.IMREAD_GRAYSCALE)
    #mask = cv2.resize(mask, (1792, 1792), interpolation=cv2.INTER_CUBIC)#扩大预测


    mask=mask/255.0
    mask[mask >= 0.5] = 1
    mask[mask <= 0.5] = 0
    mask = np.array(mask).astype(np.uint8)
    #print('mask.shape', mask.shape)
    # print('label', source+'label/'+name)

    output_label = pred_result
    #print('output_label', type(output_label))
    #print('output_label', output_label[100,100])
    gt = mask #这里需要乘以4
    #print('gt', type(gt))
    #print('gt', gt[100,100])
    running_metrics.update(gt, output_label) #gt, output_label需要维度一致，数据类型为int


    pred[pred>4] = 255
    pred[pred<=4] = 0
    pred = np.concatenate([pred[:,:,None],pred[:,:,None],pred[:,:,None]],axis=2)
    #print(pred)
    cv2.imwrite(target + label_name+'pred.png',pred.astype(np.uint8))



# 计算测试Imou
score, class_iou = running_metrics.get_scores()
MIoU = 0.0000
i = 0
for k, v in score.items():
    i = i + 1
    print("score.items", k, v)
    if i == 4:
        MIoU = v
print('Mean IoU', MIoU)

for i in range(n_classes):
    print("class_iou", i, class_iou[i])
'''



