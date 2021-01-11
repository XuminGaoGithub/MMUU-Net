
import torch
import torch.utils.data as data
from torch.autograd import Variable as V

import cv2
import numpy as np
import os
import math,random


#opencv增强-- https://blog.csdn.net/weixin_44936889/article/details/103742580#12_random_resize_and_crop_35

class RandomErasing(object): #https://blog.csdn.net/qq_34291583/article/details/103302310
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''

    #def __init__(self, probability=0.5, sl=0.02, sh=0.05, r1=0.05, mean=[0.4914, 0.4822, 0.4465]):
    def __init__(self, probability=0.5, sl=0.02, sh=0.05, r1=0.1, mean=[99.9,99.9,99.9]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img, lbl):
        #print("img.shape:",img.shape)
        #print("lbl.shape:", lbl.shape)
        #print("lbl.ndim:", lbl.ndim)

        if random.uniform(0, 1) > self.probability:
            return img,lbl

        for attempt in range(100):
            area = img.shape[0] * img.shape[1]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.shape[1] and h < img.shape[0]:
                x1 = random.randint(0, img.shape[0] - h)
                y1 = random.randint(0, img.shape[1] - w)
                if img.shape[2] == 3:
                    img[x1:x1 + h, y1:y1 + w,0] = self.mean[0]
                    img[x1:x1 + h, y1:y1 + w,1] = self.mean[1]
                    img[x1:x1 + h, y1:y1 + w,2] = self.mean[2]
                else:
                    img[x1:x1 + h, y1:y1 + w,0] = self.mean[0]

                if lbl.ndim==3:
                        if lbl.shape[2] == 3:
                            lbl[x1:x1 + h, y1:y1 + w,0] = self.mean[0]
                            lbl[x1:x1 + h, y1:y1 + w,1] = self.mean[1]
                            lbl[x1:x1 + h, y1:y1 + w,2] = self.mean[2]
                else:
                    lbl[x1:x1 + h, y1:y1 + w] = self.mean[0]

                #print("img.shape:",img.shape)
                #print("lbl.shape:", lbl.shape)

                #cv2.imshow("img",img)
                #cv2.imshow("lbl", lbl)
                #cv2.imwrite("img.png",img)
                #cv2.imwrite("lbl.png", lbl)
                #cv2.waitKey(1000)
                return img,lbl

        return img,lbl

def random_crop_and_resize(image,lbl,size=512,u=0.5):

    #image = resize_image(image)

    h, w = image.shape[:2]


    y = np.random.randint(0, h-size)
    x = np.random.randint(0, w-size)

    if np.random.random() < u:
            image = image[y:y+size, x:x+size, :]
            lbl = lbl[y:y + size, x:x + size]
            image=resize_image(image,(h,w))
            lbl = resize_image(lbl, (h,w))

    #cv2.imshow("img",image)
    #cv2.imshow("lbl", lbl)
    #cv2.waitKey(1000)

    return image,lbl

def resize_image(image,size=(1024,1024), bias=5):


    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)

    return image



def PepperandSalt(src,lbl, percetage=0.2,u=0.5):
    NoiseImg_src = src
    NoiseImg_lbl = lbl
    NoiseNum = int(percetage * src.shape[0] * src.shape[1])

    if np.random.random() < u:
            for i in range(NoiseNum):
                randX = random.randint(0, src.shape[0] - 1)
                randY = random.randint(0, src.shape[1] - 1)
                if random.randint(0, 1) <= 0.5:
                    NoiseImg_src[randX, randY] = 0
                    NoiseImg_lbl[randX, randY] = 0
                else:
                    NoiseImg_src[randX, randY] = 255
                    NoiseImg_lbl[randX, randY] = 255

    cv2.imshow("img",NoiseImg_src)
    cv2.imshow("lbl", NoiseImg_lbl)
    cv2.waitKey(100)

    return NoiseImg_src,NoiseImg_lbl

means=0.2
sigma=0.3
def GaussianNoise(src,lbl, means=0.2, sigma=0.3,u=0.5):
    NoiseImg_src = src
    NoiseImg_lbl = lbl
    rows = NoiseImg_src.shape[0]
    cols = NoiseImg_lbl.shape[1]
    if np.random.random() < u:
            for i in range(rows):
                for j in range(cols):
                    NoiseImg_src[i, j] = NoiseImg_src[i, j] + random.gauss(means, sigma)
                    NoiseImg_lbl[i, j] = NoiseImg_lbl[i, j] + random.gauss(means, sigma)
                    if NoiseImg_src[i, j,0] < 0:
                        NoiseImg_src[i, j,0] = 0
                        NoiseImg_src[i, j,1] = 0
                        NoiseImg_src[i, j,2] = 0
                        NoiseImg_lbl[i, j] = 0

                    elif NoiseImg_src[i, j,0] > 255:
                        NoiseImg_src[i, j,0] = 255
                        NoiseImg_src[i, j, 1] = 255
                        NoiseImg_src[i, j, 2] = 255
                        NoiseImg_lbl[i, j] = 255
    cv2.imshow("img", NoiseImg_src)
    cv2.imshow("lbl", NoiseImg_lbl)
    cv2.waitKey(1000)

    return NoiseImg_src,NoiseImg_lbl





def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        #image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image

def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0), 
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask

def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)


    return image, mask

def randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)
    return image, mask

def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image=np.rot90(image)
        mask=np.rot90(mask)

    return image, mask

def default_loader(id, root):

    img = cv2.imread(os.path.join(root+'image/','{}.png').format(id))
    (h, w) = img.shape[:2]
    #img = cv2.resize(img,(256,256),interpolation=cv2.INTER_CUBIC)
    #print(os.path.join(root+'image/','{}.png').format(id))

    mask = cv2.imread(os.path.join(root+'label/','{}_label.png').format(id), cv2.IMREAD_GRAYSCALE)
    #mask = cv2.imread(os.path.join(root+'label/','{}.png').format(id), cv2.IMREAD_GRAYSCALE)
    #mask = cv2.resize(mask,(256,256),interpolation=cv2.INTER_CUBIC)
    #print(os.path.join(root+'label/','{}_label.png').format(id))



    img = randomHueSaturationValue(img,
                                   hue_shift_limit=(-30, 30),
                                   sat_shift_limit=(-5, 5),
                                   val_shift_limit=(-15, 15))

    #img, mask = random_crop_and_resize(img, mask, size=int(img.shape[0]/2))#效果不好不加入
    #img, mask = PepperandSalt(img, mask)
    #img, mask = GaussianNoise(img, mask)


    img, mask = randomShiftScaleRotate(img, mask,
                                       shift_limit=(-0.1, 0.1),
                                       scale_limit=(-0.1, 0.1),
                                       aspect_limit=(-0.1, 0.1),
                                       rotate_limit=(-0, 0))
    img, mask = randomHorizontalFlip(img, mask)
    img, mask = randomVerticleFlip(img, mask)
    img, mask = randomRotate90(img, mask)

    re=RandomErasing()#效果提升很高
    img, mask = re(img, mask)

    ###(1)mloss
    mask1=mask
    mask2=cv2.resize(mask,(h // 2,w // 2),interpolation=cv2.INTER_CUBIC)
    mask3=cv2.resize(mask,(h // 4,w // 4),interpolation=cv2.INTER_CUBIC)
    mask4=cv2.resize(mask,(h // 8,w // 8),interpolation=cv2.INTER_CUBIC)
    mask5=cv2.resize(mask,(h // 16,w // 16),interpolation=cv2.INTER_CUBIC)
    mask1 = np.expand_dims(mask1, axis=2)
    mask1 = np.array(mask1, np.float32).transpose(2, 0, 1) / 255.0
    mask1[mask1 >= 0.5] = 1
    mask1[mask1 <= 0.5] = 0
    mask2 = np.expand_dims(mask2, axis=2)
    mask2 = np.array(mask2, np.float32).transpose(2, 0, 1) / 255.0
    mask2[mask2 >= 0.5] = 1
    mask2[mask2 <= 0.5] = 0
    mask3 = np.expand_dims(mask3, axis=2)
    mask3 = np.array(mask3, np.float32).transpose(2, 0, 1) / 255.0
    mask3[mask3 >= 0.5] = 1
    mask3[mask3 <= 0.5] = 0
    mask4 = np.expand_dims(mask4, axis=2)
    mask4 = np.array(mask4, np.float32).transpose(2, 0, 1) / 255.0
    mask4[mask4 >= 0.5] = 1
    mask4[mask4 <= 0.5] = 0
    mask5 = np.expand_dims(mask5, axis=2)
    mask5 = np.array(mask5, np.float32).transpose(2, 0, 1) / 255.0
    mask5[mask5 >= 0.5] = 1
    mask5[mask5 <= 0.5] = 0
    ###


    mask = np.expand_dims(mask, axis=2)
    img = np.array(img, np.float32).transpose(2,0,1)/255.0 * 3.2 - 1.6
    mask = np.array(mask, np.float32).transpose(2,0,1)/255.0
    mask[mask>=0.5] = 1
    mask[mask<=0.5] = 0



    #mask = abs(mask-1)
    #print(mask.shape, mask2.shape,mask3.shape,mask4.shape)
    #return img, mask,mask1,mask2,mask3,mask4,mask5
    return img,mask, mask2, mask3, mask4,mask5, mask


def default_loader_valid(id, root):
    img = cv2.imread(os.path.join(root+'image/','{}.png').format(id))
    print(os.path.join(root+'image/','{}.png').format(id))
    mask = cv2.imread(os.path.join(root+'label/','{}_label.png').format(id), cv2.IMREAD_GRAYSCALE)
    print(os.path.join(root+'label/','{}.png').format(id))

    '''
    img = randomHueSaturationValue(img,
                                   hue_shift_limit=(-30, 30),
                                   sat_shift_limit=(-5, 5),
                                   val_shift_limit=(-15, 15))

    #img, mask = random_crop_and_resize(img, mask, size=int(img.shape[0]/2))#效果不好不加入
    #img, mask = PepperandSalt(img, mask)
    #img, mask = GaussianNoise(img, mask)

    img, mask = randomShiftScaleRotate(img, mask,
                                       shift_limit=(-0.1, 0.1),
                                       scale_limit=(-0.1, 0.1),
                                       aspect_limit=(-0.1, 0.1),
                                       rotate_limit=(-0, 0))
    img, mask = randomHorizontalFlip(img, mask)
    img, mask = randomVerticleFlip(img, mask)
    img, mask = randomRotate90(img, mask)

    re=RandomErasing()#效果提升很高
    img, mask = re(img, mask)
    '''

    mask = np.expand_dims(mask, axis=2)
    img = np.array(img, np.float32).transpose(2,0,1)/255.0 * 3.2 - 1.6
    mask = np.array(mask, np.float32).transpose(2,0,1)/255.0
    mask[mask>=0.5] = 1
    mask[mask<=0.5] = 0
    #mask = abs(mask-1)
    return img, mask



class ImageFolder(data.Dataset):

    def __init__(self, trainlist, root):
        self.ids = trainlist
        self.loader = default_loader
        self.root = root

    def __getitem__(self, index):
        id = self.ids[index]
        img, mask,mask1,mask2,mask3,mask4,mask5 = self.loader(id, self.root)
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)

        ### (2)mloss
        mask1 = torch.Tensor(mask1)
        mask2 = torch.Tensor(mask2)
        mask3 = torch.Tensor(mask3)
        mask4 = torch.Tensor(mask4)
        mask5 = torch.Tensor(mask5)
        ###

        return img, mask,mask1,mask2,mask3,mask4,mask5

    def __len__(self):
        return len(list(self.ids))
        #return 2 #(只使用两张图片进行训练)


#没使用
class ImageFolder_valid(data.Dataset):

    def __init__(self, trainlist, root):
        self.ids = trainlist
        self.loader = default_loader_valid
        self.root = root

    def __getitem__(self, index):
        id = self.ids[index]
        img, mask = self.loader(id, self.root)
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        return img, mask

    def __len__(self):
        return len(list(self.ids))
