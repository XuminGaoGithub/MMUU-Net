import torch
import torch.nn as nn
from torch.autograd import Variable as V


import cv2
import numpy as np

class MyFrame():
    def __init__(self, net, loss, lr=2e-4, evalmode = False):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8,weight_decay=5e-4, amsgrad=False)
        #self.optimizer = torch.optim.RMSprop(params=self.net.parameters(), lr=lr)
        self.loss = loss()
        self.old_lr = lr

        if evalmode:
            for i in self.net.modules():
                if isinstance(i, nn.BatchNorm2d):
                    i.eval()
    ###(3)mloss
    def set_input(self, img_batch, mask_batch=None, mask_batch1=None,mask_batch2=None,mask_batch3=None,mask_batch4=None,mask_batch5=None,img_id=None):
        self.img = img_batch
        self.mask = mask_batch
        self.mask1 = mask_batch1
        self.mask2 = mask_batch2
        self.mask3 = mask_batch3
        self.mask4 = mask_batch4
        self.mask5 = mask_batch5
        #self.masks = [mask_batch1,mask_batch2,mask_batch3,mask_batch4,mask_batch5,mask_batch]
        self.masks = [mask_batch,mask_batch1, mask_batch2, mask_batch3, mask_batch4, mask_batch]
        #print('type(self.mask)',type(self.mask))
        self.img_id = img_id
        
    def test_one_img(self, img):
        pred = self.net.forward(img)
        
        pred[pred>0.5] = 1
        pred[pred<=0.5] = 0

        mask = pred.squeeze().cpu().data.numpy()
        return mask
    
    def test_batch(self):
        self.forward(volatile=True)
        mask =  self.net.forward(self.img).cpu().data.numpy().squeeze(1)
        mask[mask>0.5] = 1
        mask[mask<=0.5] = 0
        
        return mask, self.img_id
    
    def test_one_img_from_path(self, path):
        img = cv2.imread(path)
        img = np.array(img, np.float32)/255.0 * 3.2 - 1.6
        img = V(torch.Tensor(img).cuda())
        
        mask = self.net.forward(img).squeeze().cpu().data.numpy()#.squeeze(1)
        mask[mask>0.5] = 1
        mask[mask<=0.5] = 0
        
        return mask
        
    def forward(self, volatile=False):
        self.img = V(self.img.cuda(), volatile=volatile)
        if self.mask is not None:
            self.mask = V(self.mask.cuda(), volatile=volatile)
            ###(3)mloss
            self.masks[0] = V(self.masks[0].cuda(), volatile=volatile)
            self.masks[1] = V(self.masks[1].cuda(), volatile=volatile)
            self.masks[2] = V(self.masks[2].cuda(), volatile=volatile)
            self.masks[3] = V(self.masks[3].cuda(), volatile=volatile)
            self.masks[4] = V(self.masks[4].cuda(), volatile=volatile)
            self.masks[5] = V(self.masks[5].cuda(), volatile=volatile)


        
    def optimize(self):
        self.forward()
        self.optimizer.zero_grad()
        pred = self.net.forward(self.img)

        #print('self.mask', self.mask)
        #print('pred', pred)

        loss = self.loss(self.mask, pred)
        loss.backward()
        self.optimizer.step()
        #return loss.data[0]
        return loss.item()

    def optimize_MLOSS(self):
        self.forward()
        self.optimizer.zero_grad()
        preds = self.net.forward(self.img)
        #print('preds', len(preds))
        #print('preds',preds[0])
        ###(5)mloss
        #loss = torch.zeros(1).cuda()

        #print('self.mask[0]', self.masks[1].size())
        #print('preds[0]', preds[1].size())
        loss = self.loss(self.masks[0], preds[0])
        loss = loss + self.loss(self.masks[1], preds[1])
        loss = loss + self.loss(self.masks[2], preds[2])
        loss = loss + self.loss(self.masks[3], preds[3])
        loss = loss + self.loss(self.masks[4], preds[4])
        loss = loss + self.loss(self.masks[5], preds[5])
        loss = loss/6.0

        #print('self.mask[5]', self.masks[5])
        #print('preds[5]', preds[5])

        #loss = self.loss(self.masks[0], preds[0])
        #for o in preds:
            #loss = loss + self.loss(self.mask, o)

        loss.backward()
        self.optimizer.step()
        #return loss.data[0]
        return loss.item()


    def optimize_valid(self):
        self.forward()
        self.optimizer.zero_grad()
        pred = self.net.forward(self.img)
        loss = self.loss(self.mask, pred)
        loss.backward()
        self.optimizer.step()
        #return loss.data[0]
        return loss.item(),pred


        
    def save(self, path):

        #real_model = self.net.state_dict().module
        #torch.save(real_model, path)

        torch.save(self.net.state_dict(), path)
        
    def load(self, path):
        self.net.load_state_dict(torch.load(path))
    
    def update_lr(self, new_lr, mylog, factor=False):
        if factor:
            new_lr = self.old_lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        print (mylog, 'update learning rate: %f -> %f' % (self.old_lr, new_lr))
        print ('update learning rate: %f -> %f' % (self.old_lr, new_lr))
        self.old_lr = new_lr


class MyFrame_valid():
    def __init__(self, net, loss, lr=2e-4, evalmode=False):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)
        # self.optimizer = torch.optim.RMSprop(params=self.net.parameters(), lr=lr)
        self.loss = loss()
        self.old_lr = lr

        if evalmode:
            for i in self.net.modules():
                if isinstance(i, nn.BatchNorm2d):
                    i.eval()
        self.net=self.net.eval()

    def set_input(self, img_batch, mask_batch=None, img_id=None):
        self.img = img_batch
        self.mask = mask_batch
        self.img_id = img_id

    def test_one_img(self, img):
        pred = self.net.forward(img)

        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0

        mask = pred.squeeze().cpu().data.numpy()
        return mask

    def test_batch(self):
        self.forward(volatile=True)
        mask = self.net.forward(self.img).cpu().data.numpy().squeeze(1)
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

        return mask, self.img_id

    def test_one_img_from_path(self, path):
        img = cv2.imread(path)
        img = np.array(img, np.float32) / 255.0 * 3.2 - 1.6
        img = V(torch.Tensor(img).cuda())

        mask = self.net.forward(img).squeeze().cpu().data.numpy()  # .squeeze(1)
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

        return mask

    def forward(self, volatile=False):
        self.img = V(self.img.cuda(), volatile=volatile)
        if self.mask is not None:
            self.mask = V(self.mask.cuda(), volatile=volatile)

    def optimize(self):
        self.forward()
        self.optimizer.zero_grad()
        pred = self.net.forward(self.img)
        loss = self.loss(self.mask, pred)
        loss.backward()
        self.optimizer.step()
        # return loss.data[0]
        return loss.item()

    def optimize_valid(self):
        self.forward()
        self.optimizer.zero_grad()
        pred = self.net.forward(self.img)
        loss = self.loss(self.mask, pred)
        loss.backward()
        self.optimizer.step()
        # return loss.data[0]
        return loss.item(), pred

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))

    def update_lr(self, new_lr, mylog, factor=False):
        if factor:
            new_lr = self.old_lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        print(mylog, 'update learning rate: %f -> %f' % (self.old_lr, new_lr))
        print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
        self.old_lr = new_lr
