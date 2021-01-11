
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F
import torchvision as torchmodels

from functools import partial

nonlinearity = partial(F.relu,inplace=True)

class Dblock_more_dilate(nn.Module):
    def __init__(self,channel):
        super(Dblock_more_dilate, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out + dilate5_out
        return out

class Dblock(nn.Module):
    def __init__(self,channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        #self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        #dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out# + dilate5_out
        return out

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock,self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x
    
class DinkNet34_less_pool(nn.Module):
    def __init__(self, num_classes=1):
        super(DinkNet34_more_dilate, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        
        self.dblock = Dblock_more_dilate(256)

        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        
        #Center
        e3 = self.dblock(e3)

        # Decoder
        d3 = self.decoder3(e3) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)
    
class DinkNet34(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(DinkNet34, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        
        self.dblock = Dblock(512)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        # Center
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)


class DinkNet50(nn.Module):
    def __init__(self, num_classes=1):
        super(DinkNet50, self).__init__()

        filters = [256, 512, 1024, 2048]
        resnet = models.resnet50(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        
        self.dblock = Dblock_more_dilate(2048)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        # Center
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)
    
class DinkNet101(nn.Module):
    def __init__(self, num_classes=1):
        super(DinkNet101, self).__init__()

        filters = [256, 512, 1024, 2048]
        resnet = models.resnet101(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        
        self.dblock = Dblock_more_dilate(2048)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        # Center
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)

class LinkNet34(nn.Module):
    def __init__(self, num_classes=1):
        super(LinkNet34, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)



'''
https://github.com/gengyanlei/deeplab_v3
also fine-tune
deeplab_v3+ : pytorch resnet 18/34 Basicblock
                      resnet 50/101/152 Bottleneck
            this is not original deeplab_v3+, just be based on pytorch's resnet, so many different.
'''




#################deeplab_v3##################
class ASPP(nn.Module):
    # have bias and relu, no bn
    def __init__(self, in_channel=512, depth=256):
        super().__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Sequential(nn.Conv2d(in_channel, depth, 1, 1), nn.ReLU(inplace=True))

        self.atrous_block1 = nn.Sequential(nn.Conv2d(in_channel, depth, 1, 1),
                                           nn.ReLU(inplace=True))
        self.atrous_block6 = nn.Sequential(nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6),
                                           nn.ReLU(inplace=True))
        self.atrous_block12 = nn.Sequential(nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12),
                                            nn.ReLU(inplace=True))
        self.atrous_block18 = nn.Sequential(nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18),
                                            nn.ReLU(inplace=True))

        self.conv_1x1_output = nn.Sequential(nn.Conv2d(depth * 5, depth, 1, 1), nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear', align_corners=True)

        atrous_block1 = self.atrous_block1(x)

        atrous_block6 = self.atrous_block6(x)

        atrous_block12 = self.atrous_block12(x)

        atrous_block18 = self.atrous_block18(x)

        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net


class Deeplab_v3(nn.Module):
    # in_channel = 3 fine-tune
    def __init__(self, n_classes=21, fine_tune=True):
        super().__init__()
        encoder = torchmodels.models.resnet50(pretrained=fine_tune)
        self.start = nn.Sequential(encoder.conv1, encoder.bn1,
                                   encoder.relu)

        self.maxpool = encoder.maxpool
        self.low_feature = nn.Sequential(nn.Conv2d(64, 48, 1, 1), nn.ReLU(inplace=True))  # no bn, has bias and relu

        self.layer1 = encoder.layer1  # 256
        self.layer2 = encoder.layer2  # 512
        self.layer3 = encoder.layer3  # 1024
        self.layer4 = encoder.layer4  # 2048

        self.aspp = ASPP(in_channel=2048, depth=256)

        self.conv_cat = nn.Sequential(nn.Conv2d(256 + 48, 256, 3, 1, padding=1), nn.ReLU(inplace=True))
        self.conv_cat1 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, padding=1), nn.ReLU(inplace=True))
        self.conv_cat2 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, padding=1), nn.ReLU(inplace=True))
        self.score = nn.Conv2d(256, n_classes, 1, 1)  # no relu and first conv then upsample, reduce memory

    def forward(self, x):
        size1 = x.shape[2:]  # need upsample input size
        x = self.start(x)
        xm = self.maxpool(x)

        x = self.layer1(xm)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.aspp(x)

        low_feature = self.low_feature(xm)
        size2 = low_feature.shape[2:]
        decoder_feature = F.upsample(x, size=size2, mode='bilinear', align_corners=True)

        conv_cat = self.conv_cat(torch.cat([low_feature, decoder_feature], dim=1))
        conv_cat1 = self.conv_cat1(conv_cat)
        conv_cat2 = self.conv_cat2(conv_cat1)
        score_small = self.score(conv_cat2)
        score = F.upsample(score_small, size=size1, mode='bilinear', align_corners=True)

        return score


def deeplab_v3_50(n_classes=1, fine_tune=True):
    model = Deeplab_v3(n_classes=n_classes, fine_tune=fine_tune)
    return model
##############################################


##########unet-B-res34####################
"https://mp.weixin.qq.com/s?__biz=MzI5MDUyMDIxNA==&mid=2247493160&idx=1&sn=93c06b65e04d8c7034fd50d916c6e136&chksm=ec1c0bd1db6b82c72e18cbdf01e609127c4ed452ec1a41e13000c0f86945b34d84cd25736e76&mpshare=1&scene=1&srcid=0319Qf4aw5Fwbnzaf2c3YtRo&sharer_sharetime=1584595195859&sharer_shareid=33bf8ccb960841e5cc8b2e60cd460716&exportkey=AV5%2F%2BhUGKpn7fGIicxcF%2F9Y%3D&pass_ticket=BGNeyxTCtq9OWZUTxgcR7zN2xDYHC0EMnlBwO%2FPNmtdkeRW%2Fzuh9BZVEfKZYnagi#rd"
#(0)u-net
# 把常用的2个卷积操作简单封装下
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),  # 添加了BN层
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

'''
class Unet(nn.Module):
    def __init__(self, out_ch=1):
        super(Unet, self).__init__()

        self.conv1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)
        # 逆卷积，也可以使用上采样
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        print("c1", c1.size())

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        print("c2", c2.size())

        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        print("c3", c3.size())

        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        print("c4", c4.size())

        c5 = self.conv5(p4)
        print("c5", c5.size())

        up_6 = self.up6(c5)
        print("up_6", up_6.size())
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        print("merge6", merge6.size())
        up_7 = self.up7(c6)
        print("up_7", up_7.size())
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        print("up_8", up_8.size())
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        print("up_9", up_9.size())
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        print("c9", c9.size())
        c10 = self.conv10(c9)
        print("c10", c10.size())
        out = nn.Sigmoid()(c10)
        print("out", out.size())
        return out
'''

#(1)通过 u-net原型网络来修改成 u-net_res34(1-5步骤着重看一下)
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), #添加了BN层
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
class Unet_res34(nn.Module):
    def __init__(self, out_ch=1):
        super(Unet_res34, self).__init__()

        self.conv= nn.Conv2d(512, 1024, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.encoder = torchmodels.models.resnet34(pretrained=True)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,  #（1）入口必须采用self.encoder.conv1(入口channel为3)
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = self.encoder.layer1
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = self.encoder.layer2
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = self.encoder.layer3
        self.pool4 = nn.MaxPool2d(2)

        self.conv5 = self.encoder.layer4
        #print("self.conv5",self.conv5)

        # （2）中间连接带，之前这五步没有任何channels信息,\
        # 到这一步（进入解码之前）需要按照U-net规则为下一步做准备，即需要从self.conv5-out_channel（打印获取512） 到 self.up6-in_channel(1024)\
        # 整为512(self.conv5 = DoubleConv(512, 1024))
        #self.center = DoubleConv(512, 1024)
        self.center = nn.Conv2d(512, 1024, 3, padding=1)
        #self.center = DecoderBlockV1(512, 512, 1024, True)


        # 逆卷积，也可以使用上采样
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(512, 256)
        #self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up7 = nn.ConvTranspose2d(768, 256, 2, stride=2) #对应执行up_6 = self.b1(up_6)这一步时开启
        self.conv7 = DoubleConv(256, 128)
        #self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up8 = nn.ConvTranspose2d(384, 128, 2, stride=2)
        self.conv8 = DoubleConv(128, 64)
        #self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up9 = nn.ConvTranspose2d(192, 64, 2, stride=2)
        self.conv9 = DoubleConv(64, 32)
        self.conv11 = nn.Conv2d(128, out_ch, 1)

        #辅助调整大小和通道
        self.b1 = nn.Conv2d(512, 256, 3, padding=1)
        self.b2 = nn.Conv2d(512, 1024, 3, padding=1)
        self.b3 = nn.Conv2d(384, 256, 3, padding=1)
        self.b4 = nn.Conv2d(192, 128, 3, padding=1)
        self.b5 = nn.Conv2d(128, 1024, 3, padding=1)
        self.b6 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2,
                                     padding=1)
        self.b7 = nn.ConvTranspose2d(64, out_ch, kernel_size=4, stride=2,
                                     padding=1)

    def forward(self, x):
        c1 = self.conv1(x)
        #print("c1:", c1.size())
        p1 = self.pool1(c1)
        c2 = self.conv2(c1)
        p2 = self.pool2(c2)
        #print("c2:", c2.size())
        c3 = self.conv3(c2)
        p3 = self.pool3(c3)
        #print("c3:", c3.size())
        c4 = self.conv4(c3)
        p4 = self.pool4(c4)
        #print("c4:", c4.size())
        c5 = self.conv5(c4)
        #print("c5:", c5.size())

        center=self.center(c5)
        #print("center:", center.size())

        up_6 = self.up6(center)
        #up_6 = self.b1(up_6)  #（3）torch.cat([up_6, c4],up_6, c4的后两个通道大小一致即可，这里b1的作用只是为下一步up_7 = self.up7(c6)做准备(使得输入为512（256+256）)\
        # （其实可以直接修改self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)为self.up7 = nn.ConvTranspose2d(up_6-out_channel(不需要执行b1), 256, 2, stride=2);
        #up_9 = self.up9(c8)这一步也是这样做的）

        #print("up_6:",up_6.size())
        merge6 = torch.cat([up_6, c4], dim=1)
        #print("merge6",merge6.size())
        c6=merge6
        #c6 = self.conv6(merge6)


        up_7 = self.up7(c6)
        #print("up_7:", up_7.size())
        merge7 = torch.cat([up_7, c3], dim=1)
        #c7 = self.conv7(merge7)
        c7 = merge7
        #c7=self.b3(c7)
        up_8 = self.up8(c7)
        #print("up_8:", up_8.size())
        merge8 = torch.cat([up_8, c2], dim=1)
        #c8 = self.conv8(merge8)
        c8 = merge8
        #print("c8:", c8.size())
        up_9 = self.up9(c8)
        up_9=self.pool1(up_9)
        #print("up_9:", up_9.size())
        merge9 = torch.cat([up_9, c1], dim=1)
        #c9 = self.conv9(merge9)
        c9 = merge9
        #print("c9:", c9.size())
        c10 = self.b6(c9)   #(4)通过逆卷积，还原为原始输入大小
        #print("c10:", c10.size())
        c11 = self.b7(c10)  #(4)通过逆卷积，还原为原始输入大小
        #print("c11:", c11.size())
        out = nn.Sigmoid()(c11)
        return out
    #(5)随时通过打印来对应添加-相关调整size和channel的卷积或逆卷积

#(2) Unet_res50+self.convf9

class Unet_res50(nn.Module):
    def __init__(self, out_ch=1):
        super(Unet_res50, self).__init__()

        self.conv= nn.Conv2d(512, 1024, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.encoder = torchmodels.models.resnet50(pretrained=True)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,  #（1）入口必须采用self.encoder.conv1(入口channel为3)
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)
        self.convf9 = nn.Sequential(self.encoder.conv1,  # （1）入口必须采用self.encoder.conv1(入口channel为3)
                                    self.encoder.bn1,
                                    self.encoder.relu
                                    )
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = self.encoder.layer1
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = self.encoder.layer2
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = self.encoder.layer3
        self.pool4 = nn.MaxPool2d(2)

        self.conv5 = self.encoder.layer4
        #print("self.conv5",self.conv5)

        # （2）中间连接带，之前这五步没有任何channels信息,\
        # 到这一步（进入解码之前）需要按照U-net规则为下一步做准备，即需要从self.conv5-out_channel（打印获取512） 到 self.up6-in_channel(1024)\
        # 整为512(self.conv5 = DoubleConv(512, 1024))
        #self.center = DoubleConv(512, 1024)
        self.center = Dblock(2048)
        # self.center2 = nn.Conv2d(2048, 1024, 3, padding=1)
        # self.center = nn.Conv2d(2048, 1024, 3, padding=1)
        # self.center = DecoderBlockV1(512, 512, 1024, True)

        # 逆卷积，也可以使用上采样
        self.up6 = nn.ConvTranspose2d(2048, 1024, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        # self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up7 = nn.ConvTranspose2d(2048, 256, 2, stride=2)  # 对应执行up_6 = self.b1(up_6)这一步时开启
        self.conv7 = DoubleConv(512, 256)
        # self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up8 = nn.ConvTranspose2d(768, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        # self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up9 = nn.ConvTranspose2d(384, 128, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)

        # 辅助调整大小和通道
        self.b1 = nn.Conv2d(512, 256, 3, padding=1)
        self.b2 = nn.Conv2d(512, 1024, 3, padding=1)
        self.b3 = nn.Conv2d(384, 256, 3, padding=1)
        self.b4 = nn.Conv2d(192, 128, 3, padding=1)

        # self.b6 = nn.ConvTranspose2d(192, 128, kernel_size=4, stride=2,
        # padding=1)
        # self.b7 = nn.ConvTranspose2d(128, out_ch, kernel_size=4, stride=2,
        # padding=1)

        self.b6 = nn.ConvTranspose2d(192, 192, kernel_size=4, stride=2,
                                     padding=1)
        self.b7 = nn.ConvTranspose2d(192, out_ch, kernel_size=4, stride=2,
                                     padding=1)
        self.b9 = nn.ConvTranspose2d(192, out_ch, kernel_size=4, stride=2, padding=1)
        self.b10 = nn.ConvTranspose2d(192, out_ch, kernel_size=4, stride=2, padding=1)
        self.b11 = nn.ConvTranspose2d(195, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # print('x', x.size())
        c1 = self.conv1(x)
        # print("c1:", c1.size())
        cf9 = self.convf9(x)
        # print("c1:", cf9.size())
        p1 = self.pool1(c1)
        c2 = self.conv2(c1)
        p2 = self.pool2(c2)
        # print("c2:", c2.size())
        c3 = self.conv3(c2)
        p3 = self.pool3(c3)
        # print("c3:", c3.size())
        c4 = self.conv4(c3)
        p4 = self.pool4(c4)
        # print("c4:", c4.size())
        c5 = self.conv5(c4)
        # print("c5:", c5.size())

        center = self.center(c5)

        # print("center:", center.size())

        up_6 = self.up6(center)
        # up_6 = self.b1(up_6)  #（3）torch.cat([up_6, c4],up_6, c4的后两个通道大小一致即可，这里b1的作用只是为下一步up_7 = self.up7(c6)做准备(使得输入为512（256+256）)\
        # （其实可以直接修改self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)为self.up7 = nn.ConvTranspose2d(up_6-out_channel(不需要执行b1), 256, 2, stride=2);
        # up_9 = self.up9(c8)这一步也是这样做的）

        # print("up_6:",up_6.size())
        merge6 = torch.cat([up_6, c4], dim=1)
        # print("merge6",merge6.size())
        c6 = merge6
        # c6 = self.conv6(merge6)

        up_7 = self.up7(c6)
        # print("up_7:", up_7.size())
        merge7 = torch.cat([up_7, c3], dim=1)
        # c7 = self.conv7(merge7)
        c7 = merge7
        # c7=self.b3(c7)
        up_8 = self.up8(c7)
        # print("up_8:", up_8.size())
        merge8 = torch.cat([up_8, c2], dim=1)
        # c8 = self.conv8(merge8)
        c8 = merge8
        # print("c8:", c8.size())
        up_9 = self.up9(c8)
        # up_9=self.pool1(up_9)
        # print("up_9:", up_9.size())
        merge9 = torch.cat([up_9, cf9], dim=1)
        # c9 = self.conv9(merge9)
        c9 = merge9
        # print("c9:", c9.size())
        c10 = self.b9(c9)  # (4)通过逆卷积，还原为原始输入大小
        out = nn.Sigmoid()(c10)
        return out
    #(5)随时通过打印来对应添加-相关调整size和channel的卷积或逆卷积

#(3)
def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)
class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x
class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.scale_factor = scale_factor
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(x, size=self.size, scale_factor=self.scale_factor,
                        mode=self.mode, align_corners=self.align_corners)
        return x
class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockV2, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """

            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                Interpolate(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)
class AlbuNet(nn.Module):
    """
        UNet (https://arxiv.org/abs/1505.04597) with Resnet34(https://arxiv.org/abs/1512.03385) encoder
        Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/
        """

    def __init__(self, num_classes=1, num_filters=32, is_deconv=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with resnet34
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchmodels.models.resnet34(pretrained=True)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4

        self.center = DecoderBlockV2(512, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.dec5 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(128 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(64 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        print("conv1", conv1.size())
        conv2 = self.conv2(conv1)
        print("conv2", conv2.size())
        conv3 = self.conv3(conv2)
        print("conv3", conv3.size())
        conv4 = self.conv4(conv3)
        print("conv4", conv4.size())
        conv5 = self.conv5(conv4)
        print("conv5", conv5.size())

        center = self.center(self.pool(conv5))
        print("center",center.size())

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        print("dec5", dec5.size())
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        print("dec4", dec4.size())
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        print("dec3", dec3.size())
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        print("dec2", dec2.size())
        dec1 = self.dec1(dec2)
        print("dec1", dec1.size())
        dec0 = self.dec0(dec1)
        print("dec0", dec0.size())

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec0), dim=1)
            print("x_out", x_out.size())
        else:
            x_out = self.final(dec0)
            print("x_out", x_out.size())

        return x_out


class AlbuNet50(nn.Module):
    """
        UNet (https://arxiv.org/abs/1505.04597) with Resnet34(https://arxiv.org/abs/1512.03385) encoder
        Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/
        """

    def __init__(self, num_classes=1, num_filters=128, is_deconv=True):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with resnet34
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchmodels.models.resnet50(pretrained=True)
        #print(self.encoder)
        #self.encoder = resnest34(pretrained=False)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4

        self.center = DecoderBlockV2(2048, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.dec5 = DecoderBlockV2(2048 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(1024 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(512 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(256 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        print("conv1", conv1.size())
        conv2 = self.conv2(conv1)
        print("conv2", conv2.size())
        conv3 = self.conv3(conv2)
        print("conv3", conv3.size())
        conv4 = self.conv4(conv3)
        print("conv4", conv4.size())
        conv5 = self.conv5(conv4)
        print("conv5", conv5.size())

        center = self.center(self.pool(conv5))
        print("center",center.size())

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        print("dec5", dec5.size())
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        print("dec4", dec4.size())
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        print("dec3", dec3.size())
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        print("dec2", dec2.size())
        dec1 = self.dec1(dec2)
        print("dec1", dec1.size())
        dec0 = self.dec0(dec1)
        print("dec0", dec0.size())

        x_out = self.final(dec0)
        x_out = nn.Sigmoid()(x_out)
        '''
        if self.num_classes > 0:
            x_out = F.log_softmax(self.final(dec0), dim=1)
            print("x_out", x_out.size())
        else:
            x_out = self.final(dec0)
            print("x_out", x_out.size())
        '''


        return x_out


class D_block_more_dilate(nn.Module):
    def __init__(self, channel):
        super(D_block_more_dilate, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out + dilate5_out
        return out

#(1)D_AlbuNet50（卷积之再接dilate_conv）
"""
class D_AlbuNet50(nn.Module):
    def __init__(self, num_classes=1, num_filters=128, is_deconv=False):
        '''
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with resnet34
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        '''
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchmodels.models.resnet50(pretrained=True)
        #print(self.encoder)
        #self.encoder = resnest34(pretrained=False)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4

        self.center = DecoderBlockV2(2048, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.center2 = D_block_more_dilate(1024)

        self.dec5 = DecoderBlockV2(2048 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(1024 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(512 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(256 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        print("conv1", conv1.size())
        conv2 = self.conv2(conv1)
        print("conv2", conv2.size())
        conv3 = self.conv3(conv2)
        print("conv3", conv3.size())
        conv4 = self.conv4(conv3)
        print("conv4", conv4.size())
        conv5 = self.conv5(conv4)
        print("conv5", conv5.size())

        center = self.center(self.pool(conv5))
        center = self.center2(center)
        print("center",center.size())

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        print("dec5", dec5.size())
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        print("dec4", dec4.size())
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        print("dec3", dec3.size())
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        print("dec2", dec2.size())
        dec1 = self.dec1(dec2)
        print("dec1", dec1.size())
        dec0 = self.dec0(dec1)
        print("dec0", dec0.size())

        x_out = self.final(dec0)
        x_out = nn.Sigmoid()(x_out)
        '''
        if self.num_classes > 0:
            x_out = F.log_softmax(self.final(dec0), dim=1)
            print("x_out", x_out.size())
        else:
            x_out = self.final(dec0)
            print("x_out", x_out.size())
        '''


        return x_out
"""

#(2)D_AlbuNet50（卷积替换为dilate_conv）
class DecoderBlockV3(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockV3, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """

            self.block = nn.Sequential(
                #ConvRelu(in_channels, middle_channels),#使用D_block_more_dilate代替该步骤
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                Interpolate(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)

class D_AlbuNet50(nn.Module):
    def __init__(self, num_classes=1, num_filters=128, is_deconv=True): #出现cuda out of memory，可通过设小num_filters=32来解决
        '''
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with resnet34
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        '''
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchmodels.models.resnet50(pretrained=True)
        #print(self.encoder)
        #self.encoder = resnest34(pretrained=False)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4


        self.center1 = D_block_more_dilate(2048) #D_block_more_dilate代替了DecoderBlockV3中的卷积操作
        self.center2 = DecoderBlockV3(2048, 2048, num_filters * 8, is_deconv)#D_block_more_dilate代替了DecoderBlockV3中的卷积操作

        self.dec5 = DecoderBlockV2(2048 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(1024 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(512 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(256 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        print("conv1", conv1.size())
        conv2 = self.conv2(conv1)
        print("conv2", conv2.size())
        conv3 = self.conv3(conv2)
        print("conv3", conv3.size())
        conv4 = self.conv4(conv3)
        print("conv4", conv4.size())
        conv5 = self.conv5(conv4)
        print("conv5", conv5.size())


        center = self.center1(self.pool(conv5))
        center = self.center2(center)
        print("center",center.size())

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        print("dec5", dec5.size())
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        print("dec4", dec4.size())
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        print("dec3", dec3.size())
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        print("dec2", dec2.size())
        dec1 = self.dec1(dec2)
        print("dec1", dec1.size())
        dec0 = self.dec0(dec1)
        print("dec0", dec0.size())

        x_out = self.final(dec0)
        x_out = nn.Sigmoid()(x_out)

        return x_out

# D_ResstNet50
from resnest.torch import resnest34,resnest50
class D_ResstNet50(nn.Module):
    def __init__(self, num_classes=1, num_filters=128, is_deconv=True): #出现出现cuda out of memory，可通过设小num_filters=32来解决
        '''
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with resnet34
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        '''
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)
        self.encoder = resnest50(pretrained=False)#self.encoder =resnest34(pretrained=True)
        #print(self.encoder)
        #self.encoder = resnest34(pretrained=False)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4


        self.center1 = D_block_more_dilate(2048) #D_block_more_dilate代替了DecoderBlockV3中的卷积操作
        self.center2 = DecoderBlockV3(2048, 2048, num_filters * 8, is_deconv)#D_block_more_dilate代替了DecoderBlockV3中的卷积操作

        self.dec5 = DecoderBlockV2(2048 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(1024 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(512 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(256 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        print("conv1", conv1.size())
        conv2 = self.conv2(conv1)
        print("conv2", conv2.size())
        conv3 = self.conv3(conv2)
        print("conv3", conv3.size())
        conv4 = self.conv4(conv3)
        print("conv4", conv4.size())
        conv5 = self.conv5(conv4)
        print("conv5", conv5.size())


        center = self.center1(self.pool(conv5))
        center = self.center2(center)
        print("center",center.size())

        #print(torch.cat([center, conv5], 1).size())
        dec5 = self.dec5(torch.cat([center, conv5], 1))
        print("dec5", dec5.size())
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        print("dec4", dec4.size())
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        print("dec3", dec3.size())
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        print("dec2", dec2.size())
        dec1 = self.dec1(dec2)
        print("dec1", dec1.size())
        dec0 = self.dec0(dec1)
        print("dec0", dec0.size())

        x_out = self.final(dec0)
        x_out = nn.Sigmoid()(x_out)

        return x_out


#D_ResstNet34
class D_AlbuNet34(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, is_deconv=True):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with resnet34
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchmodels.models.resnet34(pretrained=True)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4

        #self.center = DecoderBlockV2(512, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.center1 = D_block_more_dilate(512)  # D_block_more_dilate代替了DecoderBlockV3中的卷积操作
        self.center2 = DecoderBlockV3(512, 512, num_filters * 8,
                                      is_deconv)  # D_block_more_dilate代替了DecoderBlockV3中的卷积操作

        self.dec5 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(128 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(64 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        print("conv1", conv1.size())
        conv2 = self.conv2(conv1)
        print("conv2", conv2.size())
        conv3 = self.conv3(conv2)
        print("conv3", conv3.size())
        conv4 = self.conv4(conv3)
        print("conv4", conv4.size())
        conv5 = self.conv5(conv4)
        print("conv5", conv5.size())


        #center = self.center(self.pool(conv5))
        center = self.center1(self.pool(conv5))
        center = self.center2(center)
        print("center", center.size())


        dec5 = self.dec5(torch.cat([center, conv5], 1))
        print("dec5", dec5.size())
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        print("dec4", dec4.size())
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        print("dec3", dec3.size())
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        print("dec2", dec2.size())
        dec1 = self.dec1(dec2)
        print("dec1", dec1.size())
        dec0 = self.dec0(dec1)
        print("dec0", dec0.size())

        x_out = self.final(dec0)
        x_out = nn.Sigmoid()(x_out)

        return x_out

# D_ResstNet34(pre-ResstNet50)
from resnest.torch import resnest34,resnest50
class D_ResstNet34(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, is_deconv=True):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with resnet34
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = resnest34(pretrained=True)#self.encoder =resnest34(pretrained=True)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4

        # self.center = DecoderBlockV2(512, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.center1 = D_block_more_dilate(2048)  # D_block_more_dilate代替了DecoderBlockV3中的卷积操作
        self.center2 = DecoderBlockV3(2048, 2048, num_filters * 8,
                                      is_deconv)  # D_block_more_dilate代替了DecoderBlockV3中的卷积操作

        self.dec5 = DecoderBlockV2(2048 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(1024+ num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(512+ num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(256+ num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        print("conv1", conv1.size())
        conv2 = self.conv2(conv1)
        print("conv2", conv2.size())
        conv3 = self.conv3(conv2)
        print("conv3", conv3.size())
        conv4 = self.conv4(conv3)
        print("conv4", conv4.size())
        conv5 = self.conv5(conv4)
        print("conv5", conv5.size())

        # center = self.center(self.pool(conv5))
        center = self.center1(self.pool(conv5))
        center = self.center2(center)
        print("center", center.size())

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        print("dec5", dec5.size())
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        print("dec4", dec4.size())
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        print("dec3", dec3.size())
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        print("dec2", dec2.size())
        dec1 = self.dec1(dec2)
        print("dec1", dec1.size())
        dec0 = self.dec0(dec1)
        print("dec0", dec0.size())

        x_out = self.final(dec0)
        x_out = nn.Sigmoid()(x_out)

        return x_out




##############################################


#################unet-resnest############################


###（1）-（7）与D_ResstNet50区别，合并之后直接up,而D_ResstNet50合并之前还要经过两个卷积
from resnest.torch import resnest34,resnest50 #/home/comin/anaconda3/lib/python3.7/site-packages/resnest/__init__.py中加入resnest34并进行初始化
#(1)
'''
class Unet_resst34(nn.Module):
    def __init__(self, out_ch=1):
        super(Unet_resst34, self).__init__()

        self.conv= nn.Conv2d(512, 1024, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.encoder =resnest34(pretrained=False)
        #self.encoder = resnest50(pretrained=True) #cuda out of memory(需要更好的显卡)
        #print('resnest50',self.encoder)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,  #（1）入口必须采用self.encoder.conv1(入口channel为3)
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = self.encoder.layer1
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = self.encoder.layer2
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = self.encoder.layer3
        self.pool4 = nn.MaxPool2d(2)

        self.conv5 = self.encoder.layer4
        #print("self.conv5",self.conv5)

        # （2）中间连接带，之前这五步没有任何channels信息,\
        # 到这一步（进入解码之前）需要按照U-net规则为下一步做准备，即需要从self.conv5-out_channel（打印获取512） 到 self.up6-in_channel(1024)\
        # 整为512(self.conv5 = DoubleConv(512, 1024))
        #self.center = DoubleConv(512, 1024)
        self.center1 = Dblock(2048)
        self.center2 = nn.Conv2d(2048, 1024, 3, padding=1)
        #self.center = nn.Conv2d(2048, 1024, 3, padding=1)
        #self.center = DecoderBlockV1(512, 512, 1024, True)


        # 逆卷积，也可以使用上采样
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(512, 256)
        #self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up7 = nn.ConvTranspose2d(1536, 256, 2, stride=2) #对应执行up_6 = self.b1(up_6)这一步时开启
        self.conv7 = DoubleConv(256, 128)
        #self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up8 = nn.ConvTranspose2d(768, 128, 2, stride=2)
        self.conv8 = DoubleConv(128, 64)
        #self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up9 = nn.ConvTranspose2d(384, 64, 2, stride=2)
        self.conv9 = DoubleConv(64, 32)
        self.conv11 = nn.Conv2d(128, out_ch, 1)

        #辅助调整大小和通道
        self.b1 = nn.Conv2d(512, 256, 3, padding=1)
        self.b2 = nn.Conv2d(512, 1024, 3, padding=1)
        self.b3 = nn.Conv2d(384, 256, 3, padding=1)
        self.b4 = nn.Conv2d(192, 128, 3, padding=1)
        self.b5 = nn.Conv2d(128, 1024, 3, padding=1)
        self.b6 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2,
                                     padding=1)
        self.b7 = nn.ConvTranspose2d(64, out_ch, kernel_size=4, stride=2,
                                     padding=1)

    def forward(self, x):
        c1 = self.conv1(x)
        print("c1:", c1.size())
        p1 = self.pool1(c1)
        c2 = self.conv2(c1)
        p2 = self.pool2(c2)
        print("c2:", c2.size())
        c3 = self.conv3(c2)
        p3 = self.pool3(c3)
        print("c3:", c3.size())
        c4 = self.conv4(c3)
        p4 = self.pool4(c4)
        print("c4:", c4.size())
        c5 = self.conv5(c4)
        print("c5:", c5.size())

        center=self.center1(c5)
        center = self.center2(center)
        print("center:", center.size())

        up_6 = self.up6(center)
        #up_6 = self.b1(up_6)  #（3）torch.cat([up_6, c4],up_6, c4的后两个通道大小一致即可，这里b1的作用只是为下一步up_7 = self.up7(c6)做准备(使得输入为512（256+256）)\
        # （其实可以直接修改self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)为self.up7 = nn.ConvTranspose2d(up_6-out_channel(不需要执行b1), 256, 2, stride=2);
        #up_9 = self.up9(c8)这一步也是这样做的）

        #print("up_6:",up_6.size())
        merge6 = torch.cat([up_6, c4], dim=1)
        #print("merge6",merge6.size())
        c6=merge6
        #c6 = self.conv6(merge6)


        up_7 = self.up7(c6)
        #print("up_7:", up_7.size())
        merge7 = torch.cat([up_7, c3], dim=1)
        #c7 = self.conv7(merge7)
        c7 = merge7
        #c7=self.b3(c7)
        up_8 = self.up8(c7)
        #print("up_8:", up_8.size())
        merge8 = torch.cat([up_8, c2], dim=1)
        #c8 = self.conv8(merge8)
        c8 = merge8
        #print("c8:", c8.size())
        up_9 = self.up9(c8)
        up_9=self.pool1(up_9)
        #print("up_9:", up_9.size())
        merge9 = torch.cat([up_9, c1], dim=1)
        #c9 = self.conv9(merge9)
        c9 = merge9
        #print("c9:", c9.size())
        c10 = self.b6(c9)   #(4)通过逆卷积，还原为原始输入大小
        #print("c10:", c10.size())
        c11 = self.b7(c10)  #(4)通过逆卷积，还原为原始输入大小
        #print("c11:", c11.size())
        out = nn.Sigmoid()(c11)
        return out
    #(5)随时通过打印来对应添加-相关调整size和channel的卷积或逆卷积
'''

#(2)
'''

from resnest.torch import resnest50
class Unet_resst34(nn.Module):
    def __init__(self, out_ch=1):
        super(Unet_resst34, self).__init__()

        self.conv= nn.Conv2d(512, 1024, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.encoder =resnest50(pretrained=False)
        #self.encoder = resnest50(pretrained=True)
        #print('resnest50',self.encoder)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,  #（1）入口必须采用self.encoder.conv1(入口channel为3)
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = self.encoder.layer1
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = self.encoder.layer2
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = self.encoder.layer3
        self.pool4 = nn.MaxPool2d(2)

        self.conv5 = self.encoder.layer4
        #print("self.conv5",self.conv5)

        # （2）中间连接带，之前这五步没有任何channels信息,\
        # 到这一步（进入解码之前）需要按照U-net规则为下一步做准备，即需要从self.conv5-out_channel（打印获取512） 到 self.up6-in_channel(1024)\
        # 整为512(self.conv5 = DoubleConv(512, 1024))
        #self.center = DoubleConv(512, 1024)
        self.center = Dblock(2048)
        #self.center2 = nn.Conv2d(2048, 1024, 3, padding=1)
        #self.center = nn.Conv2d(2048, 1024, 3, padding=1)
        #self.center = DecoderBlockV1(512, 512, 1024, True)


        # 逆卷积，也可以使用上采样
        self.up6 = nn.ConvTranspose2d(2048, 1024, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        #self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up7 = nn.ConvTranspose2d(2048, 256, 2, stride=2) #对应执行up_6 = self.b1(up_6)这一步时开启
        self.conv7 = DoubleConv(512, 256)
        #self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up8 = nn.ConvTranspose2d(768, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        #self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up9 = nn.ConvTranspose2d(384, 128, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)


        #辅助调整大小和通道
        self.b1 = nn.Conv2d(512, 256, 3, padding=1)
        self.b2 = nn.Conv2d(512, 1024, 3, padding=1)
        self.b3 = nn.Conv2d(384, 256, 3, padding=1)
        self.b4 = nn.Conv2d(192, 128, 3, padding=1)

        #self.b6 = nn.ConvTranspose2d(192, 128, kernel_size=4, stride=2,
                                     #padding=1)
        #self.b7 = nn.ConvTranspose2d(128, out_ch, kernel_size=4, stride=2,
                                     #padding=1)

        self.b6 = nn.ConvTranspose2d(192, 192, kernel_size=4, stride=2,
                                     padding=1)
        self.b7 = nn.ConvTranspose2d(192, out_ch, kernel_size=4, stride=2,
                                     padding=1)

    def forward(self, x):
        c1 = self.conv1(x)
        #print("c1:", c1.size())
        p1 = self.pool1(c1)
        c2 = self.conv2(c1)
        p2 = self.pool2(c2)
        #print("c2:", c2.size())
        c3 = self.conv3(c2)
        p3 = self.pool3(c3)
        #print("c3:", c3.size())
        c4 = self.conv4(c3)
        p4 = self.pool4(c4)
        #print("c4:", c4.size())
        c5 = self.conv5(c4)
        #print("c5:", c5.size())

        center=self.center(c5)

        #print("center:", center.size())

        up_6 = self.up6(center)
        #up_6 = self.b1(up_6)  #（3）torch.cat([up_6, c4],up_6, c4的后两个通道大小一致即可，这里b1的作用只是为下一步up_7 = self.up7(c6)做准备(使得输入为512（256+256）)\
        # （其实可以直接修改self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)为self.up7 = nn.ConvTranspose2d(up_6-out_channel(不需要执行b1), 256, 2, stride=2);
        #up_9 = self.up9(c8)这一步也是这样做的）

        #print("up_6:",up_6.size())
        merge6 = torch.cat([up_6, c4], dim=1)
        #print("merge6",merge6.size())
        c6=merge6
        #c6 = self.conv6(merge6)


        up_7 = self.up7(c6)
        #print("up_7:", up_7.size())
        merge7 = torch.cat([up_7, c3], dim=1)
        #c7 = self.conv7(merge7)
        c7 = merge7
        #c7=self.b3(c7)
        up_8 = self.up8(c7)
        #print("up_8:", up_8.size())
        merge8 = torch.cat([up_8, c2], dim=1)
        #c8 = self.conv8(merge8)
        c8 = merge8
        #print("c8:", c8.size())
        up_9 = self.up9(c8)
        up_9=self.pool1(up_9)
        print("up_9:", up_9.size())
        merge9 = torch.cat([up_9, c1], dim=1)
        #c9 = self.conv9(merge9)
        c9 = merge9
        print("c9:", c9.size())
        c10 = self.b6(c9)   #(4)通过逆卷积，还原为原始输入大小
        print("c10:", c10.size())
        c11 = self.b7(c10)  #(4)通过逆卷积，还原为原始输入大小
        print("c11:", c11.size())
        out = nn.Sigmoid()(c11)
        return out
    #(5)随时通过打印来对应添加-相关调整size和channel的卷积或逆卷积
'''


# (3)
'''
from resnest.torch import resnest50
class Unet_resst34(nn.Module):
    def __init__(self, out_ch=1):
        super(Unet_resst34, self).__init__()

        self.conv = nn.Conv2d(512, 1024, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # self.encoder =resnest50(pretrained=False)
        self.encoder = resnest50(pretrained=True)
        # print('resnest50',self.encoder)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,  # （1）入口必须采用self.encoder.conv1(入口channel为3)
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)
        self.convf9 = nn.Sequential(self.encoder.conv1,  # （1）入口必须采用self.encoder.conv1(入口channel为3)
                                    self.encoder.bn1,
                                    self.encoder.relu
                                    )
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = self.encoder.layer1
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = self.encoder.layer2
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = self.encoder.layer3
        self.pool4 = nn.MaxPool2d(2)

        self.conv5 = self.encoder.layer4
        # print("self.conv5",self.conv5)

        # （2）中间连接带，之前这五步没有任何channels信息,\
        # 到这一步（进入解码之前）需要按照U-net规则为下一步做准备，即需要从self.conv5-out_channel（打印获取512） 到 self.up6-in_channel(1024)\
        # 整为512(self.conv5 = DoubleConv(512, 1024))
        # self.center = DoubleConv(512, 1024)
        self.center = Dblock(2048)
        # self.center2 = nn.Conv2d(2048, 1024, 3, padding=1)
        # self.center = nn.Conv2d(2048, 1024, 3, padding=1)
        # self.center = DecoderBlockV1(512, 512, 1024, True)

        # 逆卷积，也可以使用上采样
        self.up6 = nn.ConvTranspose2d(2048, 1024, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        # self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up7 = nn.ConvTranspose2d(2048, 256, 2, stride=2)  # 对应执行up_6 = self.b1(up_6)这一步时开启
        self.conv7 = DoubleConv(512, 256)
        # self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up8 = nn.ConvTranspose2d(768, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        # self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up9 = nn.ConvTranspose2d(384, 128, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)

        # 辅助调整大小和通道
        self.b1 = nn.Conv2d(512, 256, 3, padding=1)
        self.b2 = nn.Conv2d(512, 1024, 3, padding=1)
        self.b3 = nn.Conv2d(384, 256, 3, padding=1)
        self.b4 = nn.Conv2d(192, 128, 3, padding=1)

        # self.b6 = nn.ConvTranspose2d(192, 128, kernel_size=4, stride=2,
        # padding=1)
        # self.b7 = nn.ConvTranspose2d(128, out_ch, kernel_size=4, stride=2,
        # padding=1)

        self.b6 = nn.ConvTranspose2d(192, 192, kernel_size=4, stride=2,
                                     padding=1)
        self.b7 = nn.ConvTranspose2d(192, out_ch, kernel_size=4, stride=2,
                                     padding=1)
        self.b9 = nn.ConvTranspose2d(192, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        #print('x', x.size())
        c1 = self.conv1(x)
        #print("c1:", c1.size())
        cf9 = self.convf9(x)
        #print("c1:", cf9.size())
        p1 = self.pool1(c1)
        c2 = self.conv2(c1)
        p2 = self.pool2(c2)
        # print("c2:", c2.size())
        c3 = self.conv3(c2)
        p3 = self.pool3(c3)
        # print("c3:", c3.size())
        c4 = self.conv4(c3)
        p4 = self.pool4(c4)
        # print("c4:", c4.size())
        c5 = self.conv5(c4)
        # print("c5:", c5.size())

        center = self.center(c5)

        # print("center:", center.size())

        up_6 = self.up6(center)
        # up_6 = self.b1(up_6)  #（3）torch.cat([up_6, c4],up_6, c4的后两个通道大小一致即可，这里b1的作用只是为下一步up_7 = self.up7(c6)做准备(使得输入为512（256+256）)\
        # （其实可以直接修改self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)为self.up7 = nn.ConvTranspose2d(up_6-out_channel(不需要执行b1), 256, 2, stride=2);
        # up_9 = self.up9(c8)这一步也是这样做的）

        # print("up_6:",up_6.size())
        merge6 = torch.cat([up_6, c4], dim=1)
        # print("merge6",merge6.size())
        c6 = merge6
        # c6 = self.conv6(merge6)

        up_7 = self.up7(c6)
        # print("up_7:", up_7.size())
        merge7 = torch.cat([up_7, c3], dim=1)
        # c7 = self.conv7(merge7)
        c7 = merge7
        # c7=self.b3(c7)
        up_8 = self.up8(c7)
        # print("up_8:", up_8.size())
        merge8 = torch.cat([up_8, c2], dim=1)
        # c8 = self.conv8(merge8)
        c8 = merge8
        # print("c8:", c8.size())
        up_9 = self.up9(c8)
        # up_9=self.pool1(up_9)
        # print("up_9:", up_9.size())
        merge9 = torch.cat([up_9, cf9], dim=1)
        # c9 = self.conv9(merge9)
        c9 = merge9
        #print("c9:", c9.size())
        c10 = self.b9(c9)  # (4)通过逆卷积，还原为原始输入大小
        # c10 = self.b10(c10)
        #print("c10:", c10.size())

        out = nn.Sigmoid()(c10)
        return out
'''


#(4)
'''
from resnest.torch import resnest50
class Unet_resst34(nn.Module):
    def __init__(self, out_ch=1):
        super(Unet_resst34, self).__init__()

        self.conv= nn.Conv2d(512, 1024, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        #self.encoder =resnest50(pretrained=False)
        self.encoder = resnest50(pretrained=True)
        #print('resnest50',self.encoder)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,  #（1）入口必须采用self.encoder.conv1(入口channel为3)
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)
        self.convf9 = nn.Sequential(self.encoder.conv1,  # （1）入口必须采用self.encoder.conv1(入口channel为3)
                                   self.encoder.bn1,
                                   self.encoder.relu
                                   )
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = self.encoder.layer1
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = self.encoder.layer2
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = self.encoder.layer3
        self.pool4 = nn.MaxPool2d(2)

        self.conv5 = self.encoder.layer4
        #print("self.conv5",self.conv5)

        # （2）中间连接带，之前这五步没有任何channels信息,\
        # 到这一步（进入解码之前）需要按照U-net规则为下一步做准备，即需要从self.conv5-out_channel（打印获取512） 到 self.up6-in_channel(1024)\
        # 整为512(self.conv5 = DoubleConv(512, 1024))
        #self.center = DoubleConv(512, 1024)
        self.center = Dblock(2048)
        #self.center2 = nn.Conv2d(2048, 1024, 3, padding=1)
        #self.center = nn.Conv2d(2048, 1024, 3, padding=1)
        #self.center = DecoderBlockV1(512, 512, 1024, True)


        # 逆卷积，也可以使用上采样
        self.up6 = nn.ConvTranspose2d(2048, 1024, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        #self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up7 = nn.ConvTranspose2d(2048, 256, 2, stride=2) #对应执行up_6 = self.b1(up_6)这一步时开启
        self.conv7 = DoubleConv(512, 256)
        #self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up8 = nn.ConvTranspose2d(768, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        #self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up9 = nn.ConvTranspose2d(384, 128, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)


        #辅助调整大小和通道
        self.b1 = nn.Conv2d(512, 256, 3, padding=1)
        self.b2 = nn.Conv2d(512, 1024, 3, padding=1)
        self.b3 = nn.Conv2d(384, 256, 3, padding=1)
        self.b4 = nn.Conv2d(192, 128, 3, padding=1)
        
        #self.b6 = nn.ConvTranspose2d(192, 128, kernel_size=4, stride=2,
                                     #padding=1)
        #self.b7 = nn.ConvTranspose2d(128, out_ch, kernel_size=4, stride=2,
                                     #padding=1)
        
        self.b6 = nn.ConvTranspose2d(192, 192, kernel_size=4, stride=2,
                                     padding=1)
        self.b7 = nn.ConvTranspose2d(192, out_ch, kernel_size=4, stride=2,
                                     padding=1)
        self.b9 = nn.ConvTranspose2d(192, 192, kernel_size=4, stride=2,padding=1)
        self.b10 = nn.ConvTranspose2d(192, out_ch, kernel_size=4, stride=2, padding=1)
        self.b11 = nn.ConvTranspose2d(195, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        print('x',x.size())
        c1 = self.conv1(x)
        print("c1:", c1.size())
        cf9 = self.convf9(x)
        print("c1:", cf9.size())
        p1 = self.pool1(c1)
        c2 = self.conv2(c1)
        p2 = self.pool2(c2)
        #print("c2:", c2.size())
        c3 = self.conv3(c2)
        p3 = self.pool3(c3)
        #print("c3:", c3.size())
        c4 = self.conv4(c3)
        p4 = self.pool4(c4)
        #print("c4:", c4.size())
        c5 = self.conv5(c4)
        #print("c5:", c5.size())

        center=self.center(c5)

        #print("center:", center.size())

        up_6 = self.up6(center)
        #up_6 = self.b1(up_6)  #（3）torch.cat([up_6, c4],up_6, c4的后两个通道大小一致即可，这里b1的作用只是为下一步up_7 = self.up7(c6)做准备(使得输入为512（256+256）)\
        # （其实可以直接修改self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)为self.up7 = nn.ConvTranspose2d(up_6-out_channel(不需要执行b1), 256, 2, stride=2);
        #up_9 = self.up9(c8)这一步也是这样做的）

        #print("up_6:",up_6.size())
        merge6 = torch.cat([up_6, c4], dim=1)
        #print("merge6",merge6.size())
        c6=merge6
        #c6 = self.conv6(merge6)


        up_7 = self.up7(c6)
        #print("up_7:", up_7.size())
        merge7 = torch.cat([up_7, c3], dim=1)
        #c7 = self.conv7(merge7)
        c7 = merge7
        #c7=self.b3(c7)
        up_8 = self.up8(c7)
        #print("up_8:", up_8.size())
        merge8 = torch.cat([up_8, c2], dim=1)
        #c8 = self.conv8(merge8)
        c8 = merge8
        #print("c8:", c8.size())
        up_9 = self.up9(c8)
        #up_9=self.pool1(up_9)
        #print("up_9:", up_9.size())
        merge9 = torch.cat([up_9, cf9], dim=1)
        #c9 = self.conv9(merge9)
        c9 = merge9
        print("c9:", c9.size())
        c10 = self.b9(c9)   #(4)通过逆卷积，还原为原始输入大小
        #c10 = self.b10(c10)
        print("c10:", c10.size())

        merge10 = torch.cat([c10, x], dim=1)
        c11=merge10
        c11=self.b11(c11)
        #c11 = self.b7(c10)  #(4)通过逆卷积，还原为原始输入大小
        print("c11:", c11.size())
        out = nn.Sigmoid()(c11)
        return out
'''

#(5)
'''
from resnest.torch import resnest50
class Unet_resst34(nn.Module):
    def __init__(self, out_ch=1):
        super(Unet_resst34, self).__init__()

        self.conv= nn.Conv2d(512, 1024, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.encoder =resnest50(pretrained=False)
        #self.encoder = resnest50(pretrained=True)
        #print('resnest50',self.encoder)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,  #（1）入口必须采用self.encoder.conv1(入口channel为3)
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)
        self.convf9 = nn.Sequential(self.encoder.conv1,  # （1）入口必须采用self.encoder.conv1(入口channel为3)
                                    self.encoder.bn1,
                                    self.encoder.relu
                                    )
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = self.encoder.layer1
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = self.encoder.layer2
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = self.encoder.layer3
        self.pool4 = nn.MaxPool2d(2)

        self.conv5 = self.encoder.layer4
        #print("self.conv5",self.conv5)

        # （2）中间连接带，之前这五步没有任何channels信息,\
        # 到这一步（进入解码之前）需要按照U-net规则为下一步做准备，即需要从self.conv5-out_channel（打印获取512） 到 self.up6-in_channel(1024)\
        # 整为512(self.conv5 = DoubleConv(512, 1024))
        #self.center = DoubleConv(512, 1024)
        self.center1 = Dblock(2048)
        self.center2 = nn.Conv2d(2048, 1024, 3, padding=1)
        #self.center = nn.Conv2d(2048, 1024, 3, padding=1)
        #self.center = DecoderBlockV1(512, 512, 1024, True)


        # 逆卷积，也可以使用上采样
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(512, 256)
        #self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up7 = nn.ConvTranspose2d(1536, 256, 2, stride=2) #对应执行up_6 = self.b1(up_6)这一步时开启
        self.conv7 = DoubleConv(256, 128)
        #self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up8 = nn.ConvTranspose2d(768, 128, 2, stride=2)
        self.conv8 = DoubleConv(128, 64)
        #self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up9 = nn.ConvTranspose2d(384, 64, 2, stride=2)
        self.conv9 = DoubleConv(64, 32)
        self.conv11 = nn.Conv2d(128, out_ch, 1)

        #辅助调整大小和通道
        self.b1 = nn.Conv2d(512, 256, 3, padding=1)
        self.b2 = nn.Conv2d(512, 1024, 3, padding=1)
        self.b3 = nn.Conv2d(384, 256, 3, padding=1)
        self.b4 = nn.Conv2d(192, 128, 3, padding=1)
        self.b5 = nn.Conv2d(128, 1024, 3, padding=1)
        self.b9 = nn.ConvTranspose2d(128, out_ch, kernel_size=4, stride=2, padding=1)



    def forward(self, x):
        c1 = self.conv1(x)
        #print("c1:", c1.size())
        cf9 = self.convf9(x)
        #print('cf9',cf9.size())

        p1 = self.pool1(c1)
        c2 = self.conv2(c1)
        p2 = self.pool2(c2)
        #print("c2:", c2.size())
        c3 = self.conv3(c2)
        p3 = self.pool3(c3)
        #print("c3:", c3.size())
        c4 = self.conv4(c3)
        p4 = self.pool4(c4)
        #print("c4:", c4.size())
        c5 = self.conv5(c4)
        #print("c5:", c5.size())

        center=self.center1(c5)
        center = self.center2(center)
        #print("center:", center.size())

        up_6 = self.up6(center)
        #up_6 = self.b1(up_6)  #（3）torch.cat([up_6, c4],up_6, c4的后两个通道大小一致即可，这里b1的作用只是为下一步up_7 = self.up7(c6)做准备(使得输入为512（256+256）)\
        # （其实可以直接修改self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)为self.up7 = nn.ConvTranspose2d(up_6-out_channel(不需要执行b1), 256, 2, stride=2);
        #up_9 = self.up9(c8)这一步也是这样做的）

        #print("up_6:",up_6.size())
        merge6 = torch.cat([up_6, c4], dim=1)
        #print("merge6",merge6.size())
        c6=merge6
        #c6 = self.conv6(merge6)


        up_7 = self.up7(c6)
        #print("up_7:", up_7.size())
        merge7 = torch.cat([up_7, c3], dim=1)
        #c7 = self.conv7(merge7)
        c7 = merge7
        #c7=self.b3(c7)
        up_8 = self.up8(c7)
        #print("up_8:", up_8.size())
        merge8 = torch.cat([up_8, c2], dim=1)
        #c8 = self.conv8(merge8)
        c8 = merge8
        #print("c8:", c8.size())
        up_9 = self.up9(c8)
        #up_9=self.pool1(up_9)
        #print("up_9:", up_9.size())

        merge9 = torch.cat([up_9, cf9], dim=1)
        c9 = merge9
        #print("c9:", c9.size())
        c10 = self.b9(c9)  # (4)通过逆卷积，还原为原始输入大小

        #print("c10:", c10.size())

        out = nn.Sigmoid()(c10)
        return out
'''

#(6)
'''
#Unet-rest34(pre_50)+self.convf9
from resnest.torch import resnest34
class Unet_resst34(nn.Module):
    def __init__(self, out_ch=1):
        super(Unet_resst34, self).__init__()

        self.conv = nn.Conv2d(512, 1024, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # self.encoder =resnest50(pretrained=False)
        self.encoder = resnest34(pretrained=True)
        # print('resnest50',self.encoder)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,  # （1）入口必须采用self.encoder.conv1(入口channel为3)
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)
        self.convf9 = nn.Sequential(self.encoder.conv1,  # （1）入口必须采用self.encoder.conv1(入口channel为3)
                                    self.encoder.bn1,
                                    self.encoder.relu
                                    )
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = self.encoder.layer1
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = self.encoder.layer2
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = self.encoder.layer3
        self.pool4 = nn.MaxPool2d(2)

        self.conv5 = self.encoder.layer4
        # print("self.conv5",self.conv5)

        # （2）中间连接带，之前这五步没有任何channels信息,\
        # 到这一步（进入解码之前）需要按照U-net规则为下一步做准备，即需要从self.conv5-out_channel（打印获取512） 到 self.up6-in_channel(1024)\
        # 整为512(self.conv5 = DoubleConv(512, 1024))
        # self.center = DoubleConv(512, 1024)
        self.center = Dblock(2048)
        # self.center2 = nn.Conv2d(2048, 1024, 3, padding=1)
        # self.center = nn.Conv2d(2048, 1024, 3, padding=1)
        # self.center = DecoderBlockV1(512, 512, 1024, True)

        # 逆卷积，也可以使用上采样
        self.up6 = nn.ConvTranspose2d(2048, 1024, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        # self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up7 = nn.ConvTranspose2d(2048, 256, 2, stride=2)  # 对应执行up_6 = self.b1(up_6)这一步时开启
        self.conv7 = DoubleConv(512, 256)
        # self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up8 = nn.ConvTranspose2d(768, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        # self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up9 = nn.ConvTranspose2d(384, 128, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)

        # 辅助调整大小和通道
        self.b1 = nn.Conv2d(512, 256, 3, padding=1)
        self.b2 = nn.Conv2d(512, 1024, 3, padding=1)
        self.b3 = nn.Conv2d(384, 256, 3, padding=1)
        self.b4 = nn.Conv2d(192, 128, 3, padding=1)

        # self.b6 = nn.ConvTranspose2d(192, 128, kernel_size=4, stride=2,
        # padding=1)
        # self.b7 = nn.ConvTranspose2d(128, out_ch, kernel_size=4, stride=2,
        # padding=1)

        self.b6 = nn.ConvTranspose2d(192, 192, kernel_size=4, stride=2,
                                     padding=1)
        self.b7 = nn.ConvTranspose2d(192, out_ch, kernel_size=4, stride=2,
                                     padding=1)
        self.b9 = nn.ConvTranspose2d(192, out_ch, kernel_size=4, stride=2, padding=1)
        self.b10 = nn.ConvTranspose2d(192, out_ch, kernel_size=4, stride=2, padding=1)
        self.b11 = nn.ConvTranspose2d(195, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        #print('x', x.size())
        c1 = self.conv1(x)
        #print("c1:", c1.size())
        cf9 = self.convf9(x)
        #print("c1:", cf9.size())
        p1 = self.pool1(c1)
        c2 = self.conv2(c1)
        p2 = self.pool2(c2)
        # print("c2:", c2.size())
        c3 = self.conv3(c2)
        p3 = self.pool3(c3)
        # print("c3:", c3.size())
        c4 = self.conv4(c3)
        p4 = self.pool4(c4)
        # print("c4:", c4.size())
        c5 = self.conv5(c4)
        # print("c5:", c5.size())

        center = self.center(c5)

        # print("center:", center.size())

        up_6 = self.up6(center)
        # up_6 = self.b1(up_6)  #（3）torch.cat([up_6, c4],up_6, c4的后两个通道大小一致即可，这里b1的作用只是为下一步up_7 = self.up7(c6)做准备(使得输入为512（256+256）)\
        # （其实可以直接修改self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)为self.up7 = nn.ConvTranspose2d(up_6-out_channel(不需要执行b1), 256, 2, stride=2);
        # up_9 = self.up9(c8)这一步也是这样做的）

        # print("up_6:",up_6.size())
        merge6 = torch.cat([up_6, c4], dim=1)
        # print("merge6",merge6.size())
        c6 = merge6
        # c6 = self.conv6(merge6)

        up_7 = self.up7(c6)
        # print("up_7:", up_7.size())
        merge7 = torch.cat([up_7, c3], dim=1)
        # c7 = self.conv7(merge7)
        c7 = merge7
        # c7=self.b3(c7)
        up_8 = self.up8(c7)
        # print("up_8:", up_8.size())
        merge8 = torch.cat([up_8, c2], dim=1)
        # c8 = self.conv8(merge8)
        c8 = merge8
        # print("c8:", c8.size())
        up_9 = self.up9(c8)
        # up_9=self.pool1(up_9)
        # print("up_9:", up_9.size())
        merge9 = torch.cat([up_9, cf9], dim=1)
        # c9 = self.conv9(merge9)
        c9 = merge9
        #print("c9:", c9.size())
        c10 = self.b9(c9)  # (4)通过逆卷积，还原为原始输入大小
        out = nn.Sigmoid()(c10)
        return out
'''

#(7)

#Unet-rest50(pre_50)+self.convf9
from resnest.torch import resnest34,resnest50
class Unet_resst50(nn.Module):
    def __init__(self, out_ch=1):
        super(Unet_resst50, self).__init__()

        self.conv = nn.Conv2d(512, 1024, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # self.encoder =resnest50(pretrained=False)
        self.encoder = resnest50(pretrained=False)
        # print('resnest50',self.encoder)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,  # （1）入口必须采用self.encoder.conv1(入口channel为3)
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)
        self.convf9 = nn.Sequential(self.encoder.conv1,  # （1）入口必须采用self.encoder.conv1(入口channel为3)
                                    self.encoder.bn1,
                                    self.encoder.relu
                                    )
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = self.encoder.layer1
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = self.encoder.layer2
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = self.encoder.layer3
        self.pool4 = nn.MaxPool2d(2)

        self.conv5 = self.encoder.layer4
        # print("self.conv5",self.conv5)

        # （2）中间连接带，之前这五步没有任何channels信息,\
        # 到这一步（进入解码之前）需要按照U-net规则为下一步做准备，即需要从self.conv5-out_channel（打印获取512） 到 self.up6-in_channel(1024)\
        # 整为512(self.conv5 = DoubleConv(512, 1024))
        # self.center = DoubleConv(512, 1024)
        self.center = Dblock(2048)
        # self.center2 = nn.Conv2d(2048, 1024, 3, padding=1)
        # self.center = nn.Conv2d(2048, 1024, 3, padding=1)
        # self.center = DecoderBlockV1(512, 512, 1024, True)

        # 逆卷积，也可以使用上采样
        self.up6 = nn.ConvTranspose2d(2048, 1024, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        # self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up7 = nn.ConvTranspose2d(2048, 256, 2, stride=2)  # 对应执行up_6 = self.b1(up_6)这一步时开启
        self.conv7 = DoubleConv(512, 256)
        # self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up8 = nn.ConvTranspose2d(768, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        # self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up9 = nn.ConvTranspose2d(384, 128, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)

        # 辅助调整大小和通道
        self.b1 = nn.Conv2d(512, 256, 3, padding=1)
        self.b2 = nn.Conv2d(512, 1024, 3, padding=1)
        self.b3 = nn.Conv2d(384, 256, 3, padding=1)
        self.b4 = nn.Conv2d(192, 128, 3, padding=1)

        # self.b6 = nn.ConvTranspose2d(192, 128, kernel_size=4, stride=2,
        # padding=1)
        # self.b7 = nn.ConvTranspose2d(128, out_ch, kernel_size=4, stride=2,
        # padding=1)

        self.b6 = nn.ConvTranspose2d(192, 192, kernel_size=4, stride=2,
                                     padding=1)
        self.b7 = nn.ConvTranspose2d(192, out_ch, kernel_size=4, stride=2,
                                     padding=1)
        self.b9 = nn.ConvTranspose2d(192, out_ch, kernel_size=4, stride=2, padding=1)
        self.b10 = nn.ConvTranspose2d(192, out_ch, kernel_size=4, stride=2, padding=1)
        self.b11 = nn.ConvTranspose2d(195, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        #print('x', x.size())
        c1 = self.conv1(x)
        #print("c1:", c1.size())
        cf9 = self.convf9(x)
        #print("c1:", cf9.size())
        p1 = self.pool1(c1)
        c2 = self.conv2(c1)
        p2 = self.pool2(c2)
        # print("c2:", c2.size())
        c3 = self.conv3(c2)
        p3 = self.pool3(c3)
        # print("c3:", c3.size())
        c4 = self.conv4(c3)
        p4 = self.pool4(c4)
        # print("c4:", c4.size())
        c5 = self.conv5(c4)
        # print("c5:", c5.size())

        center = self.center(c5)

        # print("center:", center.size())

        up_6 = self.up6(center)
        # up_6 = self.b1(up_6)  #（3）torch.cat([up_6, c4],up_6, c4的后两个通道大小一致即可，这里b1的作用只是为下一步up_7 = self.up7(c6)做准备(使得输入为512（256+256）)\
        # （其实可以直接修改self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)为self.up7 = nn.ConvTranspose2d(up_6-out_channel(不需要执行b1), 256, 2, stride=2);
        # up_9 = self.up9(c8)这一步也是这样做的）

        # print("up_6:",up_6.size())
        merge6 = torch.cat([up_6, c4], dim=1)
        # print("merge6",merge6.size())
        c6 = merge6
        # c6 = self.conv6(merge6)

        up_7 = self.up7(c6)
        # print("up_7:", up_7.size())
        merge7 = torch.cat([up_7, c3], dim=1)
        # c7 = self.conv7(merge7)
        c7 = merge7
        # c7=self.b3(c7)
        up_8 = self.up8(c7)
        # print("up_8:", up_8.size())
        merge8 = torch.cat([up_8, c2], dim=1)
        # c8 = self.conv8(merge8)
        c8 = merge8
        # print("c8:", c8.size())
        up_9 = self.up9(c8)
        # up_9=self.pool1(up_9)
        # print("up_9:", up_9.size())
        merge9 = torch.cat([up_9, cf9], dim=1)
        # c9 = self.conv9(merge9)
        c9 = merge9
        #print("c9:", c9.size())
        c10 = self.b9(c9)  # (4)通过逆卷积，还原为原始输入大小
        out = nn.Sigmoid()(c10)
        return out



#(9)
class MD_Unet_resst50(nn.Module):
    """
        UNet (https://arxiv.org/abs/1505.04597) with Resnet34(https://arxiv.org/abs/1512.03385) encoder
        Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/
        """

    def __init__(self, num_classes=1, num_filters=128, is_deconv=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with resnet34
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        #self.encoder = torchmodels.models.resnet50(pretrained=True)
        self.encoder = resnest50(pretrained=False)
        #print(self.encoder)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4

        self.center = DecoderBlockV2(2048, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.center2 = Dblock(1024)

        self.dec5 = DecoderBlockV2(2048 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(1024 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(512 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(256 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        print("conv1", conv1.size())
        conv2 = self.conv2(conv1)
        print("conv2", conv2.size())
        conv3 = self.conv3(conv2)
        print("conv3", conv3.size())
        conv4 = self.conv4(conv3)
        print("conv4", conv4.size())
        conv5 = self.conv5(conv4)
        print("conv5", conv5.size())

        center = self.center(self.pool(conv5))
        center = self.center2(center)
        print("center",center.size())

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        print("dec5", dec5.size())

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))

        print("dec4", dec4.size())

        dec3 = self.dec3(torch.cat([dec4, conv3], 1))

        print("dec3", dec3.size())

        dec2 = self.dec2(torch.cat([dec3, conv2], 1))

        print("dec2", dec2.size())

        dec1 = self.dec1(dec2)
        print("dec1", dec1.size())
        dec0 = self.dec0(dec1)
        print("dec0", dec0.size())



        if self.num_classes > 0:
            x_out = F.log_softmax(self.final(dec0), dim=1)
            print("x_out", x_out.size())
        else:
            x_out = self.final(dec0)
            print("x_out", x_out.size())


        return x_out


#######################################################



#（2-4）MD_ResstNet50
'''
class MD_ResstNet50(nn.Module):
    def __init__(self, num_classes=16, num_filters=2, is_deconv=True): #出现出现cuda out of memory，可通过设小num_filters=32来解决
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)
        self.encoder = resnest50(pretrained=False)#self.encoder =resnest34(pretrained=True)
        #print(self.encoder)
        #self.encoder = resnest34(pretrained=False)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4


        self.center1 = D_block_more_dilate(2048) #D_block_more_dilate代替了DecoderBlockV3中的卷积操作
        self.center2 = DecoderBlockV3(2048, 2048, num_filters * 8, is_deconv)#D_block_more_dilate代替了DecoderBlockV3中的卷积操作

        self.dec5 = DecoderBlockV2(2048 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(1024 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(512 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(256 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        #self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)
        self.final = nn.Sequential(
            nn.Conv2d(46, num_filters, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(num_filters, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        print("conv1", conv1.size())
        conv2 = self.conv2(conv1)
        print("conv2", conv2.size())
        conv3 = self.conv3(conv2)
        print("conv3", conv3.size())
        conv4 = self.conv4(conv3)
        print("conv4", conv4.size())
        conv5 = self.conv5(conv4)
        print("conv5", conv5.size())


        center = self.center1(self.pool(conv5))
        center = self.center2(center)
        print("center",center.size())

        #print(torch.cat([center, conv5], 1).size())
        dec5 = self.dec5(torch.cat([center, conv5], 1))
        print("dec5", dec5.size())
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        print("dec4", dec4.size())
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        print("dec3", dec3.size())
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        print("dec2", dec2.size())
        dec1 = self.dec1(dec2)
        print("dec1", dec1.size())
        #dec0 = self.dec0(dec1)
        #print("dec0", dec0.size())

        f = torch.cat((
            #F.upsample(dec1, scale_factor=2, mode='bilinear', align_corners=False),
            dec1,
            F.upsample(dec2, scale_factor=2, mode='bilinear', align_corners=False),
            F.upsample(dec3, scale_factor=4, mode='bilinear', align_corners=False),
            F.upsample(dec4, scale_factor=8, mode='bilinear', align_corners=False),
            F.upsample(dec5, scale_factor=16, mode='bilinear', align_corners=False),
        ), 1)
        print("f", f.size())
        f = F.dropout2d(f, p=0.50)
        print("f", f.size())


        x_out = self.final(f)
        x_out = nn.Sigmoid()(x_out)
        print("x_out",x_out)

        return x_out
'''


#（2-5）MD_ResstNet50--MLOSS

class MD_ResstNet50(nn.Module):
    def __init__(self, num_classes=1, num_filters=64, is_deconv=True): #出现出现cuda out of memory，可通过设小num_filters=32来解决
        
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)
        self.encoder = resnest50(pretrained=True)#self.encoder =resnest34(pretrained=True)
        #print(self.encoder)
        #self.encoder = resnest34(pretrained=False)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4


        self.center1 = D_block_more_dilate(2048) #D_block_more_dilate代替了DecoderBlockV3中的卷积操作
        self.center2 = DecoderBlockV3(2048, 2048, num_filters * 8, is_deconv)#D_block_more_dilate代替了DecoderBlockV3中的卷积操作

        self.dec5 = DecoderBlockV2(2048 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(1024 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(512 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(256 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        #self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)
        self.final_out = nn.Sequential(
            nn.Conv2d(5, num_filters, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(num_filters, 1, kernel_size=1, padding=0),##out_ch=1
        )
        self.final_1 = nn.Sequential(
            nn.Conv2d(num_filters, 1, kernel_size=1, padding=0),
        )
        self.final_2 = nn.Sequential(

            nn.Conv2d(num_filters * 2 * 2, 1, kernel_size=1, padding=0),
        )
        self.final_3 = nn.Sequential(

            nn.Conv2d(num_filters * 2, 1, kernel_size=1, padding=0),
        )
        self.final_4 = nn.Sequential(

            nn.Conv2d(num_filters * 8, 1, kernel_size=1, padding=0),
        )
        self.final_5 = nn.Sequential(
            nn.Conv2d(num_filters * 8, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        print("conv1", conv1.size())
        conv2 = self.conv2(conv1)
        print("conv2", conv2.size())
        conv3 = self.conv3(conv2)
        print("conv3", conv3.size())
        conv4 = self.conv4(conv3)
        print("conv4", conv4.size())
        conv5 = self.conv5(conv4)
        print("conv5", conv5.size())


        center = self.center1(self.pool(conv5))
        center = self.center2(center)
        print("center",center.size())

        #print(torch.cat([center, conv5], 1).size())
        dec5 = self.dec5(torch.cat([center, conv5], 1))
        print("dec5", dec5.size())
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        print("dec4", dec4.size())
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        print("dec3", dec3.size())
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        print("dec2", dec2.size())
        dec1 = self.dec1(dec2)
        print("dec1", dec1.size())
        #dec0 = self.dec0(dec1)
        #print("dec0", dec0.size())

        '''
        ###(1)第一种cat——way
        f = torch.cat((
            #F.upsample(conv1, scale_factor=2, mode='bilinear', align_corners=False),
            dec1,
            F.upsample(dec2, scale_factor=2, mode='bilinear', align_corners=False),
            F.upsample(dec3, scale_factor=4, mode='bilinear', align_corners=False),
            F.upsample(dec4, scale_factor=8, mode='bilinear', align_corners=False),
            F.upsample(dec5, scale_factor=16, mode='bilinear', align_corners=False),
        ), 1)
        print("f", f.size())
        f = F.dropout2d(f, p=0.50)
        print("f", f.size())


        x_out = self.final_out(f)
        print("x_out", x_out.size())


        dec1_out = self.final_1(dec1)
        print("dec1_out", dec1_out.size())
        dec2_out = self.final_2(dec2)
        print("dec2_out", dec2_out.size())
        dec3_out = self.final_3(dec3)
        print("dec3_out", dec3_out.size())
        dec4_out = self.final_4(dec4)
        print("dec4_out", dec4_out.size())
        dec5_out = self.final_5(dec5)
        print("dec5_out", dec5_out.size())
        '''

        ###(2)第二种cat——way（第一种num_filters=64出现cuda out of memory）
        dec1_out = self.final_1(dec1)
        print("dec1_out", dec1_out.size())
        dec2_out = self.final_2(dec2)
        print("dec2_out", dec2_out.size())
        dec3_out = self.final_3(dec3)
        print("dec3_out", dec3_out.size())
        dec4_out = self.final_4(dec4)
        print("dec4_out", dec4_out.size())
        dec5_out = self.final_5(dec5)
        print("dec5_out", dec5_out.size())

        f = torch.cat((
            # F.upsample(conv1, scale_factor=2, mode='bilinear', align_corners=False),
            dec1_out,
            F.upsample(dec2_out, scale_factor=2, mode='bilinear', align_corners=False),
            F.upsample(dec3_out, scale_factor=4, mode='bilinear', align_corners=False),
            F.upsample(dec4_out, scale_factor=8, mode='bilinear', align_corners=False),
            F.upsample(dec5_out, scale_factor=16, mode='bilinear', align_corners=False),
        ), 1)
        print("f1", f.size())
        f = F.dropout2d(f, p=0.50)
        print("f2", f.size())
        x_out = self.final_out(f)
        print("x_out", x_out.size())




        results = [dec1_out, dec2_out, dec3_out, dec4_out, dec5_out, x_out]
        results = [nn.Sigmoid()(r) for r in results]

        #print("results[0]",results[0])
        #results = [print('r',r.shape) for r in results]
        #results = [print('r', r) for r in results]
        #print("results", results[0])

        return results


# （2-6）MD_ResstNet34--MLOSS
"""
class MD_ResstNet34(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, is_deconv=True):  # 出现出现cuda out of memory，可通过设小num_filters=32来解决

        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)
        self.encoder = torchmodels.models.resnet34(pretrained=True)  # self.encoder =resnest34(pretrained=True)
        print(self.encoder)
        # self.encoder = resnest34(pretrained=False)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4

        self.center1 = D_block_more_dilate(512)  # D_block_more_dilate代替了DecoderBlockV3中的卷积操作
        self.center2 = DecoderBlockV3(512, 512, num_filters * 8,
                                      is_deconv)  # D_block_more_dilate代替了DecoderBlockV3中的卷积操作

        self.dec5 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(128 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(64 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        # self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)
        self.conv3_out = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1, padding=0),
        )

        self.final_out = nn.Sequential(
            nn.Conv2d(6, num_filters, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(num_filters, 1, kernel_size=1, padding=0),  ##out_ch=1
        )
        self.final_1 = nn.Sequential(
            nn.Conv2d(num_filters, 1, kernel_size=1, padding=0),
        )
        self.final_2 = nn.Sequential(

            nn.Conv2d(num_filters * 2 * 2, 1, kernel_size=1, padding=0),
        )
        self.final_3 = nn.Sequential(

            nn.Conv2d(num_filters * 2, 1, kernel_size=1, padding=0),
        )
        self.final_4 = nn.Sequential(

            nn.Conv2d(num_filters * 8, 1, kernel_size=1, padding=0),
        )
        self.final_5 = nn.Sequential(
            nn.Conv2d(num_filters * 8, 1, kernel_size=1, padding=0),
        )


    def forward(self, x):
        print("x", x.size())
        conv1 = self.conv1(x)
        print("conv1", conv1.size())
        conv2 = self.conv2(conv1)
        print("conv2", conv2.size())
        conv3 = self.conv3(conv2)
        print("conv3", conv3.size())
        conv4 = self.conv4(conv3)
        print("conv4", conv4.size())
        conv5 = self.conv5(conv4)
        print("conv5", conv5.size())

        center = self.center1(self.pool(conv5))
        print("center1", center.size())
        center = self.center2(center)
        print("center2", center.size())

        # print(torch.cat([center, conv5], 1).size())
        dec5 = self.dec5(torch.cat([center, conv5], 1))
        print("dec5", dec5.size())
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        print("dec4", dec4.size())
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        print("dec3", dec3.size())
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        print("dec2", dec2.size())
        dec1 = self.dec1(dec2)
        print("dec1", dec1.size())
        # dec0 = self.dec0(dec1)
        # print("dec0", dec0.size())

        '''
        ###(1)第一种cat——way
        f = torch.cat((
            #F.upsample(conv1, scale_factor=2, mode='bilinear', align_corners=False),
            dec1,
            F.upsample(dec2, scale_factor=2, mode='bilinear', align_corners=False),
            F.upsample(dec3, scale_factor=4, mode='bilinear', align_corners=False),
            F.upsample(dec4, scale_factor=8, mode='bilinear', align_corners=False),
            F.upsample(dec5, scale_factor=16, mode='bilinear', align_corners=False),
        ), 1)
        print("f", f.size())
        f = F.dropout2d(f, p=0.50)
        print("f", f.size())


        x_out = self.final_out(f)
        print("x_out", x_out.size())


        dec1_out = self.final_1(dec1)
        print("dec1_out", dec1_out.size())
        dec2_out = self.final_2(dec2)
        print("dec2_out", dec2_out.size())
        dec3_out = self.final_3(dec3)
        print("dec3_out", dec3_out.size())
        dec4_out = self.final_4(dec4)
        print("dec4_out", dec4_out.size())
        dec5_out = self.final_5(dec5)
        print("dec5_out", dec5_out.size())
        '''

        ###(2)第二种cat——way（第一种num_filters=64出现cuda out of memory）
        conv3_out=self.conv3_out(conv3)
        print("conv3_out", conv3_out.size())
        dec1_out = self.final_1(dec1)
        print("dec1_out", dec1_out.size())
        dec2_out = self.final_2(dec2)
        print("dec2_out", dec2_out.size())
        dec3_out = self.final_3(dec3)
        print("dec3_out", dec3_out.size())
        dec4_out = self.final_4(dec4)
        print("dec4_out", dec4_out.size())
        dec5_out = self.final_5(dec5)
        print("dec5_out", dec5_out.size())

        f = torch.cat((
            F.upsample(conv3_out, scale_factor=8, mode='bilinear', align_corners=False),
            dec1_out,
            F.upsample(dec2_out, scale_factor=2, mode='bilinear', align_corners=False),
            F.upsample(dec3_out, scale_factor=4, mode='bilinear', align_corners=False),
            F.upsample(dec4_out, scale_factor=8, mode='bilinear', align_corners=False),
            F.upsample(dec5_out, scale_factor=16, mode='bilinear', align_corners=False),
        ), 1)
        print("f1", f.size())
        f = F.dropout2d(f, p=0.50)
        print("f2", f.size())
        x_out = self.final_out(f)
        print("x_out", x_out.size())

        results = [dec1_out, dec2_out, dec3_out, dec4_out, dec5_out, x_out]
        #print('results.shape',results.shape)
        results = [nn.Sigmoid()(r) for r in results]

        # print("results[0]",results[0])
        # results = [print('r',r.shape) for r in results]
        # results = [print('r', r) for r in results]
        # print("results", results[0])

        return results
"""

# （2-7）MD_ResstNet34--MLOSS

class ASPP_V2(nn.Module):
    # have bias and relu, no bn
    def __init__(self, in_channel=512, depth=512):
        super().__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        #self.mean = nn.AdaptiveAvgPool2d((1, 1))

        self.atrous_block1 = nn.Sequential(nn.Conv2d(in_channel, depth, 1, 1),
                                           nn.ReLU(inplace=True))
        self.atrous_block6 = nn.Sequential(nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6),
                                           nn.ReLU(inplace=True))
        self.atrous_block12 = nn.Sequential(nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12),
                                            nn.ReLU(inplace=True))
        self.atrous_block18 = nn.Sequential(nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18),
                                            nn.ReLU(inplace=True))

        self.conv_1x1_output = nn.Sequential(nn.Conv2d(depth * 5, depth, 1, 1), nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[2:]
        image_features=x
        #image_features = self.mean(x)
        atrous_block1 = self.atrous_block1(x)
        #print('atrous_block1',atrous_block1.size())

        atrous_block6 = self.atrous_block6(x)
        #print('atrous_block6', atrous_block6.size())

        atrous_block12 = self.atrous_block12(x)
        #print('atrous_block12', atrous_block12.size())

        atrous_block18 = self.atrous_block18(x)
        #print('atrous_block18', atrous_block18.size())

        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        #print(" ",torch.cat([image_features, atrous_block1, atrous_block6,atrous_block12, atrous_block18], dim=1).size())
        #print('net', net.size())
        return net



class MMUU_Net(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, is_deconv=True):  # 出现出现cuda out of memory，可通过设小num_filters=32来解决

        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)
        self.encoder = torchmodels.models.resnet34(pretrained=True)  # self.encoder =resnest34(pretrained=True)
        #print(self.encoder)
        # self.encoder = resnest34(pretrained=False)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)
        self.convf1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu)


        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4

        self.center1 = ASPP_V2(512,512)  # D_block_more_dilate代替了DecoderBlockV3中的卷积操作
        self.center2 = DecoderBlockV3(512, 512, num_filters * 8,
                                      is_deconv)  # D_block_more_dilate代替了DecoderBlockV3中的卷积操作

        self.dec5 = DecoderBlockV2(512, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(384, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(320, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(128, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(128, 64)
        # self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)
        self.conv3_out = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1, padding=0),
        )

        self.final_out = nn.Sequential(
            nn.Conv2d(1024, num_filters, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(num_filters, 1, kernel_size=1, padding=0),  ##out_ch=1
        )
        self.final_1 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )
        self.final_2 = nn.Sequential(

            nn.Conv2d(num_filters * 2 * 2, 1, kernel_size=1, padding=0),
        )
        self.final_3 = nn.Sequential(

            nn.Conv2d(num_filters * 2, 1, kernel_size=1, padding=0),
        )
        self.final_4 = nn.Sequential(

            nn.Conv2d(num_filters * 8, 1, kernel_size=1, padding=0),
        )
        self.final_5 = nn.Sequential(
            nn.Conv2d(num_filters * 8, 1, kernel_size=1, padding=0),
        )
        self.final_center = nn.Sequential(
            nn.Conv2d(num_filters * 8, 1, kernel_size=1, padding=0),
        )


    def forward(self, x):
        #print("x", x.size())
        conv1 = self.conv1(x)
        #print("conv1", conv1.size())
        convf1=self.convf1(x)
        #print("convf1", convf1.size())
        conv2 = self.conv2(conv1)
        #print("conv2", conv2.size())
        conv3 = self.conv3(conv2)
        #print("conv3", conv3.size())
        conv4 = self.conv4(conv3)
        #print("conv4", conv4.size())
        conv5 = self.conv5(conv4)
        #print("conv5", conv5.size())

        #center = self.center1(self.pool(conv5))
        center = self.center1(conv5)
        #print("center1", center.size())
        center = self.center2(center)
        #print("center2", center.size())

        # print(torch.cat([center, conv5], 1).size())
        dec5 = self.dec5(torch.cat([center, conv4], 1))
        #print("dec5", dec5.size())
        dec4 = self.dec4(torch.cat([dec5, conv3], 1))
        #print("dec4", dec4.size())
        dec3 = self.dec3(torch.cat([dec4, conv2], 1))
        #print("dec3", dec3.size())
        dec2 = self.dec2(torch.cat([dec3, convf1], 1))
        #print("dec2", dec2.size())
        dec0 = self.dec0(dec2)
        #print("dec0", dec0.size())


        ###(2)第二种cat——way（第一种num_filters=64出现cuda out of memory）
        conv3_out=self.conv3_out(conv3)
        #print("conv3_out", conv3_out.size())
        dec0_out = self.final_1(dec0)
        #print("dec0_out", dec0_out.size())
        ##dec2_out = self.final_2(dec2)
        ##print("dec2_out", dec2_out.size())
        dec3_out = self.final_3(dec3)
        #print("dec3_out", dec3_out.size())
        dec4_out = self.final_4(dec4)
        #print("dec4_out", dec4_out.size())
        dec5_out = self.final_5(dec5)
        #print("dec5_out", dec5_out.size())
        center_out = self.final_center(center)
        #print("center_out", center_out.size())

        f = torch.cat((
            F.upsample(conv3, scale_factor=8, mode='bilinear', align_corners=False),
            dec0,
            #F.upsample(dec2_out, scale_factor=1, mode='bilinear', align_corners=False),
            F.upsample(dec3, scale_factor=2, mode='bilinear', align_corners=False),
            F.upsample(dec4, scale_factor=4, mode='bilinear', align_corners=False),
            F.upsample(dec5, scale_factor=8, mode='bilinear', align_corners=False),
            F.upsample(center, scale_factor=16, mode='bilinear', align_corners=False),
        ), 1)
        #print("f1", f.size())
        f = F.dropout2d(f, p=0.50)
        #print("f2", f.size())
        x_out = self.final_out(f)
        #print("x_out", x_out.size())

        results = [dec0_out, dec3_out, dec4_out, dec5_out,center_out,x_out]
        #print('results.shape',results.shape)
        results = [nn.Sigmoid()(r) for r in results]

        # print("results[0]",results[0])
        # results = [print('r',r.shape) for r in results]
        # results = [print('r', r) for r in results]
        # print("results", results[0])


        return results



