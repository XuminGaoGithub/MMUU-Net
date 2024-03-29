#MMUU-NET(MMUU-Net：A Robust and Effective Network for Farmland Segmentation of Satellite Imagery):

#Authur：Xumin Gao, et al (Beijing Mcfly Technology Co. Ltd)

#E-mail: comin15071460998@gmail.com

#Paper: 

Xumin Gao, et al. MMUU-Net: A Robust and Effective Network for Farmland Segmentation of Satellite Imagery[J]. Journal of Physics: Conference Series, 2020, 1651(1):012189 (7pp).


#Demo video:

1)https://www.bilibili.com/video/BV1Ma4y1W72W

2)https://www.bilibili.com/video/BV1fi4y1M7vU/


# Requirements
- Cuda 10.1
- Python 3.7
- Pytorch 1.3.0
- Opencv 4.1.1
- scipy 1.2.1
- imageio 2.3.0
- visdom 0.1.8.9

# Usage

### Data
Place '*train*', '*valid*' and '*test*' data folders in the '*dataset*' folder.

### Train
- Run `python train.py` to train the MMUU_NET.

### Predict
- Run `python test.py` to predict on the MMUU_NET.

### MMUU_Net

MMUU-Net/networks/MMUU_Net.py


### Download trained MMUU-NET [百度网盘]:

链接(Link)：https://pan.baidu.com/s/1G8WhL5NEbaj-gyQOojgEoA 
提取码(Password)：iat0 


#Abstract

The MMUU-Net, which is a robust and effective network for satellite imagery segmentation. The encoder in the U-Net was replaced by ResNeSt, which can greatly improve classification accuracy. An ASPP layer was added in the middle. A multi-scale feature fusion module was designed in the decoder and a corresponding robust loss function was designed to improve multi- scale information fusion. Finally, in order to eliminate the adhesion phenomenon of preliminary segmentationa，a two-stage segmentation strategy including the coarse segmentation and the refined segmentation was proposed. The MIoU of MMUU-Net was improved by 10.91% compared with that of U-Net.

1.MMUU-Net

![Image text](https://github.com/XuminGaoGithub/MMUU-Net/blob/main/1.png)

2.The comparison results of  different segmentation network

![Image text](https://github.com/XuminGaoGithub/MMUU-Net/blob/main/2.png)
 (a) Original image, (b) Ground truth, (c) U-Net, (d) D-LinkNet, (e) RCF, (f) MMUU-Net (The first stage), (g) MMUU-Net (The second stage)



3.Farmland segmentation and area calculation using satellite imagery

![Image text](https://github.com/XuminGaoGithub/MMUU-Net/blob/main/3.png)

4.On the basis of farmland segmentation, we use clustering algorithm for crop classification

![Image text](https://github.com/XuminGaoGithub/MMUU-Net/blob/main/4.png)

