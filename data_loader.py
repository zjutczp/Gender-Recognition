import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from PIL import ImageEnhance
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from skimage import measure, filters
import cv2
import matplotlib.pyplot as plt
# 8.25：指纹图像处理（高通滤波+阈值处理）
def gaussHighPassFilter(shape, radius=10):  # 高斯高通滤波器
        # 高斯滤波器：# Gauss = 1/(2*pi*s2) * exp(-(x**2+y**2)/(2*s2))
        u, v = np.mgrid[-1:1:2.0/shape[0], -1:1:2.0/shape[1]]
        D = np.sqrt(u**2 + v**2)
        D0 = radius / shape[0]
        kernel = 1 - np.exp(- (D ** 2) / (2 *D0**2))
        return kernel

def dft2Image(image):  #　最优扩充的快速傅立叶变换
        # 中心化, centralized 2d array f(x,y) * (-1)^(x+y)
        mask = np.ones(image.shape)
        mask[1::2, ::2] = -1
        mask[::2, 1::2] = -1
        fImage = image * mask  # f(x,y) * (-1)^(x+y)

        # 最优 DFT 扩充尺寸
        rows, cols = image.shape[:2]  # 原始图片的高度和宽度
        rPadded = cv2.getOptimalDFTSize(rows)  # 最优 DFT 扩充尺寸
        cPadded = cv2.getOptimalDFTSize(cols)  # 用于快速傅里叶变换

        # 边缘扩充(补0), 快速傅里叶变换
        dftImage = np.zeros((rPadded, cPadded, 2), np.float32)  # 对原始图像进行边缘扩充
        dftImage[:rows, :cols, 0] = fImage  # 边缘扩充，下侧和右侧补0
        cv2.dft(dftImage, dftImage, cv2.DFT_COMPLEX_OUTPUT)  # 快速傅里叶变换
        return dftImage


def imgHPFilter(image, D0=50):  #　图像高通滤波
        rows, cols = image.shape[:2]  # 图片的高度和宽度
        # 快速傅里叶变换
        dftImage = dft2Image(image)  # 快速傅里叶变换 (rPad, cPad, 2)
        rPadded, cPadded = dftImage.shape[:2]  # 快速傅里叶变换的尺寸, 原始图像尺寸优化

        # 构建 高斯高通滤波器 (Gauss low pass filter)
        hpFilter = gaussHighPassFilter((rPadded, cPadded), radius=D0)  # 高斯高通滤波器

        # 在频率域修改傅里叶变换: 傅里叶变换 点乘 高通滤波器
        dftHPfilter = np.zeros(dftImage.shape, dftImage.dtype)  # 快速傅里叶变换的尺寸(优化尺寸)
        for j in range(2):
            dftHPfilter[:rPadded, :cPadded, j] = dftImage[:rPadded, :cPadded, j] * hpFilter

        # 对高通傅里叶变换 执行傅里叶逆变换，并只取实部
        idft = np.zeros(dftImage.shape[:2], np.float32)  # 快速傅里叶变换的尺寸(优化尺寸)
        cv2.dft(dftHPfilter, idft, cv2.DFT_REAL_OUTPUT + cv2.DFT_INVERSE + cv2.DFT_SCALE)

        # 中心化, centralized 2d array g(x,y) * (-1)^(x+y)
        mask2 = np.ones(dftImage.shape[:2])
        mask2[1::2, ::2] = -1
        mask2[::2, 1::2] = -1
        idftCen = idft * mask2  # g(x,y) * (-1)^(x+y)

        # 截取左上角，大小和输入图像相等
        result = np.clip(idftCen, 0, 255)  # 截断函数，将数值限制在 [0,255]
        imgHPF = result.astype(np.uint8)
        imgHPF = imgHPF[:rows, :cols]
        return imgHPF


# 变暗
def Darker(image,percetage=0.9):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    #get darker
    for xi in range(0,w):
        for xj in range(0,h):
            image_copy[xj,xi,0] = int(image[xj,xi,0]*percetage)
            image_copy[xj,xi,1] = int(image[xj,xi,1]*percetage)
            image_copy[xj,xi,2] = int(image[xj,xi,2]*percetage)
    return image_copy

###预处理：剪裁512*512，灰度拉伸，自适应直方图均衡化，自适应阈值法进行二值化

# 以torch.utils.data.Dataset为基类创建MyDataset
class MyDataset(Dataset):
    # stpe1:初始化
    def __init__(self, img_path, transform=None):
        super(MyDataset, self).__init__()
        self.root = img_path
        f = open(self.root, 'r')  # 打开标签文件
        data = f.readlines()

        imgs = []
        sexlabels = []
        # angles = []
        for line in data:   # 遍历标签文件每行
            line = line.rstrip()    # 删除字符串末尾的空格
            word = line.split()     # 通过空格分割字符串，变成列表
            # imgs.append((word[0], int(word[1]), int(word[2])))  # 把图片名words[0]，标签int(words[1])放到imgs里
            imgs.append((word[0], int(word[1])))  # 把图片名words[0]，标签int(words[1])放到imgs里
            sexlabels.append(word[1])  #把性别标签word[1]放在labels里
            # angles.append(word[2])


        self.img = imgs
        self.label = sexlabels
        # self.angle = angles
        # self.label = agelabels
        self.transform = transform


    def __len__(self):
        return len(self.label)
        return len(self.img)

    def __getitem__(self, item):    # 检索函数
        # img, sexlabel , agelabel,angle = self.img[item]
        # img, sexlabel, angle = self.img[item]
        img, sexlabel = self.img[item]
        imgo = cv2.imread(img, 0)
        imgo = cv2.resize(imgo,(224,224))
        # img = Image.open(img).convert('L')  #转为RGB图像，如果为L则为灰度图
        # img = np.array(img)
        # print('Y')
        img = imgo
        rows, cols = img.shape[:2]
        img = imgHPFilter(img, D0=50)
        img = np.clip(img, 0, 1)
      # # 对图像进行自适应直方图均衡化
      #   clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
      #   img = clahe.apply(img)
      #
      #   # 使用自适应阈值法进行二值化
      #   img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 4)
      #
        img = Image.fromarray(img)      #angle代表旋转角度，0代表不变，1代表90，2代表180，3代表270，相当于数据集扩大了四倍


        # if angle==1:
        #         img = ImageEnhance.Brightness(img)# 亮度增强
        #         brightness = 1.5
        #         img = img.enhance(brightness)
        # elif angle==2:
        #     img = ImageEnhance.Contrast(img)    #对比度增强
        #     contrast = 1.5
        #     img = img.enhance(contrast)
        # elif angle==3:
        #     img = ImageEnhance.Sharpness(img)
        #     sharpness = 3.0
        #     img = img.enhance(sharpness)
        # 此时img是PIL.Image类型   label是str类型
        # elif angle==4:
        #     img = img.rotate(90)
        # elif angle==5:
        #     img = img.rotate(180)
        # elif angle==6:
        #     img = img.rotate(270)
        if self.transform is not None:
            img = self.transform(img)
            # print(img.max())

        return img, sexlabel
#
# #数据准备
# root_train = r"C:\data\SOCOFing_txt\gender\train_list.txt"
# root_test = r"C:\data\SOCOFing_txt\gender\test_list.txt"
# train_dataset = MyDataset(root_train)
# test_dataset = MyDataset(root_test)
# image_dataset = {'train':train_dataset, 'valid':test_dataset}
# dataset_sizes = {x:len(image_dataset[x]) for x in ['train', 'valid']}
#
# img,_  = train_dataset[7]
# print(img.size)
# img = np.asarray(img)
#
# plt.subplot(133), plt.imshow(img, 'gray'), plt.title('Threshold'), plt.xticks([]), plt.yticks([])
# plt.show()
