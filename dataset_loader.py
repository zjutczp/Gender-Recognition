import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from skimage import measure, filters
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
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
###预处理：剪裁512*512，灰度拉伸，自适应直方图均衡化，自适应阈值法进行二值化

def crop_with_gravity(img, size=[224,224]):
    img_ori = img

    if img.shape[-1] == 3:                                                #3通道
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)                       #灰度化
    if img.min() == img.max():                                            #？
        return img[:size[0], :size[1]]
    #thresh = filters.threshold_otsu(img)
    #dst = (img<thresh) * 1.0
    dst = 255 -img                                                        #灰度拉伸
    m = measure.moments(dst, order=2)
    # y
    cr = m[1, 0] / m[0, 0]
    # x
    cc = m[0, 1] / m[0, 0]
    [H, W] = img.shape
    [h, w] = size
    [y_h, x_w] = [cr, cc]
    y_1 = y_h - h / 2.; x_1 = x_w - w / 2.
    y_1, x_1 = np.ceil(y_1).astype(int), np.ceil(x_1).astype(int)
    y_2 = y_1 + h;      x_2 = x_1 + w
    #print(x_1, y_1)
    if x_1 < 0:
        x_1 = 0; x_2 = w
    if y_1 < 0:
        y_1 = 0; y_2 = h
    if x_2 > W - 1:
        x_1 = W - w; x_2 = W
    if y_2 > H - 1:
        y_1 = H - h; y_2 = H
    dst = img_ori[y_1:y_2, x_1:x_2]
    # print(x_1, y_1)
    return dst

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
        agelabels = []
        # angles = []

        for line in data:   # 遍历标签文件每行
            line = line.rstrip()    # 删除字符串末尾的空格
            word = line.split()     # 通过空格分割字符串，变成列表
            # imgs.append((word[0],int(word[1]),int(word[2]),int(word[-1])))    # 把图片名words[0]，标签int(words[1]),int(words[1])放到imgs里,把角度word[3]也放进imgs里
            imgs.append((word[0], int(word[1]), int(word[2])))  # 把图片名words[0]，标签int(words[1])放到imgs里,把角度word[3]也放进imgs里
            sexlabels.append(word[1])  #把性别标签word[1]放在labels里
            agelabels.append(word[2])  #把年龄标签word[2]放在labels里
            # angles.append(word[-1])

        self.img = imgs
        self.label1 = sexlabels
        # self.angle = angles
        self.label2 = agelabels
        self.transform = transform


    def __len__(self):
        return len(self.label1)
        return len(self.label2)
        return len(self.img)

    def __getitem__(self, item):    # 检索函数
        # img, sexlabel , agelabel,angle = self.img[item]
        img, sexlabel, agelabel = self.img[item]
        imgo = cv2.imread(img, 0)
        # img = Image.open(img).convert('L')  #转为RGB图像，如果为L则为灰度图
        # img = np.array(img)
        imgo = crop_with_gravity(imgo, size=[400,400])

        # print('Y')

        img = imgo
        rows, cols = img.shape[:2]
        img = imgHPFilter(img, D0=50)
        img = np.clip(img, 0, 1)
        # 对图像进行自适应直方图均衡化
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # img = clahe.apply(img)
        #
        # # 使用自适应阈值法进行二值化
        # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 4)

        img = Image.fromarray(img)      #angle代表旋转角度，0代表不变，1代表90，2代表180，3代表270，相当于数据集扩大了四倍
        # if angle==1:
        #     img = img.rotate(90)  # 翻转图片
        # elif angle==2:
        #     img = img.rotate(180)  # 翻转图片
        # elif angle==3:
        #     img = img.rotate(270)  # 翻转图片
        # # 此时img是PIL.Image类型   label是str类型

        if self.transform is not None:
            img = self.transform(img)
            # print(img.max())

        return img, sexlabel


#
# ##数据准备
# root_train = r"C:\data\txt\train_list_not_random.txt"
# root_test = r"C:\data\txt\test_list_not_random.txt"
# train_dataset = MyDataset(root_train,transform=None)
# test_dataset = MyDataset(root_test,transform=None)
# image_dataset = {'train':train_dataset, 'valid':test_dataset}
# # image_dataloader = {x:DataLoader(image_dataset[x],batch_size=BATCH_SIZE,shuffle=True,num_workers=4) for x in ['train', 'valid']}
# train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True,num_workers=0)
# test_dataloader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True,num_workers=0)
# # def get_mean_std(loader):
# #     # Var[x] = E[X**2]-E[X]**2
# #     channels_sum,channels_squared_sum,num_batches = 0,0,0
# #     for data, _ in tqdm(loader):
# #         channels_sum += torch.mean(data, dim=[0,2,3])
# #         channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
# #         num_batches += 1
# #
# #     print(num_batches)
# #     # print(channels_sum)
# #     mean = channels_sum/num_batches
# #     std = (channels_squared_sum/num_batches - mean**2) **0.5
# #
# #     return mean,std
# #
# # mean,std = get_mean_std(test_dataloader)
# #
# # print(mean)
# # print(std)
#
#
#
# img, sexlabel = train_dataset[22245]
#
# # img.show()
# print(img)
# print(sexlabel)
#
# img = np.asarray(img)
# print(img.shape)
#
# plt.imshow(img, 'gray'), plt.title('Threshold'), plt.xticks([]), plt.yticks([])
# plt.show()
