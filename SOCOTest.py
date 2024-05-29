import torch,os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from torch.autograd import Variable

from SFNet import Residual_Net
from data_loader import MyDataset
from tensorboardX import SummaryWriter
from tqdm import tqdm

import copy
import time

import matplotlib.pyplot as plt
import numpy as np
import torch,os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from torch.autograd import Variable
# from VGG import VGG16
from SFNet import Residual_Net
from data_loader import MyDataset
from tensorboardX import SummaryWriter
from tqdm import tqdm
from Mobile_vit import mobilevit_xxs
from Parnet import parnet_s
from vit import ViT
# writer = SummaryWriter('logs')  ##创建一个SummaryWriter的实例,默认目录名字为runs
from resnet import resnet34
from resnet import resnet101
# from DenseNet import DenseNet121
import warnings
from DenseNet import DenseNet121
from MyNet2 import DenseNet13

warnings.filterwarnings('ignore')

IMG_SIZE = 320
BATCH_SIZE= 32
CUDA=torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")

#其中，suppress=True 表示取消科学记数法，threshold=np.nan 完整输出（没有省略号)
np.set_printoptions(suppress=True)

##数据预处理
train_transforms = transforms.Compose([
    # transforms.Resize(IMG_SIZE),
    # transforms.RandomRotation(15),
    # transforms.RandomHorizontalFlip(p=0.5),  #随机水平翻转
    # transforms.RandomVerticalFlip(p=0.5), #随机垂直翻转
    # transforms.ColorJitter(brightness=0.2,contrast=0.1,saturation=0.1,hue=0.1),#参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
    transforms.ToTensor(),
])

val_transforms = transforms.Compose([
    # transforms.RandomRotation(15),
    # transforms.Resize(IMG_SIZE),
    # transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
])

#
# ##数据准备
# root_train = r"C:\data\SOCOFing_txt\gender\train_list.txt"
# root_test = r"C:\data\SOCOFing_txt\gender\test_list.txt"
# train_dataset = MyDataset(root_train,transform=train_transforms)
# test_dataset = MyDataset(root_test,transform=val_transforms)
# image_dataset = {'train':train_dataset, 'valid':test_dataset}
# image_dataloader = {x:DataLoader(image_dataset[x],batch_size=BATCH_SIZE,shuffle=True,num_workers=4) for x in ['train', 'valid']}
# # train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=4)
# # test_dataloader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=False,num_workers=4)
# dataset_sizes = {x:len(image_dataset[x]) for x in ['train', 'valid']}


#创建模型
# net = resnet34()
# net = resnet101()
net = DenseNet13(2)
# net = DenseNet121(2)
# net = Residual_Net()
net.to(DEVICE)


#模型保存
# filename=r'C:\data\SOCOmodel\DenseNet_224_2.pth'
filename = r'C:\data\SOCOmodel\MyNet_se_gender_7.pth'
# #输出结果保存路径
save_txt_path = r"C:\data\multi_model_fusion\Robustness_Testing\hard.txt"
f=open(save_txt_path,'w')

#加载模型
checkpoint = torch.load(filename)
best_acc = checkpoint['best_acc']
net.load_state_dict(checkpoint['state_dict'])

val_transforms = transforms.Compose([
    # transforms.Resize(IMG_SIZE),
    # transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
])

##数据准备
root_train = r"C:\data\SOCOFing_txt\gender\train_list.txt"
# root_test = r"C:\data\SOCOFing_txt\gender\test_list.txt"
root_test = r"C:\data\SOCOFing_txt\gender\Altered-Hard.txt"

# root_test = r"C:\data\grad-cam\socofing_test.txt"#测试一张图片

train_dataset = MyDataset(root_train,transform=train_transforms)
test_dataset = MyDataset(root_test,transform=val_transforms)
image_dataset = {'train':train_dataset, 'valid':test_dataset}
image_dataloader = {x:DataLoader(image_dataset[x],batch_size=1,shuffle=False,num_workers=4) for x in ['train', 'valid']}
# train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=4)
# test_dataloader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=False,num_workers=4)
dataset_sizes = {x:len(image_dataset[x]) for x in ['train', 'valid']}

def softmax(x):
    ex=np.exp(x)
    return ex/ex.sum()

# 验证过程
def val(model, device, test_loader):
    model.eval()
    correct = 0
    total_num = len(test_loader.dataset)
    # print(total_num, len(test_loader))
    with torch.no_grad():
        for data, target in test_loader:
            data, target = Variable(data).to(device), Variable(target).to(device)
            output = model(data)
            _, pred = torch.max(output.data, 1)
            # print(output[0][0].item(),output[0][1].item())
            # print(_.item())
            x = [output[0][0].item(),output[0][1].item()]  #第一个位置是预测为男性的概率，第二个位置为预测为女性的概率
            # print('%.20f'%softmax(x)[0])
            Confidence_level = '%20f'%softmax(x)[0]        #预测为男性的概率，置信度小于0.5则预测为女性，大于0.5则为男性
            Predictive_label = pred.item()          #预测标签，置信度小于0.5则预测为女性，大于0.5则为男性
            Authentic_labels = target.item()        #真实标签
            path = str('%.20f'%softmax(x)[0]) + ' ' + str(Predictive_label) + ' ' + str(Authentic_labels) + '\n'
            f.write(path)
            print('%.20f' % softmax(x)[0])
            correct += torch.sum(pred == target)
        correct = correct.data.item()
        acc = correct / total_num
        print('\nVal set: Accuracy: {}/{} ({:4f}%)\n'.format(
             correct, len(test_loader.dataset), 100 * acc))


if __name__ =='__main__':
    val(net, DEVICE, image_dataloader['valid'])








