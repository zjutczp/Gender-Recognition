import copy
import time
import torchvision.models
import matplotlib.pyplot as plt
import numpy as np
import torch,os
import torch.nn as nn
import torchvision.models as models
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
# from DenseNet import DenseNet13
# from DenseNet import DenseNet264
# from MyNet import DenseNet121
# from MyNet import DenseNet13
from MyNet8 import DenseNet13
import warnings
from torchtoolbox.transform import Cutout

warnings.filterwarnings('ignore')
IMG_SIZE = 320
BATCH_SIZE= 32
CUDA=torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")

##数据预处理
train_transforms = transforms.Compose([
    # transforms.Resize(IMG_SIZE),
    # transforms.CenterCrop(196),
    # transforms.RandomRotation(5),
    # Cutout(),
    # transforms.RandomHorizontalFlip(p=0.5),  #随机水平翻转
    # transforms.RandomVerticalFlip(p=0.5), #随机垂直翻转
    # transforms.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1,hue=0.1),#参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
    transforms.ToTensor(),
])

val_transforms = transforms.Compose([
    # transforms.Resize(IMG_SIZE),
    # transforms.CenterCrop(IMG_SIZE),
    # transforms.RandomRotation(15),
    transforms.ToTensor(),
])


##数据准备

#性别
root_train = r"C:\data\SOCOFing_txt\gender\train_list.txt"
root_test = r"C:\data\SOCOFing_txt\gender\test_list.txt"
# #左右手
# root_train = r"C:\data\SOCOFing_txt\left_or_right\train_list_left_right.txt"
# root_test = r"C:\data\SOCOFing_txt\left_or_right\test_list_left_right.txt"
 
# #具体手指
# root_train = r"C:\data\SOCOFing_txt\individual_finger\train_list_finger.txt"
# root_test = r"C:\data\SOCOFing_txt\individual_finger\test_list_finger.txt"


train_dataset = MyDataset(root_train,transform=train_transforms)
test_dataset = MyDataset(root_test,transform=val_transforms)
image_dataset = {'train':train_dataset, 'valid':test_dataset}
image_dataloader = {x:DataLoader(image_dataset[x],batch_size=BATCH_SIZE,shuffle=True,num_workers=4) for x in ['train', 'valid']}
# train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=4)
# test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False,num_workers=4)
dataset_sizes = {x:len(image_dataset[x]) for x in ['train', 'valid']}


#创建模型
# net = resnet34()
# net = resnet101()
# net = DenseNet121(2)
# net = DenseNet13(2)
# net = DenseNet121(2)
# net = torchvision.models.shufflenet_v2_x0_5(pretrained=True)
# fc_features =  net.fc.in_features
# net.fc = nn.Linear(fc_features, 2)
# net.conv1[0] = nn.Conv2d(1 ,24, kernel_size=(3, 3) ,stride=(2, 2), padding=(1, 1), bias=False)
net = mobilevit_xxs(img_size = (256, 256))


# net = Residual_Net()
net.to(DEVICE)


#模型保存
filename=r'C:\data\SOCOmodel\mobilevit_xxs.pth'


##自定义实现多分类损失函数+L2正则化

class MyLoss(torch.nn.Module):
    def __init__(self,weight_decay=0.01):
        super(MyLoss, self).__init__()
        self.weight_decay = weight_decay

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets)
        l2_loss = torch.tensor(0., requires_grad=True).to(inputs.device)
        for name, param in self.named_parameters():
            if 'weight' in name:
                l2_loss+=torch.norm(param)
        loss = ce_loss + self.weight_decay * l2_loss
        return loss



#优化器设置
# criterion = nn.CrossEntropyLoss()
criterion = MyLoss()
# optimizer = torch.optim.Adam(net.parameters(), lr=0.001,weight_decay=0.01)
optimizer = torch.optim.SGD(net.parameters(),lr=0.01,momentum=0.9,weight_decay=0.001)
# optimizer = torch.optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1,last_epoch=-1)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=30,eta_min=1e-9)

#训练模块
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception = False, filename=filename):
    since = time.time()
    best_acc = 0


    checkpoint = torch.load(filename)
    best_acc = checkpoint['best_acc']
    model.load_state_dict(checkpoint['state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer'])

    # model.class_to_idx = checkpoint['mapping']

    model.to(DEVICE)


    val_acc_history = []
    train_acc_history = []
    train_losses = []
    valid_losses = []
    LRs = [optimizer.param_groups[0]['lr']]

    best_model_wts = copy.deepcopy(model.state_dict())


    for epoch in range(num_epochs):
        # print('Epoch{}/{}'.format(epoch,num_epochs-1))
        # print('-'*10)


        #训练和验证

        for phase in ['train','valid']:
            if phase == 'train':
                model.train()   #训练
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            loop = tqdm(dataloaders[phase],desc=phase)
            #把所有数据都取个遍
            for inputs,labels in loop:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)



                #清零
                optimizer.zero_grad()

                #只有训练的时候计算和跟新梯度

                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase =='train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    #训练阶段更新权重

                    if phase=='train':
                        loss.backward()
                        optimizer.step()



                #训练损失
                running_loss += loss.item()*inputs.size(0)
                running_corrects +=torch.sum(preds == labels.data)
                loop.set_description(f'Epoch [{epoch + 1}/{num_epochs}]')
                loop.set_postfix(loss=running_loss/len(dataloaders[phase].dataset), acc=(running_corrects.double()/len(dataloaders[phase].dataset)).item())


            epoch_loss = running_loss/len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double()/len(dataloaders[phase].dataset)


            # time_elapsed = time.time() - since

            # print('Time elasped {:.0f}m {:.0f}s'.format(time_elapsed //60, time_elapsed%60))
            # print('{}Loss: {:.4f} ACC: {:.4f}'.format(phase, epoch_loss, epoch_acc))


            ##得到最好那次的模型

            if phase =='valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                state = {
                    'state_dict':model.state_dict(),
                    'best_acc':best_acc,
                    'optimizer':optimizer.state_dict(),
                }
                torch.save(state, filename)
            if phase =='valid':
                val_acc_history.append(epoch_acc)
                valid_losses.append(epoch_loss)
                scheduler.step(epoch_loss)
            if phase =='train':
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)

        # print('Optimizer learning rate: {:.7f}'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])
        # print()


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed //60, time_elapsed%60))
    print('Best val Acc: {:4f}'.format(best_acc))

    #训练完后用最好的一次当模型的最终结果
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs

if __name__ =='__main__':

    net, val_acc_history, train_acc_history, valid_losses, train_losses, LRs = train_model(net, image_dataloader, criterion, optimizer, num_epochs=20, is_inception=False,filename=filename)
    x_train = range(len(train_losses))
    x_valid = range(len(valid_losses))

    fig = plt.figure(figsize=(7, 5))
    ax1 = fig.add_subplot(1, 1, 1)

    plt.plot(x_train, train_losses, 'r-', label=u'train')
    plt.plot(x_valid, valid_losses, 'b-', label=u'valid')
    plt.legend()

    plt.xlabel(u'epoch')
    plt.ylabel(u'loss')
    plt.title('Compare loss for train or valid')
    plt.show()







