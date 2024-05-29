import torch,os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from torch.autograd import Variable
# from Parnet import parnet_s
from SFNet import Residual_Net
from dataset_loader import MyDataset
from tensorboardX import SummaryWriter
from tqdm import tqdm
from Mobile_vit import mobilevit_xs
from vit import ViT
from resnet import resnet34
from resnet import resnet101
writer = SummaryWriter('logs')  ##创建一个SummaryWriter的实例,默认目录名字为runs

#设置超参数

normMean = [0.0015]
normStd = [0.0019]
normTransform = transforms.Normalize(normMean, normStd)
IMG_SIZE = 128
BATCH_SIZE= 16
CUDA=torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")

##数据预处理
train_transforms = transforms.Compose([
    # transforms.Resize(IMG_SIZE),
    # transforms.RandomResizedCrop(IMG_SIZE),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(30),
    transforms.ToTensor(),
    normTransform
])

val_transforms = transforms.Compose([
    # transforms.Resize(IMG_SIZE),
    # transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    normTransform
])

##数据准备
root_train = r"C:\data\SOCOFing_txt\gender\train_list.txt"
root_test = r"C:\data\SOCOFing_txt\gender\test_list.txt"
train_dataset = MyDataset(root_train,transform=train_transforms)
test_dataset = MyDataset(root_test,transform=val_transforms)
image_dataset = {'train':train_dataset, 'valid':test_dataset}
# image_dataloader = {x:DataLoader(image_dataset[x],batch_size=BATCH_SIZE,shuffle=True,num_workers=4) for x in ['train', 'valid']}
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=4)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False,num_workers=4)
dataset_sizes = {x:len(image_dataset[x]) for x in ['train', 'valid']}


##创建模型

# net = mobilevit_xs(img_size=(256, 256))
# model_ft = parnet_s(1, 2)
# model_ft.to(DEVICE)
# net=Residual_Net()


# net = ViT(
#         image_size = 640,
#         patch_size = 80,
#         num_classes = 2,
#         dim = 1024,
#         depth = 6,
#         heads = 16,
#         mlp_dim = 2048,
#         dropout = 0.1,
#         emb_dropout = 0.1
# )
# net.to(DEVICE)


# file_name = r'C:\data\model\Origin_SFNet_epoch_15_13_ACC-0.7892.pth'
# net=Residual_Net()
net = resnet101()
# net.load_state_dict(torch.load(file_name))
net.to(DEVICE)

# # load params
# pretrained_dict = torch.load(r'C:\data\model\Pretreatment-ResNet34epoch_7_ACC-0.7452.pth')
# # 获取当前网络的dict
# net_state_dict = net.state_dict()
# # 剔除不匹配的权值参数
# pretrained_dict_1 = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
# # 更新新模型参数字典
# net_state_dict.update(pretrained_dict_1)
# # 将包含预训练模型参数的字典"放"到新模型中
# net.load_state_dict(net_state_dict)

##损失函数和优化器

criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model_ft.parameters(), lr=1e-3)#指定 新加的fc层的学习率
optimizer = torch.optim.Adam(net.parameters(), lr=0.01,weight_decay=0.005)#指定 新加的fc层的学习率
cosine_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=20,eta_min=1e-9)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 4, gamma = 0.6, last_epoch=-1)




def train(epoch, model, train_loader):
    # # 训练模式
    # file_name = r'C:\data\model\Pretreatment-Origin_SFNet_epoch_15_1_ACC-0.7937.pth'
    # # model_ft.load_state_dict(torch.load(file_name))
    # net.load_state_dict(torch.load(file_name))
    # # print('模型载入成功\n')
    # torch.cuda.empty_cache()

    correct = 0.0
    total = 0.0
    sum_loss = 0.0

    model.train()
    loop = tqdm(train_loader, desc='Train')
    for x, y in loop:
        x, y = x.to(DEVICE), y.to(DEVICE)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            y_ = torch.argmax(y_pred, dim=1)
            correct += (y_ == y).sum().item()
            total += y.size(0)
            sum_loss += loss.item()
            running_loss = sum_loss / total
            running_acc = correct / total

        # 更新训练信息
        loop.set_description(f'Epoch [{epoch + 1}/{epochs}]')
        loop.set_postfix(loss=running_loss, acc=running_acc)

    epoch_loss = sum_loss / total
    epoch_acc = correct / total
    # writer.add_scalar('data/trainloss', sum_loss, epoch)
    # writer.add_scalar('data/trainacc', epoch_acc, epoch)
    return epoch_loss, epoch_acc


  # # 测试模式

def test(epoch, model, test_loader):

    model.eval()
    test_correct = 0.0
    test_total = 0.0
    test_sum_loss = 0.0

    with torch.no_grad():

        loop2 = tqdm(test_loader, desc='Test')
        for x, y in loop2:
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            y_ = torch.argmax(y_pred, dim=1)
            test_correct += (y_ == y).sum().item()
            test_total += y.size(0)
            test_sum_loss += loss.item()
            test_running_loss = test_sum_loss / test_total
            test_running_acc = test_correct / test_total

            # 更新测试信息
            loop2.set_postfix(loss=test_running_loss, acc=test_running_acc)

        test_epoch_loss = test_sum_loss / test_total
        test_epoch_acc = test_correct / test_total
        # writer.add_scalar('data/valloss', test_epoch_loss, epoch)
        # writer.add_scalar('data/valacc', test_epoch_acc, epoch)

    # writer.close()
    return test_epoch_loss, test_epoch_acc




# 训练

if __name__ =='__main__':
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    epochs = 20
    min_acc = 0
    for epoch in range(epochs):
        epoch_loss, epoch_acc = train(epoch, net, train_dataloader)
        # scheduler.step()
        cosine_schedule.step()
        test_epoch_loss, test_epoch_acc = test(epoch, net, test_dataloader)
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)
        test_loss.append(test_epoch_loss)
        test_acc.append(test_epoch_acc)
        if test_epoch_acc > min_acc:
            modelfile = 'C:\data\model\\' + 'Pretreatment-ResNet101_' + 'epoch_' + str(epoch + 1) + '_' + 'ACC-' + str(
                round(test_epoch_acc, 4)) + '.pth'
            torch.save(net.state_dict(), modelfile)
            min_acc = test_epoch_acc
    net.load_state_dict(torch.load(modelfile))


