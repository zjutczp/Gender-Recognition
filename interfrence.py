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
writer = SummaryWriter('logs')  ##创建一个SummaryWriter的实例,默认目录名字为runs

#设置超参数


IMG_SIZE = 300
BATCH_SIZE= 32
CUDA=torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")

##数据预处理
train_transforms = transforms.Compose([
    # transforms.Resize(IMG_SIZE),
    # transforms.RandomResizedCrop(IMG_SIZE),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(30),
    transforms.ToTensor(),
])

val_transforms = transforms.Compose([
    # transforms.Resize(IMG_SIZE),
    # transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
])

##数据准备
root_train = r"C:\data\txt\train_list_not_random.txt"
root_test = r"C:\data\txt\test_list_not_random.txt"
train_dataset = MyDataset(root_train,transform=train_transforms)
test_dataset = MyDataset(root_test,transform=val_transforms)
image_dataset = {'train':train_dataset, 'valid':test_dataset}
image_dataloader = {x:DataLoader(image_dataset[x],batch_size=BATCH_SIZE,shuffle=True) for x in ['train', 'valid']}
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


file_name = r'C:\data\model\SFNet_epoch_27_ACC-0.787.pth'
net=Residual_Net()
net.load_state_dict(torch.load(file_name))
# print('模型载入成功')
net.to(DEVICE)


##损失函数和优化器

criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model_ft.parameters(), lr=1e-3)#指定 新加的fc层的学习率
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)#指定 新加的fc层的学习率
# cosine_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=20,eta_min=1e-9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 4, gamma = 0.6, last_epoch=-1)




def train(epoch, model, train_loader):
    # 训练模式
    # file_name = r'C:\data\model\SFNet_epoch_24_ACC-0.786.pth'
    # model_ft.load_state_dict(torch.load(file_name))
    # net.load_state_dict(torch.load(file_name))
    # print('模型载入成功')
    # torch.cuda.empty_cache()

    correct = 0
    total = 0
    sum_loss = 0

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
    writer.add_scalar('data/trainloss', sum_loss, epoch)
    writer.add_scalar('data/trainacc', epoch_acc, epoch)
    return epoch_loss, epoch_acc


  # # 测试模式
ACC=0

def test(epoch, model, test_loader):
    global ACC
    test_correct = 0
    test_total = 0
    test_sum_loss = 0
    model.eval()

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
        writer.add_scalar('data/valloss', test_epoch_loss, epoch)
        writer.add_scalar('data/valacc', test_epoch_acc, epoch)
        if test_epoch_acc > ACC:
            modelfile = 'C:\data\model\\' + 'SFNet_' + 'epoch_27_' +  str(epoch+1) + '_' + 'ACC-' + str(
                round(test_epoch_acc, 3)) + '.pth'
            # torch.save(model_ft.state_dict(), modelfile)
            torch.save(net.state_dict(), modelfile)
            ACC = test_epoch_acc
    writer.close()
    return test_epoch_loss, test_epoch_acc




# 训练
epochs = 50
train_loss = []
train_acc  = []
test_loss = []
test_acc  = []

for epoch in range(epochs):
    # epoch_loss, epoch_acc, test_epoch_loss, test_epoch_acc = train_one_epoch(epoch, model_ft, image_dataloader['train'], image_dataloader['valid'])
    epoch_loss, epoch_acc = train(epoch,net,image_dataloader['train'])
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)

    # cosine_schedule.step()
    scheduler.step()

    test_epoch_loss, test_epoch_acc = test(epoch,net,image_dataloader['valid'])
    test_loss.append(test_epoch_loss)
    test_acc.append(test_epoch_acc)


