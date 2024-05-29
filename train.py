import torch,os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from torch.autograd import Variable
from Parnet import parnet_s
from dataset_loader import MyDataset
from tensorboardX import SummaryWriter
from tqdm import tqdm
writer = SummaryWriter('logs')  ##创建一个SummaryWriter的实例,默认目录名字为runs

#设置超参数

EPOCH = 30
IMG_SIZE = 64
BATCH_SIZE= 64
# IMG_MEAN = [0.485, 0.456, 0.406]
# IMG_STD = [0.229, 0.224, 0.225]
CUDA=torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")

#数据预处理
train_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
])

val_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
])


root_train = r"C:\data\txt\train_list_not_random.txt"
root_test = r"C:\data\txt\test_list_not_random.txt"

# train_dataset = MyDataset(root_train,transform=transforms.ToTensor())
# test_dataset = MyDataset(root_test,transform=transforms.ToTensor())
train_dataset = MyDataset(root_train,transform=train_transforms)
test_dataset = MyDataset(root_test,transform=val_transforms)

image_dataset = {'train':train_dataset, 'valid':test_dataset}
image_dataloader = {x:DataLoader(image_dataset[x],batch_size=BATCH_SIZE,shuffle=True) for x in ['train', 'valid']}
dataset_sizes = {x:len(image_dataset[x]) for x in ['train', 'valid']}

model_ft = parnet_s(1, 2)
model_ft.to(DEVICE)

#设置损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_ft.parameters(), lr=1e-3)#指定 新加的fc层的学习率
cosine_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=20,eta_min=1e-9)

#设置训练函数
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    loop = tqdm(enumerate(train_loader),total=len(train_loader))
    sum_loss = 0
    total_accuracy = 0
    total_num = len(train_loader.dataset)
    # print(total_num, len(train_loader))
    for batch_idx, (data, target) in loop:
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        print_loss = loss.data.item()
        sum_loss += print_loss
        accuracy = torch.mean((torch.argmax(F.softmax(output, dim=-1), dim=-1) == target).type(torch.FloatTensor))
        total_accuracy += accuracy.item()

        loop.set_description(f'Epoch [{epoch}/{EPOCH}]')
        loop.set_postfix(loss=sum_loss/(batch_idx+1),acc=float(total_accuracy)/float(BATCH_SIZE*batch_idx+len(data)))
        print()

        if (batch_idx + 1) % 1000 == 0:
            ave_loss = sum_loss / (batch_idx + 1)
            acc = total_accuracy / (batch_idx + 1)
            writer.add_scalar('data/trainloss',ave_loss,epoch)
            writer.add_scalar('data/trainacc',acc,epoch)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR:{:.9f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.item(), lr))

            print('epoch:%d,loss:%.4f,train_acc:%.4f' % (epoch, ave_loss, acc))


ACC = 0


# 验证过程
def val(model, device, test_loader):
    global ACC
    model.eval()
    test_loss = 0
    correct = 0
    total_num = len(test_loader.dataset)
    # print(total_num, len(test_loader))
    with torch.no_grad():
        for data, target in test_loader:
            data, target = Variable(data).to(device), Variable(target).to(device)
            output = model(data)
            loss = criterion(output, target)
            _, pred = torch.max(output.data, 1)
            correct += torch.sum(pred == target)
            print_loss = loss.data.item()
            test_loss += print_loss
        correct = correct.data.item()
        acc = correct / total_num
        avgloss = test_loss / len(test_loader)
        writer.add_scalar('data/valloss', avgloss, epoch)
        writer.add_scalar('data/valacc', acc, epoch)
        print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:4f}%)\n'.format(
            avgloss, correct, len(test_loader.dataset), 100 * acc))
        if acc > ACC:
            modelfile = 'C:\data\model\\'+'model_' + 'epoch_' + str(epoch) + '_' + 'ACC-' + str(round(acc, 3)) + '.pth'
            # torch.save(model_ft, 'model_' + 'epoch_' + str(epoch) + '_' + 'ACC-' + str(round(acc, 3)) + '.pth')
            torch.save(model_ft.state_dict(),modelfile)
            ACC = acc
    writer.close()


# 训练

for epoch in range(1, EPOCH + 1):
    train(model_ft, DEVICE, image_dataloader['train'], optimizer, epoch)
    cosine_schedule.step()
    val(model_ft, DEVICE, image_dataloader['valid'])

