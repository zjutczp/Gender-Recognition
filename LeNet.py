import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# 创建模型
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        # 定义模型
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=6,kernel_size=(5,5),stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels=6,out_channels=16,kernel_size=(5,5),stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=400, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=2)
        )

    def forward(self,x):
        # 定义前向算法
        x = self.features(x)
        # print(x.shape)
        x = torch.flatten(x,1)
        # print(x.shape)
        result = self.classifier(x)
        return result

# net = LeNet()
# print(net)