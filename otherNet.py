import torchvision.models
import torchvision.models as models
import torch.nn as nn
import torch

for i in dir(models):
    print(i)
# model = torchvision.models.resnet50(pretrained=True)
# fc_features = model.fc.in_features
# model.fc = nn.Linear(fc_features ,2)
# model.conv1 = nn.Conv2d(1 ,64, kernel_size=(7, 7),stride=(2,2),padding=(3,3),bias=False)
