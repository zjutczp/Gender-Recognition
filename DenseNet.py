import torch.nn as nn
import torch
from torchvision import models
from thop import profile

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate=0):
        super(_DenseLayer, self).__init__()
        self.drop_rate = drop_rate
        self.dense_layer = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_input_features, out_channels=bn_size * growth_rate, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(bn_size * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=bn_size * growth_rate, out_channels=growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.dropout = nn.Dropout(p=self.drop_rate)

    def forward(self, x):
        y = self.dense_layer(x)
        if self.drop_rate > 0:
            y = self.dropout(y)

        return torch.cat([x, y], dim=1)


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate=0):
        super(_DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(_DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class _TransitionLayer(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(_TransitionLayer, self).__init__()
        self.transition_layer = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_input_features, out_channels=num_output_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.transition_layer(x)


class DenseNet(nn.Module):
    def __init__(self, num_init_features=64, growth_rate=32, blocks=(6, 12, 24, 16), bn_size=4, drop_rate=0, num_classes=2):
        super(DenseNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        num_features = num_init_features
        self.layer1 = _DenseBlock(num_layers=blocks[0], num_input_features=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features = num_features + blocks[0] * growth_rate
        self.transtion1 = _TransitionLayer(num_input_features=num_features, num_output_features=num_features // 2)

        num_features = num_features // 2
        self.layer2 = _DenseBlock(num_layers=blocks[1], num_input_features=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features = num_features + blocks[1] * growth_rate
        self.transtion2 = _TransitionLayer(num_input_features=num_features, num_output_features=num_features // 2)

        num_features = num_features // 2
        self.layer3 = _DenseBlock(num_layers=blocks[2], num_input_features=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features = num_features + blocks[2] * growth_rate
        self.transtion3 = _TransitionLayer(num_input_features=num_features, num_output_features=num_features // 2)

        num_features = num_features // 2
        self.layer4 = _DenseBlock(num_layers=blocks[3], num_input_features=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features = num_features + blocks[3] * growth_rate

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_features, num_classes)
        # self.softmax = nn.Softmax(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)

        x = self.layer1(x)
        x = self.transtion1(x)
        x = self.layer2(x)
        x = self.transtion2(x)
        x = self.layer3(x)
        x = self.transtion3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        # x = self.softmax(x)

        return x

def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('模型权重大小为：{:.3f}MB'.format(param_size/1024/1024))
    print('模型总大小为：{:.3f}MB'.format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)

def DenseNet13(num_classes):
    return DenseNet(blocks=(6, 6, 6, 6), num_classes=num_classes)

def DenseNet121(num_classes):
    return DenseNet(blocks=(6, 12, 24, 16), num_classes=num_classes)

def DenseNet169(num_classes):
    return DenseNet(blocks=(6, 12, 32, 32), num_classes=num_classes)

def DenseNet201(num_classes):
    return DenseNet(blocks=(6, 12, 48, 32), num_classes=num_classes)

def DenseNet264(num_classes):
    return DenseNet(blocks=(6, 12, 64, 48), num_classes=num_classes)

# def read_densenet121():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = models.densenet121(pretrained=True)
#     model.to(device)
#     print(model)
#
#
# def get_densenet121(flag, num_classes):
#     if flag:
#         net = models.densenet121(pretrained=True)
#         num_input = net.classifier.in_features
#         net.classifier = nn.Linear(num_input, num_classes)
#     else:
#         net = DenseNet121(num_classes)
#
# #     return net
# getModelSize(DenseNet121(2))
# net = DenseNet121(2)
# # print(net)
# # print(net)
# x = torch.randn(1, 1, 224, 224)
# flops, params = profile(net, (x,))
# print('flops: ', flops, 'params: ', params)
# print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))

# y = net(x)
# a, pred = torch.max(y.data, 1)
# print(y)
# print(a)
# print(pred)

