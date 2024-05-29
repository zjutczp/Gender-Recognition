import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
# from torchsummary import summary


class Resdual_Block(nn.Module):              #改进的残差块
    def __init__(self, i_channel, o_channel, stride=1, downsample=None):
        super(Resdual_Block, self).__init__()
        self.conv1=nn.Conv2d(in_channels=i_channel, out_channels=o_channel, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.conv2 = nn.Conv2d(in_channels=i_channel//4, out_channels=o_channel//4, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.conv3 = nn.Conv2d(in_channels=i_channel // 4, out_channels=o_channel // 4, kernel_size=3, stride=1,
                               padding=1,
                               bias=False)
        self.conv4 = nn.Conv2d(in_channels=i_channel // 4, out_channels=o_channel // 4, kernel_size=3, stride=1,
                               padding=1,
                               bias=False)
        self.conv6 = nn.Conv2d(in_channels=i_channel // 4, out_channels=o_channel // 4, kernel_size=3, stride=1,
                               padding=1,
                               bias=False)
        self.conv5 = nn.Conv2d(in_channels=i_channel, out_channels=o_channel, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1=nn.BatchNorm2d(o_channel)
        self.bn2=nn.BatchNorm2d(o_channel // 4)
        self.bn3 = nn.BatchNorm2d(o_channel // 4)
        self.bn4 = nn.BatchNorm2d(o_channel // 4)
        self.bn6 = nn.BatchNorm2d(o_channel // 4)
        self.bn5 = nn.BatchNorm2d(o_channel)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        out=self.relu(self.bn1(self.conv1(x)))
        out1,out2,out3,out4=out.chunk(4,1)  #4是多少块，1是哪个维度                      #列分块，分成4块
        out1=self.relu(self.bn6(self.conv6(out1)))
        out2=out1+out2
        out2=self.relu(self.bn2(self.conv2(out2)))
        out3=out2+out3
        out3=self.relu(self.bn3(self.conv3(out3)))
        out4=out3+out4
        out4=self.relu(self.bn4(self.conv4(out4)))
        outx=torch.cat([out1,out2,out3,out4],1)   #1是维数，不知道是几目前
        out=self.relu(self.bn5(self.conv5(outx)))

        return out

##############################################残差块写好了#############################################


class Residual_Net(nn.Module):
    def __init__(self, stride=1, downsample=None):
        super(Residual_Net, self).__init__()
        self.conv1=nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3,
                      bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.relu1=nn.ReLU(inplace=True)
        self.pooling=nn.AvgPool2d(3,2,1)
        self.res1=Resdual_Block(64,64)
        self.conv2=nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1,
                      bias=False)
        self.bn2=nn.BatchNorm2d(128)
        self.res2=Resdual_Block(64,64)
        self.conv3=nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1,
                      bias=False)
        self.bn3=nn.BatchNorm2d(256)
        self.res3=Resdual_Block(128,128)
        self.res4=Resdual_Block(128,128)
        self.conv4=nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1,
                      bias=False)
        self.bn4=nn.BatchNorm2d(512)
        self.res5=Resdual_Block(256,256)
        self.res6 = Resdual_Block(256,256)
        self.res7 = Resdual_Block(512,512)
        self.res8 = Resdual_Block(512, 512)
        self.res9 = Resdual_Block(256, 256)
        self.res10 = Resdual_Block(512, 512)
        self.res11 = Resdual_Block(256, 256)
        self.res12 = Resdual_Block(512, 512)
        self.conv5=nn.Conv2d(in_channels=512, out_channels=2, kernel_size=1, stride=1, padding=0,
                      bias=False)
        self.fc=nn.Linear(200,2)
        self.softmax=nn.Softmax(1)

        #self.drop=nn.Dropout(0.2)


    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out1=self.pooling(out)
        out2=self.res1(out1)
        out=out1+out2
        out2=self.res2(out)
        out = out + out2
        out=self.pooling(out)

        out1 = self.relu1(self.bn2(self.conv2(out)))
        out2=self.res3(out1)
        out=out1+out2
        out2=self.res4(out)
        out = out + out2
        out=self.pooling(out)

        out=self.relu1(self.bn3(self.conv3(out)))
        out2=self.res5(out)
        out=out+out2
        out2=self.res6(out)
        out=out+out2
        # out2=self.res9(out)
        # out=out+out2
        # out2 = self.res11(out)
        # out = out + out2
        out=self.pooling(out)

        out=self.relu1(self.bn4(self.conv4(out)))
        out2=self.res7(out)
        out=out+out2
        out2=self.res8(out)
        out=out+out2
        # out2 = self.res10(out)
        # out = out + out2
        # out2 = self.res12(out)
        # out = out + out2
        out=self.conv5(out)

        # out=out.view(out.size(0), -1)
        # out=self.fc(out)
        # out=self.softmax(out)

        out=out.mean(2)                                     #？
        out=out.mean(2)
        # out=torch.split(out,1,dim=1)[1]
        # xxx=list(xxx)
        # out=torch.LongTensor(xxx)

        return out

# if __name__ == '__main__':
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     net = Residual_Net().cuda()
#     summary(net,(1,300,300))