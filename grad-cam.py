import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import torchvision.models as models
from torchvision.transforms import Compose, Normalize, ToTensor
from MyNet3 import DenseNet13
from DenseNet import DenseNet121
# from MyNet2 import DenseNet13
# from MyNet2 import DenseNet13
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


# 变暗
def Darker(image,percetage=0.9):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    #get darker
    for xi in range(0,w):
        for xj in range(0,h):
            image_copy[xj,xi,0] = int(image[xj,xi,0]*percetage)
            image_copy[xj,xi,1] = int(image[xj,xi,1]*percetage)
            image_copy[xj,xi,2] = int(image[xj,xi,2]*percetage)
    return image_copy

###预处理：剪裁512*512，灰度拉伸，自适应直方图均衡化，自适应阈值法进行二值化
class GradCAM():
    '''
    Grad-cam: Visual explanations from deep networks via gradient-based localization
    Selvaraju R R, Cogswell M, Das A, et al.
    https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html
    '''

    def __init__(self, model, target_layers, use_cuda=True):
        super(GradCAM).__init__()
        self.use_cuda = use_cuda
        self.model = model
        self.target_layers = target_layers

        self.target_layers.register_forward_hook(self.forward_hook)
        self.target_layers.register_full_backward_hook(self.backward_hook)

        self.activations = []
        self.grads = []

    def forward_hook(self, module, input, output):
        self.activations.append(output[0])

    def backward_hook(self, module, grad_input, grad_output):
        self.grads.append(grad_output[0].detach())

    def calculate_cam(self, model_input):
        if self.use_cuda:
            device = torch.device('cuda')
            self.model.to(device)  # Module.to() is in-place method
            model_input = model_input.to(device)  # Tensor.to() is not a in-place method
        self.model.eval()

        # forward
        y_hat = self.model(model_input)
        max_class = np.argmax(y_hat.cpu().data.numpy(), axis=1)
        max_class = [0]

        # backward
        model.zero_grad()
        y_c = y_hat[0, max_class]
        y_c.backward()

        # get activations and gradients
        activations = self.activations[0].cpu().data.numpy().squeeze()
        grads = self.grads[0].cpu().data.numpy().squeeze()

        # calculate weights
        weights = np.mean(grads.reshape(grads.shape[0], -1), axis=1)
        weights = weights.reshape(-1, 1, 1)
        cam = (weights * activations).sum(axis=0)
        cam = np.maximum(cam, 0)  # ReLU
        cam = cam / cam.max()
        return cam

    @staticmethod
    def show_cam_on_image(image, cam):
        # image: [H,W,C]
        h, w = image.shape[:2]


        cam = cv2.resize(cam, (h, w))
        cam = cam / cam.max()
        heatmap = cv2.applyColorMap((255 * cam).astype(np.uint8), cv2.COLORMAP_JET)  # [H,W,C]
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
        image = image / image.max()
        heatmap = heatmap / heatmap.max()

        result = 0.7 * heatmap + 0.3 * image

        result = result / result.max()

        plt.figure()
        plt.imshow((result * 255).astype(np.uint8))
        plt.colorbar(shrink=0.8)
        plt.tight_layout()
        plt.xticks([])  # 去掉x轴
        plt.yticks([])  # 去掉y轴
        plt.axis('off')  # 去掉坐标轴
        plt.show()

    @staticmethod
    def preprocess_image(img):
        preprocessing = Compose([
            ToTensor(),
        ])
        return preprocessing(img.copy()).unsqueeze(0)


if __name__ == '__main__':
    # image = cv2.imread(r'C:\data\small_data\woman\2__F_Left_thumb_finger.BMP',0)  # (224,224,3)
    image = cv2.imread(r'C:\data\grad-cam\socofing_243_M\243__M_Right_thumb_finger.BMP',0)
    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = cv2.resize(image,(224,224))
    rows, cols = image.shape[:2]
    image = imgHPFilter(image, D0=50)
    image = np.clip(image, 0, 1)



    input_tensor = GradCAM.preprocess_image(image)
    # 模型保存
    # model = DenseNet13(2)
    model = DenseNet121(2)
    # filename = r'C:\data\SOCOmodel\MyNet_se_gender_8.pth'
    filename = r'C:\data\SOCOmodel\DenseNet_224_2.pth'
    checkpoint = torch.load(filename)
    best_acc = checkpoint['best_acc']
    model.load_state_dict(checkpoint['state_dict'])
    # model = models.resnet18(pretrained=True)
    grad_cam = GradCAM(model, model.transtion3.transition_layer[2], 224)
    cam = grad_cam.calculate_cam(input_tensor)

    ###增加两个通道
    image = np.expand_dims(image, axis=2)
    image = np.concatenate((image, image, image), axis=-1)

    GradCAM.show_cam_on_image(image, cam)
