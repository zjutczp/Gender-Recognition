
# Here is the code ：

import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from MyNet2 import DenseNet13
import cv2

def main():
    # 模型保存
    model = DenseNet13(2)
    filename = r'C:\data\SOCOmodel\MyNet_se_gender_7.pth'
    checkpoint = torch.load(filename)
    best_acc = checkpoint['best_acc']
    model.load_state_dict(checkpoint['state_dict'])
    # model = models.resnet50(pretrained=True)
    target_layers = [model.transtion3.transition_layer[2]]

    data_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Prepare image
    img_path = r'C:\data\small_data\man\1__M_Right_middle_finger.BMP'
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    # img = Image.open(img_path).convert('L')
    img = cv2.imread(img_path, 0)  # (224,224,3)
    img = cv2.resize(img, (224, 224))
    img = np.array(img, dtype=np.uint8)
    img_tensor = data_transform(img)
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    # Grad CAM
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    # targets = [ClassifierOutputTarget(281)]     # cat
    targets = [ClassifierOutputTarget(0)]  # dog

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32)/255.,
                                      grayscale_cam, use_rgb=True)

    plt.imshow(visualization)
    plt.show()


if __name__ == '__main__':
    main()
