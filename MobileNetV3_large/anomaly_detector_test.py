#!/usr/bin/python3
# coding=utf-8
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch import Tensor
from PIL import Image
from mobilenetv3 import mobilenet_v3_large
import cv2
import numpy as np


def load_model(model_path, device):
    model = mobilenet_v3_large()
    model.classifier = nn.Sequential(
        nn.Linear(960, 1280, bias=True),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(1280, 2, bias=True)
    )

    state_dict = torch.load(model_path)
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.to(device)

    return model


def val(model, img_list, device):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224), interpolation=torchvision.transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_set = []

    mean = [0.485, 0.456, 0.406]
    scale = [0.229, 0.224, 0.225]

    for img in img_list:
        input_image = cv2.imread(img)
        input_image = input_image / 225.
        input_image = cv2.resize(input_image, (224, 224), interpolation=cv2.INTER_NEAREST)
        input_image = input_image.astype(np.float32, copy=False)
        input_image -= np.array(mean)
        input_image *= np.array(scale)
        input_tensor = input_image.transpose(2, 0, 1)
        eval_set.append([input_tensor, img])

    evalloader = torch.utils.data.DataLoader(eval_set, batch_size=1, shuffle=False, num_workers=2)

    net = model
    net.eval()
    net.to(device)

    sm = nn.Softmax(dim=1)

    scores = []
    for data in evalloader:
        inputs, image = data
        inputs = inputs.to(device)
        outputs = net(inputs)
        # outputs_softmax = sm(outputs)
        outputs_softmax = outputs
        outputs_softmax = outputs_softmax.detach().cpu().numpy()
        scores.append(outputs_softmax)
    return scores


if __name__ == '__main__':
    model_path = 'MobileNetV3_best.pth'

    # device = 'cuda:0'
    device = 'cpu'

    model = load_model(model_path=model_path, device=device)

    test_data_list = ['test_images/test_1.png', 'test_images/test_2.png', 'test_images/test_3.png',
                      'test_images/test_4.png', 'test_images/test_5.png', 'test_images/test_6.png',
                      'test_images/test_7.png', 'test_images/test_8.png', 'test_images/test_9.png',
                      'test_images/test_10.png']

    scores = val(model=model, img_list=test_data_list, device=device)

    for score in scores:
        print(score[0])

    """
    [2.4059577e-09 1.0000000e+00]
    [7.9163481e-05 9.9992085e-01]
    [0.44992208 0.5500779 ]
    [1.3671563e-07 9.9999988e-01]
    [9.8134478e-07 9.9999905e-01]
    [3.1448834e-04 9.9968553e-01]
    [7.5576134e-04 9.9924421e-01]
    [3.5476997e-09 1.0000000e+00]
    [1.3016485e-05 9.9998701e-01]
    [5.6966709e-04 9.9943036e-01]
    """
