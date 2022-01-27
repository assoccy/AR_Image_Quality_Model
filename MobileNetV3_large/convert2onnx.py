#!/usr/bin/python3
# coding=utf-8
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch import Tensor
from PIL import Image
from mobilenetv3 import mobilenet_v3_large


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


if __name__ == '__main__':
    model_path = 'MobileNetV3_best.pth'
    device = 'cuda:0'
    # device = 'cpu'

    model = load_model(model_path=model_path, device=device)

    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    input_names = ['input_imgs']
    output_names = ['output']

    torch.onnx.export(model, dummy_input, 'mobilenetv3.onnx', verbose=True, input_names=input_names,
                      output_names=output_names)


