import os
import sys
import time
import torch
import logging
from torch import nn
import torch
import mobilenetv3

mlog = logging.getLogger('ConvertLogger')
mlog.setLevel(logging.INFO)
if len(mlog.handlers) > 0:
    mlog.handlers = list()

sys.path.append(os.path.join(os.path.dirname(__file__), './external'))
from torch2caffe.merge_batch_norm import BatchNormFolder


def torch2caffe(net, prototxt, caffemodel, name, vector=(1, 3, 224, 224)):
    """
    :param net:
    :param prototxt:
    :param caffemodel:
    :param name:
    :param vector:
    :return:
    """
    if isinstance(net, torch.nn.DataParallel):
        net = net.module
    device = torch.device("cpu")
    net.to(device)
    net.eval()

    input_ = torch.ones(vector)
    input_.to(device)

    # import engine here to avoid performance drop
    from utils.torch2caffe import engine
    engine.trans_net(net, input_, name)
    engine.save_prototxt(prototxt)
    engine.save_caffemodel(caffemodel)


def _help():
    print('*' * 40)
    print('Description and Usage Instructions:')
    print('Convert a model indicated by the timestamp.')
    print('$> python convert.py --model timestamp [optional flags] | --help')
    print(' * --help:            View usage instructions.')
    print(' * --model timestamp: Timestamp of the model to load.')
    print(' * --version [last | best metric_name | iter iter_number]: Which version of that model to load.')
    print('*' * 40)


def _make_divisible(ch, divisor=8, min_ch=None):
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


if __name__ == '__main__':
    model = mobilenetv3.mobilenet_v3_large(num_classes=2)
    model.classifier = nn.Sequential(
        nn.Linear(960, 1280, bias=True),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(1280, 2, bias=True)
    )
    state_dict = torch.load('MobileNetV3_best.pth')

    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)

    torch2caffe(
        model,
        os.path.join('MobileNetV3_large_caffe', 'MobileNetV3_large_caffe.prototxt'),
        os.path.join('MobileNetV3_large_caffe', 'MobileNetV3_large_caffe.caffemodel'),
        'MobileNetV3_large',
    )
