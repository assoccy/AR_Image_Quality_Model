"""
Adapted from:
    https://github.com/MIPT-Oulu/pytorch_bn_fusion/blob/master/bn_fusion.py
    https://github.com/nathanhubens/fasterai/blob/master/fasterai/bn_folder.py

"""

import torch
import torch.nn as nn
import copy


class BatchNormFolder():
    def __init__(self):
        super().__init__()

    def fold(self, model):
        """

        :param model:
        :return:
        """
        bn_merged_model = copy.deepcopy(model)

        module_names = list(bn_merged_model._modules)

        for k, name in enumerate(module_names):

            if len(list(bn_merged_model._modules[name]._modules)) > 0:
                bn_merged_model._modules[name] = self.fold(bn_merged_model._modules[name])

            else:
                if isinstance(bn_merged_model._modules[name], nn.BatchNorm2d):
                    if isinstance(bn_merged_model._modules[module_names[k - 1]], nn.Conv2d):
                        # Folded BN
                        folded_conv = self.fold_conv_bn(bn_merged_model._modules[module_names[k - 1]],
                                                        bn_merged_model._modules[name])
                        # Replace old weight values
                        # Remove the BN layer
                        bn_merged_model._modules.pop(name)
                        # Replace the Convolutional Layer by the folded version
                        bn_merged_model._modules[module_names[k - 1]] = folded_conv

        return bn_merged_model

    def fold_conv_bn(self, conv, bn):
        """

        :param conv:
        :param bn:
        :return:
        """
        assert (not (conv.training or bn.training)), "Fusion only for evaluation!"
        fused_conv = copy.deepcopy(conv)

        bn_st_dict = bn.state_dict()
        conv_st_dict = conv.state_dict()

        # BatchNorm params
        eps = bn.eps
        mu = bn_st_dict['running_mean']
        var = bn_st_dict['running_var']
        gamma = bn_st_dict['weight']

        if 'bias' in bn_st_dict:
            beta = bn_st_dict['bias']
        else:
            beta = torch.zeros(gamma.size(0)).float().to(gamma.device)

        # Conv params
        W = conv_st_dict['weight']
        if 'bias' in conv_st_dict:
            bias = conv_st_dict['bias']
        else:
            bias = torch.zeros(W.size(0)).float().to(gamma.device)

        denom = torch.sqrt(var + eps)
        b = beta - gamma.mul(mu).div(denom)
        A = gamma.div(denom)
        bias *= A
        A = A.expand_as(W.transpose(0, -1)).transpose(0, -1)

        W.mul_(A)
        bias.add_(b)

        fused_conv.weight.data.copy_(W)
        if fused_conv.bias is None:
            fused_conv.bias = torch.nn.Parameter(bias)
        else:
            fused_conv.bias.data.copy_(bias)

        return fused_conv