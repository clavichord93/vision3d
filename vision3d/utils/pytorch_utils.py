import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def reset_numpy_random_seed(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    # print(worker_id, seed)
    np.random.seed(seed)


class CosineAnnealingFunction(object):
    def __init__(self, max_epoch, eta_min=0):
        self.max_epoch = max_epoch
        self.eta_min = eta_min

    def __call__(self, last_epoch):
        return self.eta_min + (1 - self.eta_min) * (1 + math.cos(math.pi * last_epoch / self.max_epoch)) / 2


class SmoothCrossEntropyLoss(nn.Module):
    def __init__(self, eps=0.1):
        super(SmoothCrossEntropyLoss, self).__init__()
        self.eps = eps

    def forward(self, preds, labels):
        device = preds.device
        one_hot = torch.zeros_like(preds).to(device).scatter(1, labels.unsqueeze(1), 1)
        labels = one_hot * (1 - self.eps) + self.eps / preds.shape[1]
        log_probs = F.log_softmax(preds, dim=1)
        loss = -(labels * log_probs).sum(dim=1).mean()
        return loss


def _get_activation_fn(activation_fn, **kwargs):
    if activation_fn == 'relu':
        return nn.ReLU(inplace=True)
    elif activation_fn == 'lrelu':
        if 'negative_slope' in kwargs:
            negative_slope = kwargs['negative_slope']
        else:
            negative_slope = 0.01
        return nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
    elif activation_fn == 'elu':
        return nn.ELU(inplace=True)
    elif activation_fn == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise ValueError('Activation function {} is not supported'.format(activation_fn))


class ConvBlock1d(nn.Sequential):
    def __init__(self,
                 input_dim,
                 output_dim,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 batch_norm=True,
                 batch_norm_after_activation=False,
                 activation='relu',
                 dropout=None,
                 **kwargs):
        super(ConvBlock1d, self).__init__()
        bias = not batch_norm
        layers = []
        layers.append(('conv', nn.Conv1d(input_dim,
                                         output_dim,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=groups,
                                         bias=bias)))
        if batch_norm:
            layers.append(('bn', nn.BatchNorm1d(output_dim)))
        if activation is not None:
            layers.append((activation, _get_activation_fn(activation, **kwargs)))
        if batch_norm and activation is not None and batch_norm_after_activation:
            layers[-2], layers[-1] = layers[-1], layers[-2]
        if dropout is not None:
            layers.append(('dp', nn.Dropout(dropout)))
        for name, module in layers:
            self.add_module(name, module)
    
    def forward(self, inputs):
        return super(ConvBlock1d, self).forward(inputs)


class ConvBlock2d(nn.Sequential):
    def __init__(self,
                 input_dim,
                 output_dim,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 batch_norm=True,
                 batch_norm_after_activation=False,
                 activation='relu',
                 dropout=None,
                 **kwargs):
        super(ConvBlock2d, self).__init__()
        bias = not batch_norm
        layers = []
        layers.append(('conv', nn.Conv2d(input_dim,
                                         output_dim,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=groups,
                                         bias=bias)))
        if batch_norm:
            layers.append(('bn', nn.BatchNorm2d(output_dim)))
        if activation is not None:
            layers.append((activation, _get_activation_fn(activation, **kwargs)))
        if batch_norm and activation is not None and batch_norm_after_activation:
            layers[-2], layers[-1] = layers[-1], layers[-2]
        if dropout is not None:
            layers.append(('dp', nn.Dropout(dropout)))
        for name, module in layers:
            self.add_module(name, module)

    def forward(self, inputs):
        return super(ConvBlock2d, self).forward(inputs)


class LinearBlock(nn.Sequential):
    def __init__(self,
                 input_dim,
                 output_dim,
                 batch_norm=True,
                 batch_norm_after_activation=False,
                 activation='relu',
                 dropout=None,
                 **kwargs):
        super(LinearBlock, self).__init__()
        bias = not batch_norm
        layers = []
        layers.append(('fc', nn.Linear(input_dim, output_dim, bias=bias)))
        if batch_norm:
            layers.append(('bn', nn.BatchNorm1d(output_dim)))
        if activation is not None:
            layers.append((activation, _get_activation_fn(activation, **kwargs)))
        if batch_norm and activation is not None and batch_norm_after_activation:
            layers[-2], layers[-1] = layers[-1], layers[-2]
        if dropout is not None:
            layers.append(('dp', nn.Dropout(dropout)))
        for name, module in layers:
            self.add_module(name, module)

    def forward(self, inputs):
        return super(LinearBlock, self).forward(inputs)


def create_conv1d_blocks(input_dim,
                         output_dims,
                         kernel_size,
                         stride=1,
                         padding=0,
                         dilation=1,
                         groups=1,
                         batch_norm=True,
                         batch_norm_after_activation=False,
                         activation='relu',
                         dropout=None,
                         start_index=1,
                         **kwargs):
    r"""
    Create a list of ConvBlock1d. The name of the i-th ConvBlock1d is `conv{i+start_index}`.

    :param input_dim: int
        The number of the input channels.
    :param output_dims: list of int or int
        If `output_dims` is a list of int, it represents the numbers of the output channels in each ConvBlock1d.
        If `output_dims` is a int, it means there is only one ConvBlock1d.
    :param kernel_size: int
        The kernel size in convolution.
    :param stride: int
        The stride in convolution.
    :param padding: int
        The padding in convolution.
    :param dilation: int
        The dilation in convolution.
    :param groups: int
        The groups in convolution.
    :param batch_norm: bool
        Whether batch normalization is used or not.
    :param batch_norm_after_activation: bool
        If True, every ConvBlock1d is in the order of [Conv1d, Activation_Fn, BatchNorm1d].
        If False, every ConvBlock1d is in the order of [Conv1d, BatchNorm1d, Activation_Fn].
    :param activation: str
        The name of the activation function in each ConvBlock1d.
    :param dropout: None or float
        If None, no dropout is used.
        If a float, the dropout probability in each ConvBlock1d. The Dropout is used at the end of each ConvBlock1d.
    :param start_index: int
        The index used in the name of the first ConvBlock1d.
    """
    if isinstance(output_dims, int):
        output_dims = [output_dims]
    layers = []
    for i, output_dim in enumerate(output_dims):
        layers.append(('conv{}'.format(start_index + i),
                       ConvBlock1d(input_dim,
                                   output_dim,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   dilation=dilation,
                                   groups=groups,
                                   batch_norm=batch_norm,
                                   batch_norm_after_activation=batch_norm_after_activation,
                                   activation=activation,
                                   dropout=dropout,
                                   **kwargs)))
        input_dim = output_dim
    return layers


def create_conv2d_blocks(input_dim,
                         output_dims,
                         kernel_size,
                         stride=1,
                         padding=0,
                         dilation=1,
                         groups=1,
                         batch_norm=True,
                         batch_norm_after_activation=False,
                         activation='relu',
                         dropout=None,
                         start_index=1,
                         **kwargs):
    r"""
    Create a list of ConvBlock2d. The name of the i-th ConvBlock2d is `conv{i+start_index}`.

    :param input_dim: int
        The number of the input channels.
    :param output_dims: list of int or int
        If `output_dims` is a list of int, it represents the numbers of the output channels in each ConvBlock2d.
        If `output_dims` is a int, it means there is only one ConvBlock2d.
    :param kernel_size: int
        The kernel size in convolution.
    :param stride: int
        The stride in convolution.
    :param padding: int
        The padding in convolution.
    :param dilation: int
        The dilation in convolution.
    :param groups: int
        The groups in convolution.
    :param batch_norm: bool
        Whether batch normalization is used or not.
    :param batch_norm_after_activation: bool
        If True, every ConvBlock2d is in the order of [Conv2d, Activation_Fn, BatchNorm2d].
        If False, every ConvBlock2d is in the order of [Conv2d, BatchNorm2d, Activation_Fn].
    :param activation: str
        The name of the activation function in each ConvBlock2d.
    :param dropout: None or float
        If None, no dropout is used.
        If a float, the dropout probability in each ConvBlock2d. The Dropout is used at the end of each ConvBlock2d.
    :param start_index: int
        The index used in the name of the first ConvBlock2d.
    """
    if isinstance(output_dims, int):
        output_dims = [output_dims]
    layers = []
    for i, output_dim in enumerate(output_dims):
        layers.append(('conv{}'.format(start_index + i),
                       ConvBlock2d(input_dim,
                                   output_dim,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   dilation=dilation,
                                   groups=groups,
                                   batch_norm=batch_norm,
                                   batch_norm_after_activation=batch_norm_after_activation,
                                   activation=activation,
                                   dropout=dropout,
                                   **kwargs)))
        input_dim = output_dim
    return layers


def create_linear_blocks(input_dim,
                         output_dims,
                         batch_norm=True,
                         batch_norm_after_activation=False,
                         activation='relu',
                         dropout=None,
                         start_index=1,
                         **kwargs):
    r"""
    Create a list of LinearBlock. The name of the i-th LinearBlock is `conv{i+start_index}`.

    :param input_dim: int
        The number of the input channels.
    :param output_dims: list of int or int
        If `output_dims` is a list of int, it represents the numbers of the output channels in each LinearBlock.
        If `output_dims` is a int, it means there is only one LinearBlock.
    :param batch_norm: bool
        Whether batch normalization is used or not.
    :param batch_norm_after_activation: bool
        If True, every LinearBlock is in the order of [Conv2d, Activation_Fn, BatchNorm2d].
        If False, every LinearBlock is in the order of [Conv2d, BatchNorm2d, Activation_Fn].
    :param activation: str
        The name of the activation function in each LinearBlock.
    :param dropout: None or float
        If None, no dropout is used.
        If a float, the dropout probability in each LinearBlock. The Dropout is used at the end of each LinearBlock.
    :param start_index: int
        The index used in the name of the first LinearBlock.
    """
    if isinstance(output_dims, int):
        output_dims = [output_dims]
    layers = []
    for i, output_dim in enumerate(output_dims):
        layers.append(('fc{}'.format(start_index + i),
                       LinearBlock(input_dim,
                                   output_dim,
                                   batch_norm=batch_norm,
                                   batch_norm_after_activation=batch_norm_after_activation,
                                   activation=activation,
                                   dropout=dropout,
                                   **kwargs)))
        input_dim = output_dim
    return layers


class DepthwiseConv1d(nn.Sequential):
    def __init__(self,
                 input_dim,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 depth_multiplier=1,
                 batch_norm=True,
                 batch_norm_after_activation=False,
                 activation='relu',
                 dropout=None,
                 **kwargs):
        super(DepthwiseConv1d, self).__init__()

        if not isinstance(depth_multiplier, int) or depth_multiplier <= 0:
            raise ValueError('`depth_multiplier ({}) must be a positive integer.'.format(depth_multiplier))

        bias = not batch_norm
        output_dim = input_dim * depth_multiplier

        layers = []
        layers.append(('conv', nn.Conv1d(input_dim,
                                         output_dim,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=input_dim,
                                         bias=bias)))
        if batch_norm:
            layers.append(('bn', nn.BatchNorm1d(output_dim)))
        if activation is not None:
            layers.append((activation, _get_activation_fn(activation, **kwargs)))
        if batch_norm and activation is not None and batch_norm_after_activation:
            layers[-2], layers[-1] = layers[-1], layers[-2]
        if dropout is not None:
            layers.append(('dp', nn.Dropout(dropout)))
        for name, module in layers:
            self.add_module(name, module)

    def forward(self, inputs):
        return super(DepthwiseConv1d, self).forward(inputs)


class DepthwiseConv2d(nn.Sequential):
    def __init__(self,
                 input_dim,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 depth_multiplier=1,
                 batch_norm=True,
                 batch_norm_after_activation=False,
                 activation='relu',
                 dropout=None,
                 **kwargs):
        super(DepthwiseConv2d, self).__init__()

        if not isinstance(depth_multiplier, int) or depth_multiplier <= 0:
            raise ValueError('`depth_multiplier ({}) must be a positive integer.'.format(depth_multiplier))

        bias = not batch_norm
        output_dim = input_dim * depth_multiplier

        layers = []
        layers.append(('conv', nn.Conv2d(input_dim,
                                         output_dim,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=input_dim,
                                         bias=bias)))
        if batch_norm:
            layers.append(('bn', nn.BatchNorm2d(output_dim)))
        if activation is not None:
            layers.append((activation, _get_activation_fn(activation, **kwargs)))
        if batch_norm and activation is not None and batch_norm_after_activation:
            layers[-2], layers[-1] = layers[-1], layers[-2]
        if dropout is not None:
            layers.append(('dp', nn.Dropout(dropout)))
        for name, module in layers:
            self.add_module(name, module)

    def forward(self, inputs):
        return super(DepthwiseConv2d, self).forward(inputs)


class SeparableConv1d(nn.Sequential):
    def __init__(self,
                 input_dim,
                 output_dim,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 depth_multiplier=1,
                 batch_norm=True,
                 batch_norm_after_activation=False,
                 activation='relu',
                 dropout=None,
                 **kwargs):
        super(SeparableConv1d, self).__init__()

        if not isinstance(depth_multiplier, int) or depth_multiplier <= 0:
            raise ValueError('`depth_multiplier ({}) must be a positive integer.'.format(depth_multiplier))

        bias = not batch_norm
        hidden_dim = input_dim * depth_multiplier

        layers = []
        layers.append(('dwconv', nn.Conv1d(input_dim,
                                           hidden_dim,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           dilation=dilation,
                                           groups=input_dim)))
        layers.append(('pwconv', nn.Conv1d(hidden_dim, output_dim, kernel_size=1, bias=bias)))
        if batch_norm:
            layers.append(('bn', nn.BatchNorm1d(output_dim)))
        if activation is not None:
            layers.append((activation, _get_activation_fn(activation, **kwargs)))
        if batch_norm and activation is not None and batch_norm_after_activation:
            layers[-2], layers[-1] = layers[-1], layers[-2]
        if dropout is not None:
            layers.append(('dp', nn.Dropout(dropout)))
        for name, module in layers:
            self.add_module(name, module)

    def forward(self, inputs):
        return super(SeparableConv1d, self).forward(inputs)


class SeparableConv2d(nn.Sequential):
    def __init__(self,
                 input_dim,
                 output_dim,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 depth_multiplier=1,
                 batch_norm=True,
                 batch_norm_after_activation=False,
                 activation='relu',
                 dropout=None,
                 **kwargs):
        super(SeparableConv2d, self).__init__()

        if not isinstance(depth_multiplier, int) or depth_multiplier <= 0:
            raise ValueError('`depth_multiplier ({}) must be a positive integer.'.format(depth_multiplier))

        bias = not batch_norm
        hidden_dim = input_dim * depth_multiplier

        layers = []
        layers.append(('dwconv', nn.Conv2d(input_dim,
                                           hidden_dim,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           dilation=dilation,
                                           groups=input_dim)))
        layers.append(('pwconv', nn.Conv2d(hidden_dim, output_dim, kernel_size=1, bias=bias)))
        if batch_norm:
            layers.append(('bn', nn.BatchNorm2d(output_dim)))
        if activation is not None:
            layers.append((activation, _get_activation_fn(activation, **kwargs)))
        if batch_norm and activation is not None and batch_norm_after_activation:
            layers[-2], layers[-1] = layers[-1], layers[-2]
        if dropout is not None:
            layers.append(('dp', nn.Dropout(dropout)))
        for name, module in layers:
            self.add_module(name, module)

    def forward(self, inputs):
        return super(SeparableConv2d, self).forward(inputs)
