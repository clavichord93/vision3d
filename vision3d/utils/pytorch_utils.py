import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def reset_numpy_random_seed(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    # print(worker_id, seed)
    np.random.seed(seed)


def _get_activation_fn(activation_fn, negative_slope=None):
    if activation_fn == 'relu':
        return nn.ReLU(inplace=True)
    elif activation_fn == 'lrelu':
        if negative_slope is None:
            negative_slope = 0.01
        return nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
    elif activation_fn == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise ValueError('Activation function {} is not supported'.format(activation_fn))


class SmoothCrossEntropyLoss(nn.Module):
    def __init__(self, eps=0.2):
        super(SmoothCrossEntropyLoss, self).__init__()
        self.eps = eps

    def forward(self, *inputs):
        preds, labels = inputs
        device = preds.device
        one_hot = torch.zeros_like(preds).to(device)
        one_hot = one_hot.scatter(1, labels.unsqueeze(1), 1)
        num_class = preds.shape[1]
        weight = 1. - num_class / (num_class - 1) * self.eps
        bias = self.eps / (num_class - 1)
        labels = one_hot * weight + bias
        log_probs = F.log_softmax(preds, dim=1)
        loss = -(labels * log_probs).sum(dim=1).mean()
        return loss


class ConvUnit1d(nn.Sequential):
    def __init__(self,
                 input_dim,
                 output_dim,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 batch_norm=True,
                 activation='relu',
                 negative_slope=None,
                 dropout=None):
        super(ConvUnit1d, self).__init__()
        bias = not batch_norm
        self.add_module('conv',
                        nn.Conv1d(input_dim,
                                  output_dim,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=padding,
                                  dilation=dilation,
                                  groups=groups,
                                  bias=bias))
        if batch_norm:
            self.add_module('bn', nn.BatchNorm1d(output_dim))
        if activation is not None:
            activation_fn = _get_activation_fn(activation, negative_slope=negative_slope)
            self.add_module(activation, activation_fn)
        if dropout is not None:
            self.add_module('dp', nn.Dropout(dropout))
    
    def forward(self, inputs):
        return super(ConvUnit1d, self).forward(inputs)


class ConvUnit2d(nn.Sequential):
    def __init__(self,
                 input_dim,
                 output_dim,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 batch_norm=True,
                 activation='relu',
                 negative_slope=None,
                 dropout=None):
        super(ConvUnit2d, self).__init__()
        bias = not batch_norm
        self.add_module('conv',
                        nn.Conv2d(input_dim,
                                  output_dim,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=padding,
                                  dilation=dilation,
                                  groups=groups,
                                  bias=bias))
        if batch_norm:
            self.add_module('bn', nn.BatchNorm2d(output_dim))
        if activation is not None:
            activation_fn = _get_activation_fn(activation, negative_slope=negative_slope)
            self.add_module(activation, activation_fn)
        if dropout is not None:
            self.add_module('dp', nn.Dropout(dropout))

    def forward(self, inputs):
        return super(ConvUnit2d, self).forward(inputs)


class LinearUnit(nn.Sequential):
    def __init__(self,
                 input_dim,
                 output_dim,
                 batch_norm=True,
                 activation='relu',
                 negative_slope=None,
                 dropout=None):
        super(LinearUnit, self).__init__()
        bias = not batch_norm
        self.add_module('fc', nn.Linear(input_dim, output_dim, bias=bias))
        if batch_norm:
            self.add_module('bn', nn.BatchNorm1d(output_dim))
        if activation is not None:
            activation_fn = _get_activation_fn(activation, negative_slope=negative_slope)
            self.add_module(activation, activation_fn)
        if dropout is not None:
            self.add_module('dp', nn.Dropout(dropout))

    def forward(self, inputs):
        return super(LinearUnit, self).forward(inputs)


def create_conv1d_blocks(input_dim,
                         output_dims,
                         kernel_size,
                         stride=1,
                         padding=0,
                         dilation=1,
                         groups=1,
                         batch_norm=True,
                         activation='relu',
                         negative_slope=None,
                         dropout=None):
    if isinstance(output_dims, int):
        output_dims = [output_dims]
    layers = []
    for i, output_dim in enumerate(output_dims):
        index = str(i + 1)
        layers.append(('conv' + index,
                       ConvUnit1d(input_dim,
                                  output_dim,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=padding,
                                  dilation=dilation,
                                  groups=groups,
                                  batch_norm=batch_norm,
                                  activation=activation,
                                  negative_slope=negative_slope,
                                  dropout=dropout)))
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
                         activation='relu',
                         negative_slope=None,
                         dropout=None):
    if isinstance(output_dims, int):
        output_dims = [output_dims]
    layers = []
    for i, output_dim in enumerate(output_dims):
        index = str(i + 1)
        layers.append(('conv' + index,
                       ConvUnit2d(input_dim,
                                  output_dim,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=padding,
                                  dilation=dilation,
                                  groups=groups,
                                  batch_norm=batch_norm,
                                  activation=activation,
                                  negative_slope=negative_slope,
                                  dropout=dropout)))
        input_dim = output_dim
    return layers


def create_linear_blocks(input_dim,
                         output_dims,
                         batch_norm=True,
                         activation='relu',
                         negative_slope=None,
                         dropout=None):
    if isinstance(output_dims, int):
        output_dims = [output_dims]
    layers = []
    for i, output_dim in enumerate(output_dims):
        index = str(i + 1)
        layers.append(('fc' + index,
                       LinearUnit(input_dim,
                                  output_dim,
                                  batch_norm=batch_norm,
                                  activation=activation,
                                  negative_slope=negative_slope,
                                  dropout=dropout)))
        input_dim = output_dim
    return layers


def k_nearest_neighbors(points, centroids, num_neighbor):
    r"""
    Compute the kNNs of the points in `centroids` from the points in `points`.

    Note: This implementation decomposes uses less memory than the naive implementation:
    `pairwise_dist2 = torch.sum((centroids.unsqueeze(3) - points.unsqueeze(2)) ** 2, dim=1)`

    :param points: torch.Tensor (batch_size, num_feature, num_point)
        The features/coordinates of the points from which the kNNs are computed.
    :param centroids: torch.Tensor (batch_size, num_feature, num_centroid)
        The features/coordinates of the centroid points whose kNNs are computed.
    :param num_neighbor: int
        The number of nearest neighbors to compute.
    :return dist2: torch.Tensor(batch_size, num_points1, k)
        The squared distance of the kNNs of the centroids.
    :return index: torch.Tensor(batch_size, num_points1, k)
        The indices of the kNNs of the centroids.
    """
    a2 = torch.sum(centroids ** 2, dim=1).unsqueeze(2)
    ab = torch.matmul(centroids.transpose(1, 2), points)
    b2 = torch.sum(points ** 2, dim=1).unsqueeze(1)
    pairwise_dist2 = a2 - 2 * ab + b2
    dist2, index = pairwise_dist2.topk(num_neighbor, dim=2, largest=False)
    return dist2, index
