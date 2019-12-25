import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def reset_numpy_random_seed(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    # print(worker_id, seed)
    np.random.seed(seed)


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
                 has_bn=True,
                 has_relu=True,
                 leaky_slope=None,
                 dropout_ratio=None):
        super(ConvUnit1d, self).__init__()
        bias = not has_bn
        self.add_module('conv',
                        nn.Conv1d(input_dim,
                                  output_dim,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=padding,
                                  dilation=dilation,
                                  groups=groups,
                                  bias=bias))
        if has_bn:
            self.add_module('bn', nn.BatchNorm1d(output_dim))
        if has_relu:
            if leaky_slope is not None:
                self.add_module('lrelu', nn.LeakyReLU(negative_slope=leaky_slope, inplace=True))
            else:
                self.add_module('relu', nn.ReLU(inplace=True))
        if dropout_ratio is not None:
            self.add_module('dp', nn.Dropout(dropout_ratio))
    
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
                 has_bn=True,
                 has_relu=True,
                 leaky_slope=None,
                 dropout_ratio=None):
        super(ConvUnit2d, self).__init__()
        bias = not has_bn
        self.add_module('conv',
                        nn.Conv2d(input_dim,
                                  output_dim,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=padding,
                                  dilation=dilation,
                                  groups=groups,
                                  bias=bias))
        if has_bn:
            self.add_module('bn', nn.BatchNorm2d(output_dim))
        if has_relu:
            if leaky_slope is not None:
                self.add_module('lrelu', nn.LeakyReLU(negative_slope=leaky_slope, inplace=True))
            else:
                self.add_module('relu', nn.ReLU(inplace=True))
        if dropout_ratio is not None:
            self.add_module('dp', nn.Dropout(dropout_ratio))

    def forward(self, inputs):
        return super(ConvUnit2d, self).forward(inputs)


class FCUnit(nn.Sequential):
    def __init__(self, input_dim, output_dim, has_bn=True, has_relu=True, leaky_slope=None, dropout_ratio=None):
        super(FCUnit, self).__init__()
        bias = not has_bn
        self.add_module('fc', nn.Linear(input_dim, output_dim, bias=bias))
        if has_bn:
            self.add_module('bn', nn.BatchNorm1d(output_dim))
        if has_relu:
            if leaky_slope is not None:
                self.add_module('lrelu', nn.LeakyReLU(negative_slope=leaky_slope, inplace=True))
            else:
                self.add_module('relu', nn.ReLU(inplace=True))
        if dropout_ratio is not None:
            self.add_module('dp', nn.Dropout(dropout_ratio))

    def forward(self, inputs):
        return super(FCUnit, self).forward(inputs)


def create_conv1d_blocks(input_dim,
                         output_dims,
                         kernel_size,
                         stride=1,
                         padding=0,
                         dilation=1,
                         groups=1,
                         has_bn=True,
                         has_relu=True,
                         leaky_slope=None,
                         dropout_ratio=None):
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
                                  has_bn=has_bn,
                                  has_relu=has_relu,
                                  leaky_slope=leaky_slope,
                                  dropout_ratio=dropout_ratio)))
        input_dim = output_dim
    return layers


def create_conv2d_blocks(input_dim,
                         output_dims,
                         kernel_size,
                         stride=1,
                         padding=0,
                         dilation=1,
                         groups=1,
                         has_bn=True,
                         has_relu=True,
                         leaky_slope=None,
                         dropout_ratio=None):
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
                                  has_bn=has_bn,
                                  has_relu=has_relu,
                                  leaky_slope=leaky_slope,
                                  dropout_ratio=dropout_ratio)))
        input_dim = output_dim
    return layers


def create_fc_blocks(input_dim, output_dims, has_bn=True, has_relu=True, leaky_slope=None, dropout_ratio=None):
    if isinstance(output_dims, int):
        output_dims = [output_dims]
    layers = []
    for i, output_dim in enumerate(output_dims):
        index = str(i + 1)
        layers.append(('fc' + index,
                       FCUnit(input_dim,
                              output_dim,
                              has_bn=has_bn,
                              has_relu=has_relu,
                              leaky_slope=leaky_slope,
                              dropout_ratio=dropout_ratio)))
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
