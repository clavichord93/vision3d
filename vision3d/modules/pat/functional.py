from collections import OrderedDict

import torch
import torch.nn as nn

from ..pointnet2 import functional as F
from ...utils.pytorch_utils import k_nearest_neighbors


__all__ = [
    'create_pat_conv1d_blocks',
    'create_pat_conv2d_blocks',
    'dilated_k_nearest_neighbors'
]


def create_pat_conv1d_blocks(input_dim, output_dims, groups):
    layers = []
    for i, output_dim in enumerate(output_dims):
        layers.append(('conv{}'.format(i + 1),
                       nn.Sequential(OrderedDict([
                           ('conv', nn.Conv1d(input_dim, output_dim, kernel_size=1, bias=False)),
                           ('gn', nn.GroupNorm(groups, output_dim)),
                           ('elu', nn.ELU(inplace=True))
                       ]))))
        input_dim = output_dim
    return layers


def create_pat_conv2d_blocks(input_dim, output_dims, groups):
    layers = []
    for i, output_dim in enumerate(output_dims):
        layers.append(('conv{}'.format(i + 1),
                       nn.Sequential(OrderedDict([
                           ('conv', nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False)),
                           ('gn', nn.GroupNorm(groups, output_dim)),
                           ('elu', nn.ELU(inplace=True))
                       ]))))
        input_dim = output_dim
    return layers


def dilated_k_nearest_neighbors(points, num_neighbor, dilation):
    # TODO: dilated kNN, it is currently kNN
    _, index = k_nearest_neighbors(points, points, num_neighbor + 1)
    # ignore the nearest point (itself)
    index = index[:, :, 1:]
    points = F.group_gather_by_index(points, index)
    return points


def farthest_point_sampling_and_gather(points, features, num_sample):
    index = F.farthest_point_sampling(points, num_sample)
    points = F.gather_by_index(points, index)
    if features is not None:
        features = F.gather_by_index(features, index)
    return points, features
