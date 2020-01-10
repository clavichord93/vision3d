from collections import OrderedDict

import torch
import torch.nn as nn

from ..pointnet2 import functional as F
from ...utils.pytorch_utils import k_nearest_neighbors


__all__ = [
    'create_pat_conv1d_blocks',
    'create_pat_conv2d_blocks',
    'create_pat_linear_blocks',
    'k_nearest_neighbors_graph',
    'farthest_sampling_and_gather'
]


def create_pat_conv1d_blocks(input_dim, output_dims, groups, dropout=None):
    layers = []
    for i, output_dim in enumerate(output_dims):
        block = [
            ('conv', nn.Conv1d(input_dim, output_dim, kernel_size=1, bias=False)),
            ('gn', nn.GroupNorm(groups, output_dim)),
            ('elu', nn.ELU(inplace=True))
        ]
        if dropout is not None:
            block.append(('dp', nn.Dropout(dropout)))
        layers.append(('conv{}'.format(i + 1), nn.Sequential(OrderedDict(block))))
        input_dim = output_dim
    return layers


def create_pat_conv2d_blocks(input_dim, output_dims, groups, dropout=None):
    layers = []
    for i, output_dim in enumerate(output_dims):
        block = [
            ('conv', nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False)),
            ('gn', nn.GroupNorm(groups, output_dim)),
            ('elu', nn.ELU(inplace=True))
        ]
        if dropout is not None:
            block.append(('dp', nn.Dropout(dropout)))
        layers.append(('conv{}'.format(i + 1), nn.Sequential(OrderedDict(block))))
        input_dim = output_dim
    return layers


def create_pat_linear_blocks(input_dim, output_dims, groups, dropout=None):
    layers = []
    for i, output_dim in enumerate(output_dims):
        block = [
            ('fc', nn.Linear(input_dim, output_dim, bias=False)),
            ('gn', nn.GroupNorm(groups, output_dim)),
            ('elu', nn.ELU(inplace=True))
        ]
        if dropout is not None:
            block.append(('dp', nn.Dropout(dropout)))
        layers.append(('fc{}'.format(i + 1), nn.Sequential(OrderedDict(block))))
        input_dim = output_dim
    return layers


def k_nearest_neighbors_graph(points, num_neighbor, dilation=None, training=True):
    # ignore the nearest point (itself)
    if dilation > 1:
        num_neighbor_dilated = num_neighbor * dilation
        _, index = k_nearest_neighbors(points, points, num_neighbor_dilated + 1)
        if training:
            device = points.device
            sample_index = torch.randperm(num_neighbor_dilated)[:num_neighbor].to(device)
            index = index[:, :, 1:].index_select(dim=2, index=sample_index).contiguous()
        else:
            index = index[:, :, 1:].contiguous()
    else:
        _, index = k_nearest_neighbors(points, points, num_neighbor + 1)
        index = index[:, :, 1:].contiguous()
    points = F.group_gather_by_index(points, index)
    return points


def farthest_sampling_and_gather(points, features, num_sample):
    index = F.farthest_point_sampling(points, num_sample)
    points = F.gather_by_index(points, index)
    if features is not None:
        features = F.gather_by_index(features, index)
    return points, features
