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


def k_nearest_neighbors_graph(points, num_neighbor, dilation=None):
    if dilation > 1:
        device = points.device
        num_dilated_neighbor = num_neighbor * dilation
        _, index = k_nearest_neighbors(points, points, num_dilated_neighbor + 1)
        # ignore the nearest point (itself)
        sample_index = torch.randperm(num_dilated_neighbor)[:num_neighbor].to(device)
        index = index[:, :, 1:].index_select(dim=2, index=sample_index).contiguous()
    else:
        _, index = k_nearest_neighbors(points, points, num_neighbor + 1)
        index = index[:, :, 1:].contiguous()
    points = F.group_gather_by_index(points, index)
    return points
