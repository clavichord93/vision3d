from collections import OrderedDict

import torch
import torch.nn as nn

from .. import geometry

__all__ = [
    'create_pat_conv1d_blocks',
    'create_pat_conv2d_blocks',
    'create_pat_linear_blocks',
    'k_nearest_neighbors_graph',
    'farthest_point_sampling_and_gather'
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


def k_nearest_neighbors_graph(points, num_neighbor, dilation=1, training=True, ignore_nearest=True):
    num_neighbor_dilated = num_neighbor * dilation + int(ignore_nearest)
    indices = geometry.functional.k_nearest_neighbors(points, points, num_neighbor_dilated)
    start_index = 1 if ignore_nearest else 0
    # if dilation > 1 and training:
    #     device = points.device
    #     sample_indices = torch.randperm(num_neighbor_dilated)[:num_neighbor].to(device)
    #     indices = indices.index_select(dim=2, index=sample_indices)
    # indices = indices.contiguous()
    if training:
        indices = indices[:, :, start_index::dilation].contiguous()
    else:
        indices = indices[:, :, start_index:].contiguous()
    points = geometry.functional.group_gather(points, indices)
    return points


def farthest_point_sampling_and_gather(points, features, num_sample):
    indices = geometry.functional.farthest_point_sampling(points, num_sample)
    points = geometry.functional.gather(points, indices)
    if features is not None:
        features = geometry.functional.gather(features, indices)
    return points, features
