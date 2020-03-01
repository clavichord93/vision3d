from collections import OrderedDict

import torch
import torch.nn as nn

from .. import geometry
from ...utils.pytorch_utils import (create_conv1d_blocks,
                                    create_conv2d_blocks,
                                    SeparableConv2d,
                                    DepthwiseConv2d,
                                    ConvBlock2d)


class XReshape(nn.Module):
    def __init__(self, kernel_size):
        super(XReshape, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, points):
        batch_size = points.shape[0]
        points = points.squeeze(2).view(batch_size, self.kernel_size, self.kernel_size, -1).transpose(1, 2)
        return points


class XSharedMLP(nn.Module):
    def __init__(self, input_dim, num_layer, kernel_size):
        super(XSharedMLP, self).__init__()
        self.kernel_size = kernel_size
        self.stem = ConvBlock2d(input_dim,
                                self.kernel_size * self.kernel_size,
                                kernel_size=(1, self.kernel_size),
                                batch_norm_after_activation=True,
                                activation='elu')
        layers = []
        for i in range(num_layer):
            layers.append(('conv{}'.format(i + 1),
                           DepthwiseConv2d(self.kernel_size,
                                           kernel_size=(self.kernel_size, 1),
                                           depth_multiplier=self.kernel_size,
                                           batch_norm_after_activation=True,
                                           activation='elu')))
            layers.append(('reshape{}'.format(i + 1), XReshape(self.kernel_size)))
        self.layers = nn.Sequential(OrderedDict(layers))

    def forward(self, points):
        batch_size = points.shape[0]
        points = self.stem(points)
        points = points.squeeze(3).view(batch_size, self.kernel_size, self.kernel_size, -1)
        points = self.layers(points)
        return points


class XConv(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dim,
                 kernel_size,
                 dilation=1,
                 depth_multiplier=1,
                 with_global=False):
        super(XConv, self).__init__()

        self.kernel_size = kernel_size
        self.num_neighbor = kernel_size
        self.dilation = dilation
        self.depth_multiplier = depth_multiplier
        self.with_global = with_global

        layers = create_conv2d_blocks(3,
                                      [hidden_dim, hidden_dim],
                                      kernel_size=1,
                                      batch_norm_after_activation=True,
                                      activation='elu')
        self.f_shared_mlp = nn.Sequential(OrderedDict(layers))

        self.x_shared_mlp = XSharedMLP(3, num_layer=2, kernel_size=self.kernel_size)

        self.conv = SeparableConv2d(input_dim + hidden_dim,
                                    output_dim,
                                    kernel_size=(1, self.kernel_size),
                                    depth_multiplier=self.depth_multiplier,
                                    batch_norm_after_activation=True,
                                    activation='elu')

        if self.with_global:
            global_dim = output_dim // 4
            layers = create_conv1d_blocks(3,
                                          [global_dim, global_dim],
                                          kernel_size=1,
                                          batch_norm_after_activation=True,
                                          activation='elu')
            self.g_conv = nn.Sequential(OrderedDict(layers))

    def forward(self, points, features, centroids):
        indices = geometry.functional.dilated_k_nearest_neighbors(points, centroids, self.num_neighbor, self.dilation)
        aligned_points = geometry.functional.group_gather(points, indices) - centroids.unsqueeze(3)
        features1 = self.f_shared_mlp(aligned_points)
        if features is not None:
            features = geometry.functional.group_gather(features, indices)
            features = torch.cat([features, features1], dim=1)
        else:
            features = features1
        x = self.x_shared_mlp(aligned_points).transpose(1, 3)
        features = torch.matmul(features.transpose(1, 2), x).transpose(1, 2)
        features = self.conv(features).squeeze(3)
        if self.with_global:
            g_features = self.g_conv(centroids)
            features = torch.cat([features, g_features], dim=1)
        return centroids, features
