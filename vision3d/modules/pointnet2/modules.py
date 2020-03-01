from collections import OrderedDict

import torch
import torch.nn as nn

from .. import geometry
from . import functional as F
from ...utils.pytorch_utils import create_conv1d_blocks, create_conv2d_blocks


class SetAbstractionModule(nn.Module):
    def __init__(self, input_dim, output_dims, num_centroid, num_sample, radius):
        super(SetAbstractionModule, self).__init__()
        self.num_centroid = num_centroid
        self.num_sample = num_sample
        self.radius = radius
        layers = create_conv2d_blocks(input_dim, output_dims, kernel_size=1)
        self.pointnet = nn.Sequential(OrderedDict(layers))

    def forward(self, points, features):
        centroids = geometry.functional.farthest_point_sampling_and_gather(points, self.num_centroid)
        features = F.ball_query_and_group_gather(points, features, centroids, self.num_sample, self.radius)
        features = self.pointnet(features)
        features, _ = features.max(dim=3)
        return centroids, features


class MultiScaleSetAbstractionModule(nn.Module):
    def __init__(self, input_dim, output_dims_list, num_centroid, num_samples, radii):
        super(MultiScaleSetAbstractionModule, self).__init__()
        if len(output_dims_list) != len(num_samples):
            raise ValueError('The sizes of output_dims_list and num_samples do not match.')
        if len(output_dims_list) != len(radii):
            raise ValueError('The sizes of output_dims_list and radii do not match.')
        self.num_centroid = num_centroid
        self.num_samples = num_samples
        self.radii = radii
        self.pointnets = nn.ModuleList()
        for i, output_dims in enumerate(output_dims_list):
            layers = create_conv2d_blocks(input_dim, output_dims, kernel_size=1)
            self.pointnets.append(nn.Sequential(OrderedDict(layers)))

    def forward(self, points, features):
        centroids = geometry.functional.farthest_point_sampling_and_gather(points, self.num_centroid)
        overall_features = []
        for i, (num_sample, radius) in enumerate(zip(self.num_samples, self.radii)):
            current_features = F.ball_query_and_group_gather(points, features, centroids, num_sample, radius)
            current_features = self.pointnets[i](current_features)
            current_features, _ = current_features.max(dim=3)
            overall_features.append(current_features)
        features = torch.cat(overall_features, dim=1)
        return centroids, features


class GlobalAbstractionModule(nn.Module):
    def __init__(self, input_dim, output_dims):
        super(GlobalAbstractionModule, self).__init__()
        layers = create_conv1d_blocks(input_dim, output_dims, kernel_size=1)
        self.pointnet = nn.Sequential(OrderedDict(layers))

    def forward(self, points, features):
        device = points.device
        batch_size, num_coordinate, _ = points.shape
        features = torch.cat([features, points], dim=1)
        features = self.pointnet(features)
        features, _ = features.max(dim=2, keepdim=True)
        centroids = torch.zeros(batch_size, num_coordinate, 1).to(device)
        return centroids, features


class FeaturePropagationModule(nn.Module):
    r"""
    Feature Propagation Module.

    :param points1: torch.Tensor (batch_size, 3, num_point1)
        The coordinates of the non-sampled points.
    :param features1: torch.Tensor (batch_size, num_channel1, num_point1)
        The features of the non-sampled points.
    :param points2: torch.Tensor (batch_size, 3, num_point2)
        The coordinates of the sub-sampled points.
    :param features2: torch.Tensor (batch_size, num_channel2, num_point2)
        The features of the sub-sampled points.
    :return features1: torch.Tensor (batch_size, num_channel1 + num_channel2, num_point1)
        The concatenation of the original features and the interpolated features of the non-sampled points.
    """
    def __init__(self, input_dim, output_dims):
        super(FeaturePropagationModule, self).__init__()
        layers = create_conv1d_blocks(input_dim, output_dims, kernel_size=1)
        self.pointnet = nn.Sequential(OrderedDict(layers))

    def forward(self, points1, features1, points2, features2):
        num_point1 = points1.shape[2]
        num_point2 = points2.shape[2]
        if num_point2 == 1:
            features2 = features2.repeat(1, 1, num_point1)
        else:
            dist2, indices = geometry.functional.three_nearest_neighbors(points1, points2)
            weights = torch.div(1., dist2 + 1e-5)
            weights = weights / torch.sum(weights, dim=2, keepdim=True)
            features2 = geometry.functional.three_interpolate(features2, indices, weights)
        if features1 is not None:
            features1 = torch.cat([features1, features2], dim=1)
        else:
            features1 = features2
        features1 = self.pointnet(features1)
        return features1
