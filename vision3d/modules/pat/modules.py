import math

import torch
import torch.nn as nn


class AbsoluteRelativePositionEmbedding(nn.Module):
    def __init__(self, num_neighbor, base_dilation, base_num_point):
        super(AbsoluteRelativePositionEmbedding, self).__init__()
        self.num_neighbor = num_neighbor
        self.base_dilation = base_dilation
        self.base_num_point = base_num_point

    def forward(self, points):
        NotImplemented


class GroupShuffleAttention(nn.Module):
    def __init__(self, feature_dim, num_group):
        super(GroupShuffleAttention, self).__init__()
        self.num_group = num_group
        self.conv = nn.Conv1d(feature_dim, feature_dim, kernel_size=1, groups=num_group)
        self.norm = nn.GroupNorm(num_groups=32, num_channels=feature_dim)

    def forward(self, points):
        identity = points
        batch_size, num_channel, num_point = points.shape
        num_channel_per_group = num_channel // self.num_group
        points = self.conv(points)
        points = points.view(batch_size, self.num_group, num_channel_per_group, num_point)
        queries = points.transpose(2, 3)
        attention = torch.matmul(queries, points) / math.sqrt(num_channel_per_group)
        attention = nn.functional.softmax(attention, dim=2)
        points = nn.functional.elu(points)
        points = torch.matmul(points, attention)
        points = points.transpose(1, 2).contiguous().view(batch_size, num_channel, num_point)
        points += identity
        points = self.norm(points)

        return points


class GumbelSubsetSampling(nn.Module):
    def __init__(self, num_sample, temperature):
        super(GumbelSubsetSampling, self).__init__()
        self.num_sample = num_sample
        self.temperature = temperature

    def forward(self, points):
        NotImplemented
