from collections import OrderedDict

import torch
import torch.nn as nn

from . import functional as F
from ...utils.pytorch_utils import create_conv2d_blocks


class EdgeConv(nn.Module):
    r"""
    EdgeConv proposed in \"Dynamic Graph CNN for Learning on Point Clouds\".

    :param points: torch.Tensor (batch_size, num_channel1, num_point)
    :return features: torch.Tensor (batch_size, num_channel2, num_point)
    """
    def __init__(self, input_dim, output_dims, num_neighbor, activation='relu', negative_slope=None):
        super(EdgeConv, self).__init__()
        layers = create_conv2d_blocks(input_dim * 2,
                                      output_dims,
                                      kernel_size=1,
                                      activation=activation,
                                      negative_slope=negative_slope)
        self.shared_mlp = nn.Sequential(OrderedDict(layers))
        self.num_neighbor = num_neighbor

    def forward(self, points):
        features = F.dynamic_graph_update(points, points, self.num_neighbor)
        features = self.shared_mlp(features)
        features, _ = features.max(dim=3)
        return features
