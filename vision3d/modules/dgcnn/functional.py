import torch
import torch.nn as nn

from ..pointnet2 import functional as F
from ...utils.pytorch_utils import k_nearest_neighbors


__all__ = [
    'dynamic_graph_update'
]


def dynamic_graph_update(points, centroids, num_neighbor):
    r"""
    Dynamic graph update proposed in \"Dynamic Graph CNN for Learning on Point Clouds\"

    :param points: torch.Tensor (batch_size, num_channel, num_point)
        The features/coordinates of the whole point set.
    :param centroids: torch.Tensor (batch_size, num_channel, num_centroid)
        The features/coordinates of the centroids.
    :param num_neighbor: int
        The number of kNNs for each centroid.
    :return neighbors: torch.Tensor (batch_size, 2 * num_channel, num_centroid, num_neighbor)
        The concatenated features/coordinates of the kNNs for the centroids.
    """
    _, index = k_nearest_neighbors(points, centroids, num_neighbor)
    neighbors = F.group_gather_by_index(points, index)
    points = points.unsqueeze(3).repeat(1, 1, 1, num_neighbor)
    differences = neighbors - points
    features = torch.cat([points, differences], dim=1)
    return features
