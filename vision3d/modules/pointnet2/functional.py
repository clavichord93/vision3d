import torch

from .. import geometry

__all__ = [
    'ball_query_and_group_gather',
    'random_ball_query_and_group_gather'
]


def _group_gather_and_concat(points, features, centroids, indices):
    aligned_points = geometry.functional.group_gather(points, indices) - centroids.unsqueeze(3)
    if features is not None:
        features = geometry.functional.group_gather(features, indices)
        features = torch.cat([features, aligned_points], dim=1)
    else:
        features = aligned_points
    return features


def ball_query_and_group_gather(points, features, centroids, num_sample, radius):
    r"""
    Ball query without random sampling and group gathering.

    :param points: torch.Tensor (batch_size, 3, num_point)
        The coordinates of the points.
    :param features: torch.Tensor (batch_size, num_channel, num_point)
        The features of the points.
    :param centroids: torch.Tensor (batch_size, 3, num_centroid)
        The coordinates of the centroids.
    :param num_sample: int
        The number of points sampled for each centroids.
    :param radius: float
        The radius of the balls.

    :return points: torch.Tensor (batch_size, 3, num_centroid, num_sample)
        The coordinates of the sampled points.
    :return features: torch.Tensor (batch_size, num_channel, num_centroid, num_sample)
        The features of the sampled points.
    """
    indices = geometry.functional.ball_query(points, centroids, num_sample, radius)
    features = _group_gather_and_concat(points, features, centroids, indices)
    return features


def random_ball_query_and_group_gather(points, features, centroids, num_sample, radius):
    r"""
    Ball query without random sampling and group gathering.

    :param points: torch.Tensor (batch_size, 3, num_point)
        The coordinates of the points.
    :param features: torch.Tensor (batch_size, num_channel, num_point)
        The features of the points.
    :param centroids: torch.Tensor (batch_size, 3, num_centroid)
        The coordinates of the centroids.
    :param num_sample: int
        The number of points sampled for each centroids.
    :param radius: float
        The radius of the balls.

    :return points: torch.Tensor (batch_size, 3, num_centroid, num_sample)
        The coordinates of the sampled points.
    :return features: torch.Tensor (batch_size, num_channel, num_centroid, num_sample)
        The features of the sampled points.
    """
    indices = geometry.functional.random_ball_query(points, centroids, num_sample, radius)
    features = _group_gather_and_concat(points, features, centroids, indices)
    return features
