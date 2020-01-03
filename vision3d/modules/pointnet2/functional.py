import random

import torch
from torch.autograd import Function

import vision3d.ext as ext


__all__ = [
    'gather_by_index',
    'group_gather_by_index',
    'ball_query',
    'ball_query_and_group_gather',
    'random_ball_query',
    'random_ball_query_and_group_gather',
    'farthest_point_sampling',
    'farthest_point_sampling_and_gather',
    'three_nearest_neighbors',
    'three_interpolate',
]


class GatherByIndexFunction(Function):
    r"""
    Gather by index.
    Gather `num_sample` points from a set of `num_point` points.

    :param points: torch.Tensor (batch_size, num_channel, num_point)
        The features/coordinates of all points.
    :param index: torch.Tensor (batch_size, num_sample)
        The indices of the points to gather.

    :return output: torch.Tensor (batch, num_channel, num_sample)
        The features/coordinates of the gathered points.
    """
    @staticmethod
    def forward(ctx, points, index):
        batch_size, num_channel, num_point = points.shape
        ctx.save_for_backward(index)
        ctx.num_point = num_point
        output = ext.gather_by_index(points, index)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        index = ctx.saved_tensors[0]
        num_point = ctx.num_point
        grad_output = grad_output.contiguous()
        grad_points = ext.gather_by_index_grad(grad_output, index, num_point)
        return grad_points, None


gather_by_index = GatherByIndexFunction.apply


class GroupGatherByIndexFunction(Function):
    r"""
    Group gather by index.
    Gather `num_sample` points for each of the `num_centroid` centroids from a set of `num_point` points.

    :param points: torch.Tensor (batch_size, num_channel, num_point)
        The features/coordinates of all points.
    :param index: torch.Tensor (batch_size, num_centroid, num_sample)
        The indices of the points to gather.

    :return output: torch.Tensor (batch, num_channel, num_centroid, num_sample)
        The features/coordinates of the gathered points.
    """
    @staticmethod
    def forward(ctx, points, index):
        batch_size, num_channel, num_point = points.shape
        ctx.save_for_backward(index)
        ctx.num_point = num_point
        output = ext.group_gather_by_index(points, index)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        index = ctx.saved_tensors[0]
        num_point = ctx.num_point
        grad_output = grad_output.contiguous()
        grad_points = ext.group_gather_by_index_grad(grad_output, index, num_point)
        return grad_points, None


group_gather_by_index = GroupGatherByIndexFunction.apply


def farthest_point_sampling(points, num_sample):
    r"""
    Farthest point sampling.

    :param points: torch.Tensor (batch_size, 3, num_point)
        The coordinates of the points.
    :param num_sample: int
        The number of centroids sampled.
    :return index: torch.Tensor (batch_size, num_sample)
        The indices of the sampled centroids.
    """
    index = ext.farthest_point_sampling(points, num_sample)
    return index


def farthest_point_sampling_and_gather(points, num_sample):
    r"""
    Farthest point sampling and gather the coordinates of the sampled points.

    :param points: torch.Tensor (batch_size, 3, num_point)
        The coordinates of the points.
    :param num_sample: int
        The number of centroids sampled.
    :return centroids: torch.Tensor (batch_size, 3, num_sample)
        The indices of the sampled centroids.
    """
    index = ext.farthest_point_sampling(points, num_sample)
    centroids = gather_by_index(points, index)
    return centroids


def ball_query(points, centroids, num_sample, radius):
    r"""
    Ball query without random sampling.
    The first `num_sample` points in the ball for each centroids are returned.

    :param points: torch.Tensor (batch_size, 3, num_point)
        The coordinates of the points.
    :param centroids: torch.Tensor (batch_size, 3, num_centroid)
        The coordinates of the centroids.
    :param num_sample: int
        The number of points sampled for each centroids.
    :param radius: float
        The radius of the balls.

    :return index: torch.Tensor (batch_size, num_centroid, num_sample)
        The indices of the sampled points.
    """
    index = ext.ball_query_v1(centroids, points, radius, num_sample)
    return index


def random_ball_query(points, centroids, num_sample, radius):
    r"""
    Ball query with random sampling.
    All points in the ball are found, then `num_sample` points are randomly sampled as the results.

    :param points: torch.Tensor (batch_size, 3, num_point)
        The coordinates of the points.
    :param centroids: torch.Tensor (batch_size, 3, num_centroid)
        The coordinates of the centroids.
    :param num_sample: int
        The number of points sampled for each centroids.
    :param radius: float
        The radius of the balls.

    :return index: torch.Tensor (batch_size, num_centroid, num_sample)
        The indices of the sampled points.
    """
    seed = random.randint()
    index = ext.ball_query_v2(seed, centroids, points, radius, num_sample)
    return index


def _group_gather_and_concat(points, features, centroids, index):
    points = group_gather_by_index(points, index)
    aligned_points = points - centroids.unsqueeze(3)
    if features is not None:
        features = group_gather_by_index(features, index)
        features = torch.cat([features, aligned_points], dim=1)
    else:
        features = aligned_points
    return points, features


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
    index = ext.ball_query_v1(centroids, points, radius, num_sample)
    points, features = _group_gather_and_concat(points, features, centroids, index)
    return points, features


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
    seed = random.randint(0, 2147483647)
    index = ext.ball_query_v2(seed, centroids, points, radius, num_sample)
    points, features = _group_gather_and_concat(points, features, centroids, index)
    return points, features


def three_nearest_neighbors(points1, points2):
    r"""
    Compute the three nearest neighbors for the non-sampled points in the sub-sampled points.

    :param points1: torch.Tensor (batch_size, 3, num_point1)
        The points to compute the three nearest neighbors.
    :param points2: torch.Tensor (batch_size, 3, num_point2)
        The sub-sampled points set.
    :return dist2: torch.Tensor (batch_size, num_point1, 3)
        The corresponding squared distance of the found neighbors to the point.
    :return index: torch.Tensor (batch_size, num_point1, 3)
        The indices of the found neighbors.
    """
    dist2, index = ext.three_nearest_neighbors(points1, points2)
    return dist2, index


class ThreeInterpolateFunction(Function):
    r"""
    Interpolate the features for the non-sampled points from the sub-sampled points.
    Three sub-sampled points are used to interpolate one non-sampled point.

    :param features: torch.Tensor (batch_size, num_channel, num_point2)
        The features of the sub-sampled points.
    :param index: torch.Tensor (batch, num_point1, 3)
        The indices of the sub-sampled points to interpolate each non-sampled points.
    :param weight: torch.Tensor (batch, num_point1, 3)
        The weights of the sub-sampled points to interpolate each non-sampled points.
    :return outputs: torch.Tensor (batch_size, num_channel, num_point1)
        The interpolated features of the non-sampled points.
    """
    @staticmethod
    def forward(ctx, features, index, weights):
        batch_size, num_channel, num_point2 = features.shape
        ctx.save_for_backward(index, weights)
        ctx.num_point2 = num_point2
        outputs = ext.three_interpolate(features, index, weights)
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        index, weights = ctx.saved_tensors
        num_point2 = ctx.num_point2
        grad_outputs = grad_outputs.contiguous()
        grad_features = ext.three_interpolate_grad(grad_outputs, index, weights, num_point2)
        return grad_features, None, None


three_interpolate = ThreeInterpolateFunction.apply
