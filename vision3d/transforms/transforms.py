import torch

import numpy as np
from random import random
import math


class SamplePointCloud(object):
    def __init__(self, num_sample):
        self.num_sample = num_sample

    def __call__(self, inputs):
        points, features = inputs
        points = points[:self.num_sample, :]
        if features is not None:
            features = features[:self.num_sample, :]
        return points, features

    def __repr__(self):
        return self.__class__.__name__ + '(num_sample={})'.format(self.num_sample)


class RandomRotatePointCloud(object):
    def __call__(self, inputs):
        points, features = inputs
        theta = random() * 2 * math.pi
        matrix_t = np.array([[math.cos(theta), math.sin(theta), 0],
                             [-math.sin(theta), math.cos(theta), 0],
                             [0, 0, 1]])
        print(points.shape)
        points = np.matmul(points, matrix_t)
        return points, features

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomJitterPointCloud(object):
    def __init__(self, sigma, clip=0.05):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, inputs):
        points, features = inputs
        noise = np.clip(self.sigma * np.random.randn(*points.shape), -self.clip, self.clip)
        points = points + noise
        return points, features

    def __repr__(self):
        return self.__class__.__name__ + '(sigma={}, clip={})'.format(self.sigma, self.clip)


class RandomShufflePointCloud(object):
    def __call__(self, inputs):
        points, features = inputs
        indices = np.arange(points.shape[0])
        np.random.shuffle(indices)
        points = points[indices]
        if features is not None:
            features = features[indices]
        return points, features

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomDropoutPointCloud(object):
    def __init__(self, max_dropout_ratio):
        self.max_dropout_ratio = max_dropout_ratio

    def __call__(self, inputs):
        points, features = inputs
        num_point = points.shape[0]
        dropout_ratio = np.random.rand(num_point) * self.max_dropout_ratio
        dropped_indices = np.nonzero(np.random.rand(num_point) < dropout_ratio)[0]
        points[dropped_indices, :] = points[0, :]
        if features is not None:
            features[dropped_indices, :] = features[0, :]
        return points, features

    def __repr__(self):
        return self.__class__.__name__ + '(max_dropout_ratio={})'.format(self.max_dropout_ratio)


class TransposePointCloud(object):
    def __call__(self, inputs):
        points, features = inputs
        points = points.transpose()
        if features is not None:
            features = features.transpose()
        return points, features

    def __repr__(self):
        return self.__class__.__name__ + '()'


class PointCloudToTensor(object):
    def __call__(self, inputs):
        points, features = inputs
        points = torch.tensor(points, dtype=torch.float)
        if features is not None:
            features = torch.tensor(features, dtype=torch.float)
        return points, features

    def __repr__(self):
        return self.__class__.__name__ + '()'
