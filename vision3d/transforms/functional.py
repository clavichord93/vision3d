import torch

import numpy as np
import random
import math


def normalize_point_cloud(points):
    centroid = np.mean(points, axis=0)
    points = points - centroid
    normal = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
    points = points / normal
    return points


def sample_point_cloud(points, num_sample):
    points = points[:num_sample]
    return points


def random_translate_point_cloud(points):
    scale = np.random.uniform(low=2./3., high=3./2., size=(1, 3))
    bias = np.random.uniform(low=-0.2, high=0.2, size=(1, 3))
    points = points * scale + bias
    return points


def random_rotate_point_cloud(points):
    theta = random.random() * 2 * math.pi
    matrix_t = np.array([[math.cos(theta), math.sin(theta), 0],
                         [-math.sin(theta), math.cos(theta), 0],
                         [0, 0, 1]])
    points = np.matmul(points, matrix_t)
    return points


def random_rescale_point_cloud(points, low=0.8, high=1.2):
    scale = random.uniform(low, high)
    points = points * scale
    return points


def random_jitter_point_cloud(points, sigma, clip=0.05):
    noise = np.clip(sigma * np.random.randn(*points.shape), -clip, clip)
    points = points + noise
    return points


def random_shuffle_point_cloud(points):
    index = np.arange(points.shape[0])
    np.random.shuffle(index)
    points = points[index]
    return points


def random_dropout_point_cloud(points, max_dropout_ratio):
    num_point = points.shape[0]
    dropout_ratio = np.random.rand(num_point) * max_dropout_ratio
    dropped_indices = np.nonzero(np.random.rand(num_point) < dropout_ratio)[0]
    points[dropped_indices, :] = points[0, :]
    return points
