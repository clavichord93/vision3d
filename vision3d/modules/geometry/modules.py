import torch
import torch.nn as nn

from . import functional as F


class FarthestPointSampling(nn.Module):
    def __init__(self, num_sample):
        super(FarthestPointSampling, self).__init__()
        self.num_sample = num_sample

    def forward(self, points):
        samples = F.farthest_point_sampling_and_gather(points, self.num_sample)
        return samples


class RandomPointSampling(nn.Module):
    def __init__(self, num_sample):
        super(RandomPointSampling, self).__init__()
        self.num_sample = num_sample

    def forward(self, points):
        samples = F.random_point_sampling_and_gather(points, self.num_sample)
        return samples
