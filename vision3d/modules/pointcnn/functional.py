import torch


def random_point_sampling_and_gather(points, num_sample):
    return points[:, :, :num_sample].contiguous()
