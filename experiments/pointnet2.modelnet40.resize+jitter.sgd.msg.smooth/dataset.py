import torch.utils.data

from vision3d.datasets.modelnet40 import ModelNet40Dataset
import vision3d.transforms.functional as F
from vision3d.utils.pytorch_utils import reset_numpy_random_seed


class TrainTransform(object):
    def __init__(self, num_point, sigma, low, high):
        self.num_point = num_point
        self.sigma = sigma
        self.low = low
        self.high = high

    def __call__(self, points):
        points = F.sample_point_cloud(points, self.num_point)
        points = F.random_shuffle_point_cloud(points)
        points = F.random_rescale_point_cloud(points, self.low, self.high)
        points = F.random_jitter_point_cloud(points, self.sigma)
        points = points.transpose()
        points = torch.tensor(points, dtype=torch.float)
        return points

    def __repr__(self):
        format_string = self.__class__.__name__ + '(\n'
        format_string += '    SamplePointCloud(num_point={})\n'.format(self.num_point)
        format_string += '    RandomShufflePointCloud()\n'
        format_string += '    RandomRescalePointCloud(low={}, high={})\n'.format(self.low, self.high)
        format_string += '    RandomJitterPointCloud(sigma={})\n'.format(self.sigma)
        format_string += ')'
        return format_string


class TestTransform(object):
    def __init__(self, num_point):
        self.num_point = num_point

    def __call__(self, points):
        points = F.sample_point_cloud(points, self.num_point)
        points = points.transpose()
        points = torch.tensor(points, dtype=torch.float)
        return points

    def __repr__(self):
        format_string = self.__class__.__name__ + '(\n'
        format_string += '    SamplePointCloud(num_point={})\n'.format(self.num_point)
        format_string += ')'
        return format_string


def train_data_loader(config):
    train_transform = TrainTransform(config.train_num_point,
                                     config.train_jitter_sigma,
                                     config.train_rescale_low,
                                     config.train_rescale_high)
    train_dataset = ModelNet40Dataset(config.data_root, 'train', train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config.train_batch_size,
                                               shuffle=True,
                                               num_workers=config.train_num_worker,
                                               pin_memory=True,
                                               drop_last=True,
                                               worker_init_fn=reset_numpy_random_seed)
    return train_loader


def test_data_loader(config):
    test_transform = TestTransform(config.test_num_point)
    test_dataset = ModelNet40Dataset(config.data_root, 'test', test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=config.test_batch_size,
                                              num_workers=config.test_num_worker,
                                              worker_init_fn=reset_numpy_random_seed)
    return test_loader


if __name__ == '__main__':
    from config import config

    train_data_loader = train_data_loader(config)
    for i, (x, y) in enumerate(train_data_loader):
        print(i, ': ', x.shape, y.shape)

    test_data_loader = test_data_loader(config)
    for i, (x, y) in enumerate(test_data_loader):
        print(i, ': ', x.shape, y.shape)
