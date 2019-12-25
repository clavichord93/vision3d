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
        points = F.random_resize_point_cloud(points, self.low, self.high)
        points = F.random_jitter_point_cloud(points, self.sigma)
        points = points.transpose()
        points = torch.tensor(points, dtype=torch.float)
        return points

    def __repr__(self):
        format_string = self.__class__.__name__ + '(\n'
        format_string += '    SamplePointCloud({})\n'.format(self.num_point)
        format_string += '    RandomShufflePointCloud()\n'
        format_string += '    RandomResizePointCloud(low={}, high={})\n'.format(self.low, self.high)
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
        format_string += '    SamplePointCloud({})\n'.format(self.num_point)
        format_string += ')'
        return format_string


def train_data_loader(root, config):
    train_transform = TrainTransform(config.num_point, config.sigma, config.low, config.high)
    train_dataset = ModelNet40Dataset(root, 'train', train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               num_workers=config.num_worker,
                                               worker_init_fn=reset_numpy_random_seed,
                                               pin_memory=True)
    return train_loader


def test_data_loader(root, config):
    test_transform = TestTransform(config.num_point)
    test_dataset = ModelNet40Dataset(root, 'test', test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=config.batch_size,
                                              num_workers=config.num_worker,
                                              worker_init_fn=reset_numpy_random_seed)
    return test_loader


if __name__ == '__main__':
    from config import config

    train_data_loader = train_data_loader(config.PATH.data_root, config.TRAIN)
    for i, (x, y) in enumerate(train_data_loader):
        print(i, ": ", x.shape, y.shape)

    test_data_loader = test_data_loader(config.PATH.data_root, config.TEST)
    for i, (x, y) in enumerate(test_data_loader):
        print(i, ": ", x.shape, y.shape)
