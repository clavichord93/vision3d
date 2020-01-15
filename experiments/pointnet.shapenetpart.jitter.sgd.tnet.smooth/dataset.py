import torch.utils.data
import numpy as np
from tqdm import tqdm

from vision3d.datasets import ShapeNetPartDataset
import vision3d.transforms.functional as F
from vision3d.utils.pytorch_utils import reset_numpy_random_seed


class TrainTransform(object):
    def __init__(self, num_point, sigma):
        self.num_point = num_point
        self.sigma = sigma

    def __call__(self, points, labels):
        num_point = points.shape[0]
        # random sampling
        indices = np.random.choice(num_point, self.num_point, replace=True)
        points = points[indices]
        labels = labels[indices]
        # normalize
        points = F.normalize_point_cloud(points)
        # random jitter
        points = F.random_jitter_point_cloud(points, self.sigma)
        # transpose
        points = points.transpose()
        # to tensor
        points = torch.tensor(points, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.long)
        return points, labels

    def __repr__(self):
        format_string = self.__class__.__name__ + '(\n'
        format_string += '    RandomSamplePointCloud(num_point={})\n'.format(self.num_point)
        format_string += '    NormalizePointCloud()\n'
        format_string += '    RandomJitterPointCloud(sigma={})\n'.format(self.sigma)
        format_string += ')'
        return format_string


class TestTransform(object):
    def __call__(self, points, labels):
        # normalize
        points = F.normalize_point_cloud(points)
        # transpose
        points = points.transpose()
        # to tensor
        points = torch.tensor(points, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.long)
        return points, labels

    def __repr__(self):
        format_string = self.__class__.__name__ + '(\n'
        format_string += '    NormalizePointCloud()\n'
        format_string += ')'
        return format_string


def train_data_loader(config, split):
    train_transform = TrainTransform(config.train_num_point, config.train_jitter_sigma)
    train_dataset = ShapeNetPartDataset(config.data_root, split, train_transform, use_normal=False)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config.train_batch_size,
                                               shuffle=True,
                                               num_workers=config.train_num_worker,
                                               pin_memory=True,
                                               drop_last=True,
                                               worker_init_fn=reset_numpy_random_seed)
    return train_loader


def test_data_loader(config, split):
    test_transform = TestTransform()
    test_dataset = ShapeNetPartDataset(config.data_root, split, test_transform, use_normal=False)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=config.test_batch_size,
                                              num_workers=config.test_num_worker,
                                              worker_init_fn=reset_numpy_random_seed)
    return test_loader


if __name__ == '__main__':
    from config import config

    data_loader = train_data_loader(config, 'train')
    statistics = np.zeros(config.num_class, dtype=np.int)
    for i, (x, y) in enumerate(data_loader):
        for j in range(config.num_class):
            statistics[j] += np.count_nonzero(y.numpy() == j)
    print(statistics)

    data_loader = test_data_loader(config, 'test')
    statistics = np.zeros(config.num_part, dtype=np.int)
    pbar = tqdm(data_loader)
    for points, normals, labels, class_ids in pbar:
        for j in range(config.num_part):
            statistics[j] += np.count_nonzero(labels.numpy() == j)
    for part_name, num_point in zip(config.part_names, statistics):
        print('{}: {}'.format(part_name, num_point))
