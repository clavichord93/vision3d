import torch.utils.data
import numpy as np
from tqdm import tqdm

from vision3d.datasets.shapenetpart import ShapeNetPartNormalDataset
import vision3d.transforms.functional as F
from vision3d.utils.pytorch_utils import reset_numpy_random_seed


class TrainTransform(object):
    def __init__(self, num_point, sigma):
        self.num_point = num_point
        self.sigma = sigma

    def __call__(self, points, normals, labels):
        num_point = points.shape[0]
        # random sampling
        index = np.random.choice(num_point, self.num_point, replace=True)
        points = points[index]
        normals = normals[index]
        labels = labels[index]
        # normalize
        points = F.normalize_point_cloud(points)
        # random jitter
        points = F.random_jitter_point_cloud(points, self.sigma)
        # transpose
        points = points.transpose()
        normals = normals.transpose()
        # to tensor
        points = torch.tensor(points, dtype=torch.float)
        normals = torch.tensor(normals, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.long)
        return points, normals, labels

    def __repr__(self):
        format_string = self.__class__.__name__ + '(\n'
        format_string += '    RandomSamplePointCloud({})\n'.format(self.num_point)
        format_string += '    NormalizePointCloud()\n'
        format_string += '    RandomJitterPointCloud({})\n'.format(self.sigma)
        format_string += ')'
        return format_string


class TestTransform(object):
    def __call__(self, points, normals, labels):
        # normalize
        points = F.normalize_point_cloud(points)
        # transpose
        points = points.transpose()
        normals = normals.transpose()
        # to tensor
        points = torch.tensor(points, dtype=torch.float)
        normals = torch.tensor(normals, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.long)
        return points, normals, labels

    def __repr__(self):
        format_string = self.__class__.__name__ + '(\n'
        format_string += '    NormalizePointCloud()\n'
        format_string += ')'
        return format_string


def train_data_loader(root, split, config):
    train_transform = TrainTransform(config.num_point, config.sigma)
    train_dataset = ShapeNetPartNormalDataset(root, split, train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               num_workers=config.num_worker,
                                               worker_init_fn=reset_numpy_random_seed,
                                               pin_memory=True)
    return train_loader


def test_data_loader(root, split, config):
    test_transform = TestTransform()
    test_dataset = ShapeNetPartNormalDataset(root, split, test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=config.batch_size,
                                              num_workers=config.num_worker,
                                              worker_init_fn=reset_numpy_random_seed)
    return test_loader


if __name__ == '__main__':
    from config import config

    # train_data_loader = train_data_loader(config.PATH.data_root, 'train', config.TRAIN)
    # statistics = np.zeros(config.DATA.num_class, dtype=np.int)
    # for i, (x, y) in enumerate(train_data_loader):
    #     for j in range(config.DATA.num_class):
    #         statistics[j] += np.count_nonzero(y.numpy() == j)
    # print(statistics)

    test_data_loader = test_data_loader(config.PATH.data_root, 'test', config.TEST)
    statistics = np.zeros(config.DATA.num_part, dtype=np.int)
    pbar = tqdm(test_data_loader)
    for points, normals, labels, class_ids in pbar:
        for j in range(config.DATA.num_part):
            statistics[j] += np.count_nonzero(labels.numpy() == j)
    for part_name, num_point in zip(config.DATA.part_names, statistics):
        print('{}: {}'.format(part_name, num_point))
