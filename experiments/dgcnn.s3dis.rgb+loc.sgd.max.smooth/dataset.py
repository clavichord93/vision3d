import time

import torch.utils.data
import numpy as np
from tqdm import tqdm

from vision3d.datasets import S3DISDataset, S3DISWholeSceneDataset, S3DISWholeSceneHdf5Dataset
import vision3d.transforms.functional as F
from vision3d.utils.pytorch_utils import reset_numpy_random_seed


class TrainTransform(object):
    def __call__(self, points, features, labels):
        # transpose
        points = points.transpose()
        features = features.transpose()
        # to tensor
        points = torch.tensor(points, dtype=torch.float)
        features = torch.tensor(features, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.long)
        return points, features, labels

    def __repr__(self):
        format_string = self.__class__.__name__ + '(\n'
        format_string += '    TransposePointCloud()\n'
        format_string += '    ToTensor()\n'
        format_string += ')'
        return format_string


class TestTransform(object):
    def __call__(self, points, features, point_indices, labels):
        # transpose
        points = points.transpose(0, 2, 1)
        features = features.transpose(0, 2, 1)
        # to tensor
        points = torch.tensor(points, dtype=torch.float)
        features = torch.tensor(features, dtype=torch.float)
        point_indices = torch.tensor(point_indices, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        return points, features, point_indices, labels

    def __repr__(self):
        format_string = self.__class__.__name__ + '(\n'
        format_string += '    TransposePointCloud()\n'
        format_string += '    ToTensor()\n'
        format_string += ')'
        return format_string


def train_data_loader(config, test_area):
    train_transform = TrainTransform()
    train_dataset = S3DISDataset(config.data_root,
                                 train_transform,
                                 test_area=test_area,
                                 num_sample=config.train_num_point,
                                 block_size=config.train_block_size,
                                 training=True,
                                 normalized_location=config.train_normalized_location)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config.train_batch_size,
                                               shuffle=True,
                                               num_workers=config.train_num_worker,
                                               pin_memory=True,
                                               drop_last=True,
                                               worker_init_fn=reset_numpy_random_seed)
    return train_loader


def test_data_loader(config, test_area):
    test_transform = TestTransform()
    test_dataset = S3DISWholeSceneHdf5Dataset(config.testing_hdf5_root,
                                              test_transform,
                                              test_area=test_area,
                                              normalized_location=config.test_normalized_location)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1,
                                              num_workers=config.test_num_worker,
                                              pin_memory=True,
                                              worker_init_fn=reset_numpy_random_seed)
    return test_loader


if __name__ == '__main__':
    from config import config

    # data_loader = train_data_loader(config, test_area=5)
    # start_time = time.time()
    # for batch in train_data_loader:
    #     points, features, labels = batch
    #     print(points.shape, features.shape, labels.shape, ':', time.time() - start_time)
    #     start_time = time.time()

    start_time = time.time()
    for test_area in range(1, 7):
        print('Test area: {}'.format(test_area))
        data_loader = test_data_loader(config, test_area=test_area)
        print('Data loader created: {:.3f}s collapsed.'.format(time.time() - start_time))
        pbar = tqdm(data_loader)
        start_time = time.time()
        for points, features, point_indices, labels in pbar:
            print('{:.3f}s collapsed.'.format(time.time() - start_time))
            start_time = time.time()
