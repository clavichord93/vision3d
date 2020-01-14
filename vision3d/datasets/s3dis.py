import os.path as osp

import numpy as np
import torch
import torch.utils.data
import h5py


class _S3DISDatasetBase(torch.utils.data.Dataset):
    num_class = 13
    class_names = [
        'ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
        'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter'
    ]
    areas = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_5', 'Area_6']


class S3DISDataset(_S3DISDatasetBase):
    def __init__(self, root, transform, test_area, training=True, normalized_location=True):
        super(S3DISDataset, self).__init__()
        if test_area not in self.areas:
            raise ValueError('Invalid test_area: {}.'.format(test_area))

        self.transform = transform
        self.normalized_location = normalized_location

        with open(osp.join(root, 'room_filelist.txt')) as f:
            lines = f.readlines()
        if training:
            indices = np.array([test_area not in line for line in lines], dtype=np.bool)
        else:
            indices = np.array([test_area in line for line in lines], dtype=np.bool)

        with open(osp.join(root, 'all_files.txt')) as f:
            lines = f.readlines()
        data_files = [osp.join(root, osp.basename(line.rstrip())) for line in lines]

        points_list = []
        labels_list = []
        for data_file in data_files:
            h5file = h5py.File(data_file)
            points_list.append(h5file['data'][:])
            labels_list.append(h5file['label'][:])
        points_list = np.concatenate(points_list, axis=0)
        labels_list = np.concatenate(labels_list, axis=0)
        points_list = points_list[indices]
        self.points_list = points_list[:, :, :3]
        if self.normalized_location:
            self.features_list = points_list[:, :, 3:]
        else:
            self.features_list = points_list[:, :, 3:6]
        self.labels_list = labels_list[indices]
        self.num_shape = self.points_list.shape[0]

    def __getitem__(self, index):
        points, features, labels = self.transform(self.points_list[index],
                                                  self.features_list[index],
                                                  self.labels_list[index])
        return points, features, labels

    def __len__(self):
        return self.num_shape
