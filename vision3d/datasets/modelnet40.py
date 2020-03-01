import os
import os.path as osp

import h5py
import torch.utils.data
import numpy as np


class _ModelNet40DatasetBase(torch.utils.data.Dataset):
    num_class = 40
    class_names = [
        'airplane', 'bathtub', 'bed', 'bench', 'bookshelf',
        'bottle', 'bowl', 'car', 'chair', 'cone',
        'cup', 'curtain', 'desk', 'door', 'dresser',
        'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp',
        'laptop', 'mantel', 'monitor', 'night_stand', 'person',
        'piano', 'plant', 'radio', 'range_hood', 'sink',
        'sofa', 'stairs', 'stool', 'table', 'tent',
        'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox'
    ]


class ModelNet40Dataset(_ModelNet40DatasetBase):
    def __init__(self, root, split, transform, use_normal=False):
        super(ModelNet40Dataset, self).__init__()

        if split not in ['train', 'test']:
            raise ValueError('Invalid split "{}"!'.format(split))

        self.data_root = root
        self.transform = transform
        self.use_normal = use_normal

        with open(osp.join(self.data_root, '{}_files.txt'.format(split))) as f:
            lines = f.readlines()
        data_files = [line.rstrip() for line in lines]

        points_list = []
        labels = []
        for data_file in data_files:
            h5file = h5py.File(osp.join(self.data_root, data_file), 'r')
            data = h5file['data'][:]
            label = h5file['label'][:]
            points_list.append(data)
            labels.append(label)
        self.points_list = np.concatenate(points_list)
        self.labels = np.concatenate(labels)
        if self.use_normal:
            if self.points_list.shape[1] != 6:
                raise ValueError('Dataset does not have normal features.')
            self.features_list = self.points_list[:, :, 3:]
        if self.points_list.shape[1] > 3:
            self.points_list = self.points_list[:, :, :3]
        self.length = self.points_list.shape[0]

    def __getitem__(self, index):
        if self.use_normal:
            points, features = self.transform(self.points_list[index], self.features_list[index])
        else:
            points = self.transform(self.points_list[index])
        label = int(self.labels[index])
        return points, label

    def __len__(self):
        return self.length
