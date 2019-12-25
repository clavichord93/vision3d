import os.path as osp
import glob

import h5py
import torch.utils.data
import numpy as np


class ModelNet40Dataset(torch.utils.data.Dataset):
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

    def __init__(self, root, phase, transform):
        data_files = glob.glob(osp.join(root, 'ply_data_{}*.h5'.format(phase)))
        all_points = []
        all_labels = []
        for data_file in data_files:
            h5file = h5py.File(data_file, 'r')
            points = h5file['data'][:]
            labels = h5file['label'][:]
            all_points.append(points)
            all_labels.append(labels)
        self.points = np.concatenate(all_points)
        self.labels = np.concatenate(all_labels)
        self.num_shape = self.points.shape[0]
        self.transform = transform

    def __getitem__(self, index):
        points = self.transform(self.points[index])
        label = int(self.labels[index, 0])
        return points, label

    def __len__(self):
        return self.num_shape
