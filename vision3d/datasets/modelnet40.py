import os
import os.path as osp
import glob
import random
import json

import h5py
import torch.utils.data
import numpy as np
from tqdm import tqdm


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


def preprocess(class_name_to_idx, data_root, data_files, split):
    random.shuffle(data_files)
    num_file = len(data_files)
    block_size = 2048
    num_block = (num_file + block_size - 1) // block_size
    for idx in range(num_block):
        start_idx = idx * block_size
        end_idx = min((idx + 1) * block_size, num_file)
        block_files = data_files[start_idx:end_idx]
        pbar = tqdm(block_files)
        data = []
        label = []
        for i, data_file in enumerate(pbar):
            class_name = data_file[:-5]
            data_file = osp.join(data_root, class_name, data_file + '.txt')
            data.append(np.loadtxt(data_file, delimiter=',', dtype=np.float))
            label.append(class_name_to_idx[class_name])
        data = np.stack(data, axis=0)
        label = np.array(label, dtype=np.uint8)
        print(data.shape)
        print(label.shape)
        h5file = osp.join(data_root, 'ply_data_normal_resampled_{}{}.h5'.format(split, idx))
        h5 = h5py.File(h5file)
        h5.create_dataset('data', data=data, compression='gzip', dtype=data.dtype)
        h5.create_dataset('label', data=label, compression='gzip', dtype=label.dtype)
        h5.close()
        id2file = osp.join(data_root, 'ply_data_normal_resampled_{}_{}_id2file.json'.format(split, idx))
        with open(id2file, 'w') as f:
            json.dump(block_files, f)
        print('{}_{} completed'.format(split, idx))


class ModelNet40Dataset(_ModelNet40DatasetBase):
    def __init__(self, root, split, transform):
        super(ModelNet40Dataset, self).__init__()
        if split not in ['train', 'test']:
            raise ValueError('Invalid split "{}"!'.format(split))
        data_files = glob.glob(osp.join(root, 'ply_data_normal_resampled_{}*.h5'.format(split)))
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
        points = self.transform(self.points[index, :, :3])
        label = int(self.labels[index])
        return points, label

    def __len__(self):
        return self.num_shape


if __name__ == '__main__':
    data_root = '/data/ModelNet40/modelnet40_normal_resampled'

    class_names = _ModelNet40DatasetBase.class_names
    class_name_to_idx = {class_name: i for i, class_name in enumerate(class_names)}

    with open(osp.join(data_root, 'modelnet40_train.txt')) as f:
        lines = f.readlines()
    train_files = [line.rstrip() for line in lines]
    preprocess(class_name_to_idx, data_root, train_files, 'train')

    with open(osp.join(data_root, 'modelnet40_test.txt')) as f:
        lines = f.readlines()
    test_files = [line.rstrip() for line in lines]
    preprocess(class_name_to_idx, data_root, test_files, 'test')
