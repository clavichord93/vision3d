import os.path as osp
import random
import json

import numpy as np
import h5py
from tqdm import tqdm

from vision3d.datasets import ModelNet40Dataset


def preprocess(class_name_to_idx, data_root, data_files, split):
    random.shuffle(data_files)
    num_file = len(data_files)
    block_size = 2048
    num_block = (num_file + block_size - 1) // block_size
    generated_files = []
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
        generated_files.append(h5file + '\n')
        h5 = h5py.File(h5file)
        h5.create_dataset('data', data=data, compression='gzip', dtype=data.dtype)
        h5.create_dataset('label', data=label, compression='gzip', dtype=label.dtype)
        h5.close()
        id2file = osp.join(data_root, 'ply_data_normal_resampled_{}_{}_id2file.json'.format(split, idx))
        with open(id2file, 'w') as f:
            json.dump(block_files, f)
        print('{}_{} completed'.format(split, idx))
    with open(osp.join(data_root, '{}_files.txt'.format(split)), 'w') as f:
        f.writelines(generated_files)


if __name__ == '__main__':
    data_root = '/data/ModelNet40/modelnet40_normal_resampled'

    class_names = ModelNet40Dataset.class_names
    class_name_to_idx = {class_name: i for i, class_name in enumerate(class_names)}

    with open(osp.join(data_root, 'modelnet40_train.txt')) as f:
        lines = f.readlines()
    train_files = [line.rstrip() for line in lines]
    preprocess(class_name_to_idx, data_root, train_files, 'train')

    with open(osp.join(data_root, 'modelnet40_test.txt')) as f:
        lines = f.readlines()
    test_files = [line.rstrip() for line in lines]
    preprocess(class_name_to_idx, data_root, test_files, 'test')
