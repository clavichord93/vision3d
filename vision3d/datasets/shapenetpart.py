import os.path as osp
import json
import pickle
import sys

from tqdm import tqdm
import torch.utils.data
import numpy as np

from vision3d.utils.python_utils import ensure_dir


class _ShapeNetPartDatasetBase(torch.utils.data.Dataset):
    num_class = 16
    num_part = 50
    class_names = [
        'Airplane', 'Bag', 'Cap', 'Car',
        'Chair', 'Earphone', 'Guitar', 'Knife',
        'Lamp', 'Laptop', 'Motorbike', 'Mug',
        'Pistol', 'Rocket', 'Skateboard', 'Table'
    ]
    part_names = [
        'Airplane0', 'Airplane1', 'Airplane2', 'Airplane3',
        'Bag0', 'Bag1',
        'Cap0', 'Cap1',
        'Car0', 'Car1', 'Car2', 'Car3',
        'Chair0', 'Chair1', 'Chair2', 'Chair3',
        'Earphone0', 'Earphone1', 'Earphone2',
        'Guitar0', 'Guitar1', 'Guitar2',
        'Knife0', 'Knife1',
        'Lamp0', 'Lamp1', 'Lamp2', 'Lamp3',
        'Laptop0', 'Laptop1',
        'Motorbike0', 'Motorbike1', 'Motorbike2', 'Motorbike3', 'Motorbike4', 'Motorbike5',
        'Mug0', 'Mug1',
        'Pistol0', 'Pistol1', 'Pistol2',
        'Rocket0', 'Rocket1', 'Rocket2',
        'Skateboard0', 'Skateboard1', 'Skateboard2',
        'Table0', 'Table1', 'Table2',
    ]
    class_id_to_part_ids = [
        [0, 1, 2, 3],
        [4, 5],
        [6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
        [16, 17, 18],
        [19, 20, 21],
        [22, 23],
        [24, 25, 26, 27],
        [28, 29],
        [30, 31, 32, 33, 34, 35],
        [36, 37],
        [38, 39, 40],
        [41, 42, 43],
        [44, 45, 46],
        [47, 48, 49]
    ]
    class_name_to_part_ids = {
        'Airplane': [0, 1, 2, 3],
        'Bag': [4, 5],
        'Cap': [6, 7],
        'Car': [8, 9, 10, 11],
        'Chair': [12, 13, 14, 15],
        'Earphone': [16, 17, 18],
        'Guitar': [19, 20, 21],
        'Knife': [22, 23],
        'Lamp': [24, 25, 26, 27],
        'Laptop': [28, 29],
        'Motorbike': [30, 31, 32, 33, 34, 35],
        'Mug': [36, 37],
        'Pistol': [38, 39, 40],
        'Rocket': [41, 42, 43],
        'Skateboard': [44, 45, 46],
        'Table': [47, 48, 49]
    }
    part_id_to_class_id = [
        0, 0, 0, 0, 1, 1, 2, 2, 3, 3,
        3, 3, 4, 4, 4, 4, 5, 5, 5, 6,
        6, 6, 7, 7, 8, 8, 8, 8, 9, 9,
        10, 10, 10, 10, 10, 10, 11, 11, 12, 12,
        12, 13, 13, 13, 14, 14, 14, 15, 15, 15
    ]
    synset_to_class_name = {
        '02691156': 'Airplane',
        '02773838': 'Bag',
        '02954340': 'Cap',
        '02958343': 'Car',
        '03001627': 'Chair',
        '03261776': 'Earphone',
        '03467517': 'Guitar',
        '03624134': 'Knife',
        '03636649': 'Lamp',
        '03642806': 'Laptop',
        '03790512': 'Motorbike',
        '03797390': 'Mug',
        '03948459': 'Pistol',
        '04099429': 'Rocket',
        '04225987': 'Skateboard',
        '04379243': 'Table',
    }


def _get_part_id(synset, index):
    class_name = _ShapeNetPartDatasetBase.synset_to_class_name[synset]
    return _ShapeNetPartDatasetBase.class_name_to_part_ids[class_name][index - 1]


def _get_class_id(synset):
    class_name = _ShapeNetPartDatasetBase.synset_to_class_name[synset]
    return _ShapeNetPartDatasetBase.class_names.index(class_name)


def preprocess(root):
    r"""
    Preprocess the dataset with normals and dump to pickle files for faster loading.

    :param root: str
        The root path of ShapeNetPart dataset.
    """
    data_dict = {}
    for split in ['train', 'val', 'test']:
        print('Processing "{}" split...'.format(split))
        data_split_file = osp.join(root, 'train_test_split', 'shuffled_{}_file_list.json'.format(split))
        max_num_point = 0
        min_num_point = sys.maxsize
        with open(data_split_file) as f:
            points_list = []
            normals_list = []
            labels_list = []
            class_ids = []
            shapenet_paths = json.load(f)
            pbar = tqdm(shapenet_paths)
            for shapenet_path in pbar:
                synset, shape_id = shapenet_path.split('/')[1:]
                data_file = osp.join(root, synset, '{}.txt'.format(shape_id))
                with open(data_file) as f:
                    lines = f.readlines()
                    split_lines = [line.strip().split() for line in lines]

                    points = [[float(x) for x in split_line[:3]] for split_line in split_lines]
                    points = np.array(points, dtype=np.float)
                    points_list.append(points)

                    normals = [[float(x) for x in split_line[3:-1]] for split_line in split_lines]
                    normals_list.append(np.array(normals, dtype=np.float))

                    labels = [int(split_line[-1].split('.')[0]) for split_line in split_lines]
                    labels_list.append(np.array(labels, dtype=np.int))

                    class_id = _get_class_id(synset)
                    class_ids.append(class_id)
                num_point = len(points)
                max_num_point = max(max_num_point, num_point)
                min_num_point = min(min_num_point, num_point)
                message = 'num_point: {}'.format(num_point)
                pbar.set_description(message)
            data_dict[split] = {'points': points_list,
                                'normals': normals_list,
                                'labels': labels_list,
                                'class_ids': class_ids}
        print('"{}" finished (max_num_point={}, min_num_point={})'.format(split, max_num_point, min_num_point))
    data_dict['trainval'] = {'points': data_dict['train']['points'] + data_dict['val']['points'],
                             'normals': data_dict['train']['normals'] + data_dict['val']['normals'],
                             'labels': data_dict['train']['labels'] + data_dict['val']['labels'],
                             'class_ids': data_dict['train']['class_ids'] + data_dict['val']['class_ids']}

    for split in ['train', 'val', 'trainval', 'test']:
        dump_file = osp.join(root, '{}.pickle'.format(split))
        with open(dump_file, 'wb') as f:
            pickle.dump(data_dict[split], f)
            print('"{}" saved.'.format(dump_file))


class ShapeNetPartDataset(_ShapeNetPartDatasetBase):
    def __init__(self, root, split, transform, use_normal=True):
        if split not in ['train', 'val', 'trainval', 'test']:
            raise ValueError('Invalid split "{}"!'.format(split))
        super(ShapeNetPartDataset, self).__init__()

        dump_file = osp.join(root, '{}.pickle'.format(split))
        with open(dump_file, 'rb') as f:
            data_dict = pickle.load(f)

        self.transform = transform
        self.use_normal = use_normal

        self.points_list = data_dict['points']
        if self.use_normal:
            self.normals_list = data_dict['normals']
        self.labels_list = data_dict['labels']
        self.class_ids = data_dict['class_ids']
        self.length = len(self.points_list)

    def __getitem__(self, index):
        points = self.points_list[index]
        labels = self.labels_list[index]
        class_id = self.class_ids[index]
        if self.use_normal:
            points, normals, labels = self.transform(points, self.normals_list[index], labels)
            return points, normals, labels, class_id
        else:
            points, labels = self.transform(points, labels)
            return points, labels, class_id

    def __len__(self):
        return self.length


if __name__ == '__main__':
    root = '/data/ShapeNetPart/shapenetcore_partanno_segmentation_benchmark_v0_normal'
    preprocess(root)

    dataset = ShapeNetPartDataset(root, 'trainval', None)
    max_num_point = 0
    for points in dataset.points_list:
        num_point = points.shape[0]
        max_num_point = max(max_num_point, num_point)
    print(max_num_point)
