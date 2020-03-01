import os.path as osp
import pickle

import torch.utils.data


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
    dataset = ShapeNetPartDataset(root, 'trainval', None)
    max_num_point = 0
    for points in dataset.points_list:
        num_point = points.shape[0]
        max_num_point = max(max_num_point, num_point)
    print(max_num_point)
