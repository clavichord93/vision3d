import os.path as osp
import json
import pickle
import sys

from tqdm import tqdm
import numpy as np

from vision3d.datasets import ShapeNetPartDataset


def _get_class_id(synset):
    class_name = ShapeNetPartDataset.synset_to_class_name[synset]
    return ShapeNetPartDataset.class_names.index(class_name)


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
            data_dict[split] = {
                'points': points_list,
                'normals': normals_list,
                'labels': labels_list,
                'class_ids': class_ids
            }
        print('"{}" finished (max_num_point={}, min_num_point={})'.format(split, max_num_point, min_num_point))
    data_dict['trainval'] = {
        'points': data_dict['train']['points'] + data_dict['val']['points'],
        'normals': data_dict['train']['normals'] + data_dict['val']['normals'],
        'labels': data_dict['train']['labels'] + data_dict['val']['labels'],
        'class_ids': data_dict['train']['class_ids'] + data_dict['val']['class_ids']
    }

    for split in ['train', 'val', 'trainval', 'test']:
        dump_file = osp.join(root, '{}.pickle'.format(split))
        with open(dump_file, 'wb') as f:
            pickle.dump(data_dict[split], f)
            print('"{}" saved.'.format(dump_file))


if __name__ == '__main__':
    root = '/data/ShapeNetPart/shapenetcore_partanno_segmentation_benchmark_v0_normal'
    preprocess(root)
