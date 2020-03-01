import os
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


def _load_scene(data_file):
    scene = np.load(data_file)
    points = scene[:, :3]
    features = scene[:, 3:6] / 255.
    min_point = np.amin(points, axis=0)
    max_point = np.amax(points, axis=0)
    points -= min_point
    scene_size = max_point - min_point
    labels = scene[:, 6].astype(np.int8)
    return points, features, labels, scene_size


def _random_sample_indices(indices, num_sample):
    num_index = indices.shape[0]
    if num_index >= num_sample:
        indices = np.random.choice(indices, num_sample, replace=False)
    else:
        extra_indices = np.random.choice(indices, num_sample - num_index, replace=True)
        indices = np.concatenate([indices, extra_indices], axis=0)
        np.random.shuffle(indices)
    return indices


class S3DISDataset(_S3DISDatasetBase):
    r"""
    S3SIDDataset uses pre-computed npy files from the official PointNet implementation and conducts data augmentation
    on the fly. Read [here](https://github.com/charlesq34/pointnet/blob/master/sem_seg/README.md) for more details.

    The code is modified from [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch).
    """
    def __init__(self,
                 root,
                 transform,
                 test_area=5,
                 num_sample=4096,
                 block_size=1.0,
                 training=True,
                 normalized_location=True):
        super(S3DISDataset, self).__init__()

        self.test_area = 'Area_{}'.format(test_area)
        if self.test_area not in self.areas:
            raise ValueError('Invalid test_area: {}.'.format(self.test_area))

        self.data_root = root
        self.num_sample = num_sample
        self.block_size = block_size
        self.transform = transform
        self.training = training
        self.normalized_location = normalized_location

        with open(osp.join(self.data_root, 'scene_names.txt')) as f:
            lines = f.readlines()
        if self.training:
            self.scene_names = [line.rstrip() for line in lines if self.test_area not in line]
        else:
            self.scene_names = [line.rstrip() for line in lines if self.test_area not in line]
        self.num_scene = len(self.scene_names)
        data_files = [osp.join(self.data_root, scene_name + '.npy') for scene_name in self.scene_names]

        self.points_list = []
        self.features_list = []
        self.labels_list = []
        self.scene_sizes = []
        # num_points = []
        for data_file in data_files:
            points, features, labels, scene_size = _load_scene(data_file)
            self.points_list.append(points)
            self.features_list.append(features)
            self.labels_list.append(labels)
            self.scene_sizes.append(scene_size)
            # num_points.append(points.shape[0])

        self.scene_indices = []
        for i in range(self.num_scene):
            num_block_x = np.prod(np.ceil(self.scene_sizes[i][0:2] / self.block_size)).astype(int)
            self.scene_indices += [i] * num_block_x
        self.length = len(self.scene_indices)

        # total_num_point = np.sum(num_points)
        # sample_probs = np.array(num_points, dtype=np.float) / total_num_point
        # num_iteration = int(total_num_point / num_sample)
        # self.scene_indices = []
        # for i in range(self.num_scene):
        #     self.scene_indices += [i] * int(np.ceil(sample_probs[i] * num_iteration))
        # self.length = len(self.scene_indices)

    def __getitem__(self, index):
        index = self.scene_indices[index]
        points = self.points_list[index]
        features = self.features_list[index]
        labels = self.labels_list[index]
        scene_size = self.scene_sizes[index]

        while True:
            center = np.random.rand(2) * scene_size[:2]
            lower_bound = center - self.block_size / 2.
            upper_bound = center + self.block_size / 2.
            indices = np.nonzero(np.logical_and(
                np.logical_and(points[:, 0] >= lower_bound[0], points[:, 0] <= upper_bound[0]),
                np.logical_and(points[:, 1] >= lower_bound[1], points[:, 1] <= upper_bound[1])
            ))[0]
            if indices.shape[0] > 1024:
                break

        indices = _random_sample_indices(indices, self.num_sample)
        points = points[indices]
        features = features[indices]
        if self.normalized_location:
            normalized_locations = points / scene_size
            features = np.concatenate([features, normalized_locations], axis=1)
        barycenter = np.mean(points[:, :2], axis=0)
        points[:, :2] = points[:, :2] - barycenter
        labels = labels[indices]

        points, features, labels = self.transform(points, features, labels)

        return points, features, labels

    def __len__(self):
        return self.length


class S3DISHdf5Dataset(_S3DISDatasetBase):
    r"""
    S3SIDHdf5Dataset uses pre-computed hdf5 files from the official PointNet implementation.
    Download from [here](https://shapenet.cs.stanford.edu/media/indoor3d_sem_seg_hdf5_data.zip).
    """
    def __init__(self, root, transform, test_area, training=True, normalized_location=True):
        super(S3DISHdf5Dataset, self).__init__()

        self.test_area = 'Area_{}'.format(test_area)
        if self.test_area not in self.areas:
            raise ValueError('Invalid test_area: {}.'.format(self.test_area))

        self.data_root = root
        self.transform = transform
        self.training = training
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


class S3DISWholeSceneDataset(_S3DISDatasetBase):
    r"""
    S3SIDWholeSceneDataset uses pre-computed npy files from the official PointNet implementation. All points in a scene
    are used for evaluation. Small blocks are merged with the nearest block and Large blocks are divided into batches.

    The code is modified from [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch).
    """
    def __init__(self,
                 root,
                 transform,
                 test_area=5,
                 num_sample=4096,
                 block_size=1.0,
                 block_stride=0.5,
                 normalized_location=True):
        super(S3DISWholeSceneDataset, self).__init__()

        self.test_area = 'Area_{}'.format(test_area)
        if self.test_area not in self.areas:
            raise ValueError('Invalid test_area: {}.'.format(self.test_area))

        self.data_root = root
        self.transform = transform
        self.num_sample = num_sample
        self.block_size = block_size
        self.block_stride = block_stride
        self.normalized_location = normalized_location

        with open(osp.join(self.data_root, 'scene_names.txt')) as f:
            lines = f.readlines()
        self.scene_names = [line.rstrip() for line in lines if self.test_area in line]
        self.num_scene = len(self.scene_names)
        data_files = [osp.join(self.data_root, scene_name + '.npy') for scene_name in self.scene_names]

        self.points_list = []
        self.features_list = []
        self.labels_list = []
        self.scene_sizes = []
        for data_file in data_files:
            points, features, labels, scene_size = _load_scene(data_file)
            self.points_list.append(points)
            self.features_list.append(features)
            self.labels_list.append(labels)
            self.scene_sizes.append(scene_size)
        self.length = len(self.points_list)

    def _find_nearest_block(self, points, barycenter, points_list, barycenters):
        num_block = len(points_list)
        best_dist2 = np.inf
        best_dist2_index = -1
        for i in range(num_block):
            if points_list[i].shape[0] + points.shape[0] > self.num_sample / 2:
                dist2 = np.sum((barycenters[i] - barycenter) ** 2)
                if dist2 < best_dist2:
                    best_dist2 = dist2
                    best_dist2_index = i
        return best_dist2_index

    def __getitem__(self, index):
        points = self.points_list[index]
        features = self.features_list[index]
        labels = self.labels_list[index]
        scene_size = self.scene_sizes[index]

        num_block_x = int(np.ceil((scene_size[0] - self.block_size) / self.block_stride)) + 1
        num_block_y = int(np.ceil((scene_size[1] - self.block_size) / self.block_stride)) + 1

        points_list = []
        features_list = []
        point_indices_list = []
        barycenters = []
        for i in range(num_block_x):
            for j in range(num_block_y):
                lower_bound = np.array([i, j]) * self.block_size
                upper_bound = np.array([(i + 1), (j + 1)]) * self.block_size
                point_indices = np.nonzero(np.logical_and(
                    np.logical_and(points[:, 0] >= lower_bound[0], points[:, 0] <= upper_bound[0]),
                    np.logical_and(points[:, 1] >= lower_bound[1], points[:, 1] <= upper_bound[1])
                ))[0]
                if point_indices.shape[0] == 0:
                    continue
                points_list.append(points[point_indices])
                features_list.append(features[point_indices])
                point_indices_list.append(point_indices)
                barycenters.append(np.mean(points[:, :2], axis=0))

        # merge small blocks
        num_block = len(points_list)
        block_index = 0
        while block_index < num_block:
            if points_list[block_index].shape[0] > self.num_sample / 2:
                block_index += 1
                continue
            points = points_list.pop(block_index)
            features = features_list.pop(block_index)
            point_indices = point_indices_list.pop(block_index)
            barycenter = barycenters.pop(block_index)
            nearest_block_index = self._find_nearest_block(points, barycenter, points_list, barycenters)
            merged_points = np.concatenate([points_list[nearest_block_index], points], axis=0)
            merged_features = np.concatenate([features_list[nearest_block_index], features], axis=0)
            merged_point_indices = np.concatenate([point_indices_list[nearest_block_index], point_indices], axis=0)
            points_list[nearest_block_index] = merged_points
            features_list[nearest_block_index] = merged_features
            point_indices_list[nearest_block_index] = merged_point_indices
            barycenters[nearest_block_index] = np.mean(merged_points[:, :2], axis=0)
            num_block -= 1

        # divide large blocks
        num_block = len(points_list)
        batch_points_list = []
        batch_features_list = []
        batch_point_indices_list = []
        for i in range(num_block):
            points = points_list[i]
            features = features_list[i]
            point_indices = point_indices_list[i]

            num_point = points.shape[0]
            indices = np.arange(num_point)
            if num_point % self.num_sample != 0:
                num_supplementary_point = self.num_sample - num_point % self.num_sample
                supplementary_indices = np.random.choice(indices, num_supplementary_point, replace=True)
                indices = np.concatenate([indices, supplementary_indices], axis=0)
            np.random.shuffle(indices)

            num_point = indices.shape[0]
            batch_indices_list = np.split(indices, num_point / self.num_sample, axis=0)

            for batch_indices in batch_indices_list:
                batch_points = points[batch_indices]
                batch_features = features[batch_indices]
                if self.normalized_location:
                    normalized_locations = batch_points / scene_size
                    batch_features = np.concatenate([batch_features, normalized_locations], axis=1)
                batch_point_indices = point_indices[batch_indices]
                batch_barycenter = np.mean(batch_points[:, :2], axis=0)
                batch_points[:, :2] -= batch_barycenter
                batch_points_list.append(batch_points)
                batch_features_list.append(batch_features)
                batch_point_indices_list.append(batch_point_indices)

        batch_points_list = np.stack(batch_points_list, axis=0)
        batch_features_list = np.stack(batch_features_list, axis=0)
        batch_point_indices_list = np.stack(batch_point_indices_list, axis=0)

        batch_points_list, batch_features_list, batch_point_indices_list, labels = \
            self.transform(batch_points_list, batch_features_list, batch_point_indices_list, labels)

        return batch_points_list, batch_features_list, batch_point_indices_list, labels

    def __len__(self):
        return self.length


class S3DISWholeSceneHdf5Dataset(_S3DISDatasetBase):
    r"""
    S3SIDWholeSceneHdf5Dataset uses pre-computed hdf5 files generated by S3DISWholeSceneDataset. All points in a scene
    are used for evaluation. Small blocks are merged with the nearest block and Large blocks are divided into batches.
    """
    def __init__(self, root, transform, test_area=5, normalized_location=True):
        super(S3DISWholeSceneHdf5Dataset, self).__init__()

        self.test_area = 'Area_{}'.format(test_area)
        if self.test_area not in self.areas:
            raise ValueError('Invalid test_area: {}.'.format(self.test_area))

        self.data_root = root
        self.transform = transform
        self.normalized_location = normalized_location

        with open(osp.join(self.data_root, 'scene_names.txt')) as f:
            lines = f.readlines()
        self.scene_names = [line.rstrip() for line in lines if self.test_area in line]
        data_files = [osp.join(self.data_root, scene_name + '.h5') for scene_name in self.scene_names]
        self.num_scene = len(self.scene_names)

        self.batch_points_lists = []
        self.batch_features_lists = []
        self.batch_point_indices_lists = []
        self.labels_list = []
        for data_file in data_files:
            h5file = h5py.File(data_file)
            self.batch_points_lists.append(h5file['points'][:])
            self.batch_features_lists.append(h5file['features'][:])
            self.batch_point_indices_lists.append(h5file['point_indices'][:])
            self.labels_list.append(h5file['labels'][:])

    def __getitem__(self, index):
        batch_points_list = self.batch_points_lists[index]
        batch_features_list = self.batch_features_lists[index]
        batch_point_indices_list = self.batch_point_indices_lists[index]
        labels = self.labels_list[index]

        batch_points_list, batch_features_list, batch_point_indices_list, labels = \
            self.transform(batch_points_list, batch_features_list, batch_point_indices_list, labels)

        return batch_points_list, batch_features_list, batch_point_indices_list, labels

    def __len__(self):
        return self.num_scene
