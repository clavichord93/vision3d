import os
import os.path as osp

import numpy as np
import torch
import torch.utils.data


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


class S3DISDatasetV3(_S3DISDatasetBase):
    def __init__(self,
                 root,
                 transform,
                 test_area=5,
                 num_sample=4096,
                 block_size=1.0,
                 training=True,
                 normalized_location=True):
        super(S3DISDatasetV3, self).__init__()

        self.test_area = 'Area_{}'.format(test_area)
        if self.test_area not in self.areas:
            raise ValueError('Invalid test_area: {}.'.format(self.test_area))

        self.num_sample = num_sample
        self.block_size = block_size
        self.transform = transform
        self.training = training
        self.normalized_location = normalized_location

        with open(osp.join(root, 'scene_names.txt')) as f:
            lines = f.readlines()
        self.scene_names = [line.rstrip() for line in lines]
        if self.training:
            data_files = [scene_name + '.npy' for scene_name in self.scene_names if self.test_area not in scene_name]
        else:
            data_files = [scene_name + '.npy' for scene_name in self.scene_names if self.test_area in scene_name]
        data_files = [osp.join(root, data_file) for data_file in data_files]
        self.num_scene = len(data_files)

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
        points[:, :2] = points[:, :2] - center
        labels = labels[indices]

        points, features, labels = self.transform(points, features, labels)

        return points, features, labels

    def __len__(self):
        return self.length


class S3DISWholeSceneDataset(_S3DISDatasetBase):
    def __init__(self,
                 root,
                 transform,
                 test_area=5,
                 num_sample=4096,
                 block_size=1.0,
                 stride=0.5,
                 normalized_location=True):
        super(S3DISWholeSceneDataset, self).__init__()
        self.transform = transform
        self.test_area = 'Area_{}'.format(test_area)
        self.num_sample = num_sample
        self.block_size = block_size
        self.stride = stride
        self.normalized_location = normalized_location

        with open(osp.join(root, 'scene_names.txt')) as f:
            lines = f.readlines()
        self.scene_names = [line.rstrip() for line in lines]
        data_files = [
            osp.join(root, scene_name + '.npy') for scene_name in self.scene_names if self.test_area in scene_name
        ]
        self.num_scene = len(data_files)

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
        num_block = barycenters
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

        num_block_x = int(np.ceil((scene_size[0] - self.block_size) / self.stride)) + 1
        num_block_y = int(np.ceil((scene_size[1] - self.block_size) / self.stride)) + 1

        points_list = []
        features_list = []
        labels_list = []
        point_indices_list = []
        barycenters = []
        for i in range(num_block_x):
            for j in range(num_block_y):
                lower_bound = np.array([i * self.block_size, j * self.block_size, 0])
                upper_bound = np.array([(i + 1) * self.block_size, (j + 1) * self.block_size, scene_size[2]])
                point_indices = np.nonzero(np.logical_and(
                    np.logical_and(points[:, 0] >= lower_bound[0], points[:, 0] <= upper_bound[0]),
                    np.logical_and(points[:, 1] >= lower_bound[1], points[:, 1] <= upper_bound[1])
                ))[0]
                if point_indices.shape[0] == 0:
                    continue
                points_list.append(points[point_indices])
                features_list.append(features[point_indices])
                labels_list.append(labels[point_indices])
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
            labels = labels_list.pop(block_index)
            point_indices = point_indices_list.pop(block_index)
            barycenter = barycenters.pop(block_index)
            nearest_block_index = self._find_nearest_block(points, barycenter, points_list, barycenters)
            merged_points = np.concatenate([points_list[nearest_block_index], points], axis=0)
            merged_features = np.concatenate([features_list[nearest_block_index], features], axis=0)
            merged_labels = np.concatenate([labels_list[nearest_block_index], labels], axis=0)
            merged_point_indices = np.concatenate([point_indices_list[nearest_block_index], point_indices], axis=0)
            points_list[nearest_block_index] = merged_points
            features_list[nearest_block_index] = merged_features
            labels_list[nearest_block_index] = merged_labels
            point_indices_list[nearest_block_index] = merged_point_indices
            barycenters[nearest_block_index] = np.mean(merged_points[:, :2], axis=0)
            num_block -= 1

        # divide large blocks
        num_block = len(points_list)
        batch_points_list = []
        batch_features_list = []
        batch_labels_list = []
        batch_point_indices_list = []
        batch_barycenters = []
        for i in range(num_block):
            points = points_list[i]
            features = features_list[i]
            labels = labels_list[i]
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
                batch_labels = labels[batch_indices]
                batch_point_indices = point_indices[batch_indices]
                batch_barycenter = np.mean(batch_points[:, :2], axis=0)
                batch_points[:, :2] -= batch_barycenter
                batch_points_list.append(batch_points)
                batch_features_list.append(batch_features)
                batch_labels_list.append(batch_labels)
                batch_point_indices_list.append(batch_point_indices)
                batch_barycenters.append(batch_barycenters)

        batch_points_list = np.stack(batch_points_list, axis=0)
        batch_features_list = np.stack(batch_features_list, axis=0)
        batch_labels_list = np.stack(batch_labels_list, axis=0)
        batch_point_indices_list = np.stack(batch_point_indices_list, axis=0)
        batch_barycenters = np.stack(batch_barycenters, axis=0)

        return batch_points_list, batch_features_list, batch_labels_list, batch_point_indices_list, batch_barycenters

    def __len__(self):
        return self.length
