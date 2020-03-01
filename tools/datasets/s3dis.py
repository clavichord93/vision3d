import os.path as osp

import h5py

from vision3d.datasets import S3DISWholeSceneDataset


def save_hdf5(data_root, scene_name, points, features, point_indices, labels):
    print('Saving {}...'.format(scene_name))
    h5file = h5py.File(osp.join(data_root, '..', 's3dis_testing_all_hdf5', scene_name + '.h5'))
    h5file.create_dataset('points', data=points, compression='gzip', dtype=points.dtype)
    h5file.create_dataset('features', data=features, compression='gzip', dtype=features.dtype)
    h5file.create_dataset('point_indices', data=point_indices, compression='gzip', dtype=point_indices.dtype)
    h5file.create_dataset('labels', data=labels, compression='gzip', dtype=labels.dtype)
    print('{} saved.'.format(scene_name))
    print(points.shape, features.shape, point_indices.shape, labels.shape)
    h5file.close()


if __name__ == '__main__':
    data_root = '/data/S3DIS/stanford_indoor3d'

    def transform(*inputs):
        return inputs

    for test_area in range(1, 7):
        dataset = S3DISWholeSceneDataset(data_root, transform, test_area=test_area)
        num_scene = len(dataset)
        for i in range(num_scene):
            batch_points_list, batch_features_list, batch_point_indices_list, labels = dataset[i]
            scene_name = dataset.scene_names[i]
            save_hdf5(data_root, scene_name, batch_points_list, batch_features_list, batch_point_indices_list, labels)
