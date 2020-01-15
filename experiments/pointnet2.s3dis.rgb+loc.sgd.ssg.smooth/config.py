import os
import os.path as osp
import argparse

from easydict import EasyDict as edict

from vision3d.utils.python_utils import ensure_dir
from vision3d.datasets import S3DISDataset as dataset

config = edict()

# random seed
config.seed = 7351

# dir

config.root_dir = '/home/zheng/workspace/vision3d'
config.working_dir = osp.dirname(osp.realpath(__file__))
config.program_name = osp.basename(config.working_dir)
config.output_dir = osp.join(config.root_dir, 'output', config.program_name)
config.snapshot_dir = osp.join(config.output_dir, 'snapshots')
config.logs_dir = osp.join(config.output_dir, 'logs')
config.events_dir = osp.join(config.output_dir, 'events')
config.data_root = '/data/S3DIS/stanford_indoor3d'
config.testing_hdf5_root = '/data/S3DIS/s3dis_testing_all_hdf5'

ensure_dir(config.output_dir)
ensure_dir(config.snapshot_dir)
ensure_dir(config.logs_dir)
ensure_dir(config.events_dir)

# data
config.num_class = dataset.num_class
config.class_names = dataset.class_names

# train config
config.train_batch_size = 32
config.train_num_worker = 8
config.train_num_point = 4096
config.train_block_size = 1.0
config.train_normalized_location = True

# test config
config.test_batch_size = 32
config.test_num_worker = 8
config.test_num_point = 4096
config.test_block_size = 1.0
config.test_block_stride = 0.5
config.test_normalized_location = True

# optim config
config.learning_rate = 0.1
config.eta_min = config.learning_rate * 1e-2
config.momentum = 0.9
config.weight_decay = 1e-4
config.max_epoch = 250

# model
config.label_smoothing_eps = 0.1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--link_output', dest='link_output', action='store_true', help='link output dir')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.link_output:
        os.symlink(config.output_dir, 'output')
