import os
import os.path as osp
import argparse

from easydict import EasyDict as edict

from vision3d.utils.python_utils import ensure_dir
from vision3d.datasets import ModelNet40Dataset as dataset

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
config.data_root = '/data/ModelNet40/modelnet40_ply_hdf5_2048'

ensure_dir(config.output_dir)
ensure_dir(config.snapshot_dir)
ensure_dir(config.logs_dir)
ensure_dir(config.events_dir)

# data
config.num_class = dataset.num_class
config.class_names = dataset.class_names

# train config
config.train_num_point = 1024
config.train_jitter_sigma = 0.01
config.train_rescale_low = 0.8
config.train_rescale_high = 1.2
config.train_batch_size = 32
config.train_num_worker = 8

# test config
config.test_num_point = 1024
config.test_batch_size = 32
config.test_num_worker = 8

# optim config
config.learning_rate = 0.1
config.eta_min = 1e-3
config.momentum = 0.9
config.weight_decay = 1e-4
config.max_epoch = 250

# model
config.tnet_loss_alpha = 0.001
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
