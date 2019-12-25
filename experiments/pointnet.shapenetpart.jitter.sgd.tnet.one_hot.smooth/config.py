import os
import os.path as osp
import argparse

from easydict import EasyDict as edict

from vision3d.utils.python_utils import ensure_dir
from vision3d.datasets.shapenetpart import ShapeNetPartNormalDataset as dataset

config = edict()

# random seed
config.seed = 7351

# dir
config.PATH = edict()
config.PATH.root_dir = '/home/zheng/workspace/vision3d'
config.PATH.working_dir = osp.dirname(osp.realpath(__file__))
config.PATH.program_name = osp.basename(config.PATH.working_dir)
config.PATH.output_dir = osp.join(config.PATH.root_dir, 'output', config.PATH.program_name)
config.PATH.snapshot_dir = osp.join(config.PATH.output_dir, 'snapshots')
config.PATH.logs_dir = osp.join(config.PATH.output_dir, 'logs')
config.PATH.events_dir = osp.join(config.PATH.output_dir, 'events')
config.PATH.data_root = '/data/ShapeNetPart/shapenetcore_partanno_segmentation_benchmark_v0_normal'

ensure_dir(config.PATH.output_dir)
ensure_dir(config.PATH.snapshot_dir)
ensure_dir(config.PATH.logs_dir)
ensure_dir(config.PATH.events_dir)

# data
config.DATA = edict()
config.DATA.num_class = dataset.num_class
config.DATA.class_names = dataset.class_names
config.DATA.num_part = dataset.num_part
config.DATA.part_names = dataset.part_names
config.DATA.class_id_to_part_ids = dataset.class_id_to_part_ids
config.DATA.part_id_to_class_id = dataset.part_id_to_class_id

config.TRAIN = edict()
config.TRAIN.num_point = 2048
config.TRAIN.sigma = 0.01
config.TRAIN.batch_size = 32
config.TRAIN.num_worker = 8

config.TEST = edict()
config.TEST.batch_size = 1
config.TEST.num_worker = 8

# train
config.OPTIMIZER = edict()
config.OPTIMIZER.learning_rate = 0.1
config.OPTIMIZER.eta_min = 0.001
config.OPTIMIZER.momentum = 0.9
config.OPTIMIZER.weight_decay = 1e-4
config.OPTIMIZER.max_epoch = 250

# model
config.MODEL = edict()
config.MODEL.alpha = 0.001
config.MODEL.eps = 0.2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--link_output', dest='link_output', action='store_true', help='link output dir')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.link_output:
        os.symlink(config.PATH.output_dir, 'output')
