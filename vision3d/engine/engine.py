from collections import OrderedDict
import sys
import os
import argparse
import random

import torch
import torch.backends.cudnn as cudnn
import numpy as np

from .logger import create_logger


class State(object):
    def __init__(self):
        self.epoch = -1
        self.iteration = -1
        self.model = None
        self.optimizer = None

    def register(self, **kwargs):
        for key, value in kwargs.items():
            if key not in ['epoch', 'iteration', 'model', 'optimizer']:
                raise ValueError('State does not have a member named "{}".'.format(key))
            setattr(self, key, value)


class Engine(object):
    def __init__(self, log_file=None, default_parser=None, seed=None, cudnn_deterministic=False):
        # parser, logger, state
        self.parser = default_parser
        self.inject_default_parser()
        self.args = self.parser.parse_args()
        self.devices = self.args.devices
        self.snapshot = self.args.snapshot
        self.logger = create_logger(log_file=log_file)
        self.state = State()

        message = 'Command executed: {}'.format(' '.join(sys.argv))
        self.logger.info(message)

        # cuda
        os.environ['CUDA_VISIBLE_DEVICES'] = self.devices
        if not torch.cuda.is_available():
            raise RuntimeError('No CUDA devices available.')
        self.num_device = torch.cuda.device_count()
        self.data_parallel = self.num_device > 1
        if self.data_parallel:
            self.logger.info('Using DataParallel mode ({} GPUs available).'.format(self.num_device))
        else:
            self.logger.info('Using Single-GPU mode.')
        self.cudnn_deterministic = cudnn_deterministic

        # random seed & deterministic
        self.seed = seed
        self.initialize()

    def inject_default_parser(self):
        if self.parser is None:
            self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--devices', metavar='GPUs', required=True, help='devices to use')
        self.parser.add_argument('--snapshot', metavar='F', default=None, help='load from snapshot')

    def initialize(self):
        if self.seed is not None:
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
        if self.cudnn_deterministic:
            cudnn.benchmark = False
            cudnn.deterministic = True
        else:
            cudnn.benchmark = True

    def register_state(self, **kwargs):
        self.state.register(**kwargs)

    def save_snapshot(self, file_path):
        model_state_dict = self.state.model.state_dict()
        if self.data_parallel:
            model_state_dict = OrderedDict([(key[7:], value) for key, value in model_state_dict.items()])

        state_dict = {
            'epoch': self.state.epoch,
            'iteration': self.state.iteration,
            'model': model_state_dict,
            'optimizer': self.state.optimizer.state_dict()
        }
        torch.save(state_dict, file_path)
        self.logger.info('Snapshot saved to "{}"'.format(file_path))

    def load_snapshot(self, snapshot):
        state_dict = torch.load(snapshot, map_location=torch.device('cpu'))

        self.logger.info('Loading from "{}".'.format(snapshot))

        if 'epoch' in state_dict:
            epoch = state_dict['epoch']
            self.state.epoch = state_dict['epoch']
            self.logger.info('Epoch has been loaded: {}.'.format(epoch))

        if 'iteration' in state_dict:
            iteration = state_dict['iteration']
            self.state.iteration = state_dict['iteration']
            self.logger.info('Iteration has been loaded: {}.'.format(iteration))

        if 'model' in state_dict:
            self._load_model(state_dict['model'])
            self.logger.info('Model has been loaded.')

        if 'optimizer' in state_dict and self.state.optimizer is not None:
            self.state.optimizer.load_state_dict(state_dict['optimizer'])
            self.logger.info('Optimizer has been loaded.')

        self.logger.info('Snapshot loaded.')

    def _load_model(self, state_dict):
        if self.data_parallel:
            state_dict = OrderedDict([('module.' + key, value) for key, value in state_dict.items()])
        self.state.model.load_state_dict(state_dict, strict=False)

        snapshot_keys = set(state_dict.keys())
        model_keys = set(self.state.model.state_dict().keys())
        missing_keys = model_keys - snapshot_keys
        unexpected_keys = snapshot_keys - model_keys
        if self.data_parallel:
            missing_keys = set([missing_key[7:] for missing_key in missing_keys])
            unexpected_keys = set([unexpected_key[7:] for unexpected_key in unexpected_keys])

        if len(missing_keys) > 0:
            message = 'Missing keys: {}'.format(missing_keys)
            self.logger.warning(message)
        if len(unexpected_keys) > 0:
            message = 'Unexpected keys: {}'.format(unexpected_keys)
            self.logger.warning(message)

    def step(self):
        self.state.iteration += 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        torch.cuda.empty_cache()
        if exc_type is not None:
            message = 'Error: {}'.format(exc_value)
            self.logger.error(message)
