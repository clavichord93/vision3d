import os
import argparse
import logging
import random

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np

from vision3d.engine.logger import create_logger


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
    def __init__(self, log_file=None, default_parser=None, seed=None):
        self.parser = default_parser
        self.inject_default_parser()
        self.args = self.parser.parse_args()
        self.devices = self.args.devices
        self.snapshot = self.args.snapshot

        os.environ['CUDA_VISIBLE_DEVICES'] = self.devices

        if not torch.cuda.is_available():
            raise RuntimeError('No CUDA devices available.')
        self.num_device = torch.cuda.device_count()
        self.data_parallel = self.num_device > 1

        self.state = State()
        self.logger = create_logger(output_file=log_file)

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
        cudnn.benchmark = True
        # cudnn.benchmark = False
        # cudnn.deterministic = True

    def register_state(self, **kwargs):
        self.state.register(**kwargs)

    def get_cuda_model(self):
        model = self.state.model
        if self.data_parallel:
            self.log('Using DataParallel mode ({} GPUs available).'.format(self.num_device))
            model = nn.DataParallel(model).cuda()
        else:
            self.log('Using Single-GPU mode.')
            model = model.cuda()
        return model

    def save_snapshot(self, file_path):
        state_dict = {
            'epoch': self.state.epoch,
            'iteration': self.state.iteration,
            'model': self.state.model.state_dict(),
            'optimizer': self.state.optimizer.state_dict()
        }
        torch.save(state_dict, file_path)
        self.log('Snapshot saved to "{}"'.format(file_path))

    def load_snapshot(self, snapshot):
        state_dict = torch.load(snapshot)

        self.log('Loading from "{}".'.format(snapshot))

        if 'epoch' in state_dict:
            epoch = state_dict['epoch']
            self.state.epoch = state_dict['epoch']
            self.log('Epoch has been loaded: {}.'.format(epoch))

        if 'iteration' in state_dict:
            iteration = state_dict['iteration']
            self.state.iteration = state_dict['iteration']
            self.log('Iteration has been loaded: {}.'.format(iteration))

        if 'model' in state_dict:
            self.state.model.load_state_dict(state_dict['model'])
            self.log('Model has been loaded.')

        if 'optimizer' in state_dict and self.state.optimizer is not None:
            self.state.optimizer.load_state_dict(state_dict['optimizer'])
            self.log('Optimizer has been loaded.')

        self.log('Snapshot loaded.')

    def step(self):
        self.state.iteration += 1

    def log(self, message, level='INFO'):
        level = logging.getLevelName(level)
        self.logger.log(level, message)

    def __enter__(self):
        return self

    def __exit__(self, type_, value_, trace_):
        torch.cuda.empty_cache()
