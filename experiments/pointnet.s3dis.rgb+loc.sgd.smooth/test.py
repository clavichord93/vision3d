import argparse
import os
import os.path as osp
from tqdm import tqdm
import time

import torch
import torch.nn as nn
import numpy as np

from vision3d.utils.metrics import AccuracyMeter, MeanIoUMeter, AverageMeter
from vision3d.engine import Engine
from dataset import test_data_loader
from config import config
from model import create_model


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_area', metavar='D', type=int, required=True, help='area for testing')
    parser.add_argument('--start_epoch', metavar='S', default=0, type=int, help='start epoch')
    parser.add_argument('--end_epoch', metavar='E', default=config.max_epoch-1, type=int, help='end epoch')
    parser.add_argument('--steps', metavar='N', type=int, default=1, help='epoch steps for testing')
    parser.add_argument('--verbose', action='store_true', help='verbose mode')
    parser.add_argument('--tensorboardx', action='store_true', help='use tensorboardX')
    return parser


def test_one_epoch(engine, data_loader, model, epoch):
    model.eval()
    accuracy_meter = AccuracyMeter(config.num_class)
    mean_iou_meter = MeanIoUMeter(config.num_class)
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    start_time = time.time()

    for i, scene in pbar:
        points, features, point_indices, labels = scene
        points = points.squeeze(0)
        features = features.squeeze(0)
        point_indices = point_indices.squeeze(0).numpy()
        labels = labels.squeeze(0).numpy()

        num_point_in_scene = labels.shape[0]
        preds = np.zeros([num_point_in_scene, config.num_class])

        num_sample = points.shape[0]
        num_batch = (num_sample + config.test_batch_size - 1) // config.test_batch_size
        prepare_time = time.time() - start_time

        for batch_index in range(num_batch):
            start_index = batch_index * config.test_batch_size
            end_index = min((batch_index + 1) * config.test_batch_size, num_sample)
            batch_points = points[start_index:end_index]
            batch_features = features[start_index:end_index]
            batch_point_indices = point_indices[start_index:end_index].flatten()

            batch_points = batch_points.cuda()
            batch_features = batch_features.cuda()

            with torch.no_grad():
                batch_outputs = model(batch_points, batch_features)
                batch_outputs = nn.functional.softmax(batch_outputs, dim=1).transpose(1, 2).contiguous()\
                    .view(-1, config.num_class).detach().cpu().numpy()
                np.add.at(preds, batch_point_indices, batch_outputs)

        preds = preds.argmax(axis=1)
        accuracy_meter.add_results(preds, labels)
        mean_iou_meter.add_results(preds, labels)

        process_time = time.time() - start_time - prepare_time

        message = 'Epoch {}, data: {:.3f}s, proc: {:.3f}s'.format(epoch, prepare_time, process_time)
        pbar.set_description(message)

        start_time = time.time()

    accuracy = accuracy_meter.overall_accuracy()
    mean_iou = mean_iou_meter.mean_iou()

    message = 'Epoch {}, acc: {:.3f}, mIoU: {:.3f}'.format(epoch, accuracy, mean_iou)
    engine.logger.info(message)
    if engine.args.verbose:
        for i in range(config.num_class):
            part_name = config.part_names[i]
            accuracy_per_class = accuracy_meter.accuracy_per_class(i)
            engine.logger.info('  {}, acc: {:.3f}'.format(part_name, accuracy_per_class))
        for i in range(config.num_class):
            class_name = config.class_names[i]
            mean_iou_per_class = mean_iou_meter.mean_iou_per_class(i)
            engine.logger.info('  {}, mIoU: {:.3f}'.format(class_name, mean_iou_per_class))

    return accuracy, mean_iou


def main():
    parser = make_parser()
    log_file = osp.join(config.logs_dir, 'test-{}.log'.format(time.strftime('%Y%m%d-%H%M%S')))
    with Engine(log_file=log_file, default_parser=parser, seed=config.seed) as engine:
        start_time = time.time()
        data_loader = test_data_loader(config, engine.args.test_area)
        loading_time = time.time() - start_time
        message = 'Data loader created: {:.3f}s collapsed.'.format(loading_time)
        engine.logger.info(message)

        model = create_model(config.num_class).cuda()
        if engine.data_parallel:
            model = nn.DataParallel(model)
        engine.register_state(model=model)

        if engine.snapshot is not None:
            engine.load_snapshot(engine.snapshot)
            test_one_epoch(engine, data_loader, model, engine.state.epoch)
        else:
            best_accuracy = 0
            best_accuracy_epoch = -1
            best_mean_iou = 0
            best_mean_iou_epoch = -1
            for epoch in range(engine.args.start_epoch, engine.args.end_epoch + 1, engine.args.steps):
                snapshot = osp.join(config.snapshot_dir, 'epoch-{}.pth.tar'.format(epoch))
                engine.load_snapshot(snapshot)
                accuracy, mean_iou = test_one_epoch(engine, data_loader, model, epoch)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_accuracy_epoch = epoch
                if mean_iou > best_mean_iou:
                    best_mean_iou = mean_iou
                    best_mean_iou_epoch = epoch
            message = 'Best acc: {:.3f}, best epoch: {}'.format(best_accuracy, best_accuracy_epoch)
            engine.logger.info(message)
            message = 'Best mIoU: {:.3f}, best epoch: {}'.format(best_mean_iou, best_mean_iou_epoch)
            engine.logger.info(message)


if __name__ == '__main__':
    main()
