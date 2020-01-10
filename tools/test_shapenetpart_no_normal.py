import argparse
import os
import os.path as osp
from tqdm import tqdm
import time

import torch
import torch.nn as nn

from vision3d.utils.metrics import AccuracyMeter, PartMeanIoUMeter
from vision3d.engine.engine import Engine
from dataset import test_data_loader
from config import config
from model import create_model


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', metavar='D', default='test', help='data split for testing')
    parser.add_argument('--start_epoch', metavar='S', default=0, type=int, help='start epoch')
    parser.add_argument('--end_epoch', metavar='E', default=config.max_epoch-1, type=int, help='end epoch')
    parser.add_argument('--steps', metavar='N', type=int, default=1, help='epoch steps for testing')
    parser.add_argument('--verbose', action='store_true', help='verbose mode')
    parser.add_argument('--tensorboardx', action='store_true', help='use tensorboardX')
    return parser


def test_one_epoch(engine, data_loader, model, epoch):
    model.eval()
    accuracy_meter = AccuracyMeter(config.num_part)
    miou_meter = PartMeanIoUMeter(config.num_class, config.num_part, config.class_id_to_part_ids)
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    start_time = time.time()

    for i, batch in pbar:
        points, labels, class_ids = batch
        points = points.cuda()
        class_ids = class_ids.cuda()

        prepare_time = time.time() - start_time

        with torch.no_grad():
            outputs = model(points, class_ids)
            preds = outputs.argmax(dim=1).detach().cpu().numpy()
            labels = labels.cpu().numpy()
            class_ids = class_ids.cpu().numpy()
            accuracy_meter.add_results(preds, labels)
            miou_meter.add_results(preds, labels, class_ids)

        process_time = time.time() - start_time - prepare_time

        message = 'Epoch {}, '.format(epoch) + \
                  'data: {:.3f}s, proc: {:.3f}s'.format(prepare_time, process_time)
        pbar.set_description(message)

        start_time = time.time()

    accuracy = accuracy_meter.accuracy()
    instance_miou = miou_meter.instance_miou()
    class_miou = miou_meter.class_miou()

    message = 'Epoch {}, '.format(epoch) + \
              'acc: {:.3f}, '.format(accuracy) + \
              'mIoU (instance): {:.3f}, '.format(instance_miou) + \
              'mIoU (category): {:.3f}'.format(class_miou)
    engine.logger.info(message)
    if engine.args.verbose:
        for i in range(config.num_part):
            part_name = config.part_names[i]
            accuracy = accuracy_meter.accuracy_per_class(i)
            engine.logger.info('  {}, acc: {:.3f}'.format(part_name, accuracy))
        for i in range(config.num_class):
            class_name = config.class_names[i]
            mean_iou = miou_meter.instance_miou_per_class(i)
            engine.logger.info('  {}, mIoU: {:.3f}'.format(class_name, mean_iou))

    return accuracy, instance_miou, class_miou


def main():
    parser = make_parser()
    log_file = osp.join(config.logs_dir, 'test-{}.log'.format(time.strftime('%Y%m%d-%H%M%S')))
    with Engine(log_file=log_file, default_parser=parser, seed=config.seed) as engine:
        start_time = time.time()
        data_loader = test_data_loader(config, engine.args.split)
        loading_time = time.time() - start_time
        message = 'Data loader created: {:.3f}s collapsed.'.format(loading_time)
        engine.logger.info(message)

        model = create_model(config.num_class, config.num_part).cuda()
        if engine.data_parallel:
            model = nn.DataParallel(model)
        engine.register_state(model=model)

        if engine.snapshot is not None:
            engine.load_snapshot(engine.snapshot)
            test_one_epoch(engine, data_loader, model, engine.state.epoch)
        else:
            best_accuracy = 0
            best_accuracy_epoch = -1
            best_instance_miou = 0
            best_instance_miou_epoch = -1
            best_class_miou = 0
            best_class_miou_epoch = -1
            for epoch in range(engine.args.start_epoch, engine.args.end_epoch + 1, engine.args.steps):
                snapshot = osp.join(config.snapshot_dir, 'epoch-{}.pth.tar'.format(epoch))
                engine.load_snapshot(snapshot)
                accuracy, instance_miou, class_miou = test_one_epoch(engine, data_loader, model, epoch)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_accuracy_epoch = epoch
                if instance_miou > best_instance_miou:
                    best_instance_miou = instance_miou
                    best_instance_miou_epoch = epoch
                if class_miou > best_class_miou:
                    best_class_miou = class_miou
                    best_class_miou_epoch = epoch
            message = 'Best acc: {:.3f}, best epoch: {}'.format(best_accuracy, best_accuracy_epoch)
            engine.logger.info(message)
            message = 'Best instance mIoU: {:.3f}, best epoch: {}'.format(best_instance_miou, best_instance_miou_epoch)
            engine.logger.info(message)
            message = 'Best category mIoU: {:.3f}, best epoch: {}'.format(best_class_miou, best_class_miou_epoch)
            engine.logger.info(message)


if __name__ == '__main__':
    main()
