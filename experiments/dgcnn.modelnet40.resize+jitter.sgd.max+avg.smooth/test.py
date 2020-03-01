import argparse
import os
import os.path as osp
from tqdm import tqdm
import time

import torch
import torch.nn as nn

from vision3d.utils.metrics import AccuracyMeter
from vision3d.engine import Engine
from dataset import test_data_loader
from config import config
from model import create_model


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_epoch', metavar='S', default=0, type=int, help='start epoch')
    parser.add_argument('--end_epoch', metavar='E', default=config.max_epoch-1, type=int, help='end epoch')
    parser.add_argument('--steps', metavar='N', type=int, default=1, help='epoch steps for testing')
    parser.add_argument('--verbose', action='store_true', help='verbose mode')
    parser.add_argument('--tensorboardx', action='store_true', help='use tensorboardX')
    return parser


def test_one_epoch(engine, data_loader, model, epoch):
    model.eval()
    accuracy_meter = AccuracyMeter(config.num_class)
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    start_time = time.time()

    for i, batch in pbar:
        points, labels = batch
        points = points.cuda()

        prepare_time = time.time() - start_time

        with torch.no_grad():
            outputs = model(points)
            preds = outputs.argmax(dim=1).detach().cpu().numpy()
            labels = labels.numpy()
            accuracy_meter.add_results(preds, labels)

        process_time = time.time() - start_time - prepare_time

        message = 'Epoch {}, data: {:.3f}s, proc: {:.3f}s'.format(epoch, prepare_time, process_time)
        pbar.set_description(message)

        start_time = time.time()

    accuracy = accuracy_meter.overall_accuracy()

    message = 'Epoch {}, acc: {:.3f}'.format(epoch, accuracy)
    engine.logger.info(message)
    if engine.args.verbose:
        for i in range(config.num_class):
            class_name = config.class_names[i]
            accuracy_per_class = accuracy_meter.accuracy_per_class(i)
            engine.logger.info('  {}: {:.3f}'.format(class_name, accuracy_per_class))

    return accuracy


def main():
    parser = make_parser()
    log_file = osp.join(config.logs_dir, 'test-{}.log'.format(time.strftime('%Y%m%d-%H%M%S')))
    with Engine(log_file=log_file, default_parser=parser, seed=config.seed) as engine:
        start_time = time.time()
        data_loader = test_data_loader(config)
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
            best_epoch = -1
            for epoch in range(engine.args.start_epoch, engine.args.end_epoch + 1, engine.args.steps):
                snapshot = osp.join(config.snapshot_dir, 'epoch-{}.pth.tar'.format(epoch))
                engine.load_snapshot(snapshot)
                accuracy = test_one_epoch(engine, data_loader, model, epoch)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_epoch = epoch
            message = 'Best acc: {:.3f}, best epoch: {}'.format(best_accuracy, best_epoch)
            engine.logger.info(message)


if __name__ == '__main__':
    main()
