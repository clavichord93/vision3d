import argparse
import os.path as osp
from tqdm import tqdm
import time

import torch

from vision3d.utils.metrics import OverallAccuracy
from vision3d.engine.engine import Engine
from dataset import test_data_loader
from config import config
from model import create_model


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_epoch', metavar='S', default=0, type=int, help='start epoch')
    parser.add_argument('--end_epoch', metavar='E', default=config.max_epoch-1, type=int, help='end epoch')
    parser.add_argument('--verbose', action='store_true', help='verbose mode')
    parser.add_argument('--tensorboardx', action='store_true', help='use tensorboardX')
    return parser


def test_epoch(engine, data_loader, epoch, verbose=False):
    model = engine.get_cuda_model()
    model.eval()

    oa_metric = OverallAccuracy(config.num_class)

    start_time = time.time()

    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for i, batch in pbar:
        points, labels = batch
        points = points.cuda()

        prepare_time = time.time() - start_time

        outputs = model(points)
        preds = outputs.argmax(dim=1)

        preds = preds.detach().cpu().numpy()
        labels = labels.numpy()
        oa_metric.add_results(preds, labels)

        process_time = time.time() - start_time - prepare_time

        message = 'Epoch {}, data: {:.3f}s, proc: {:.3f}s'.format(epoch, prepare_time, process_time)
        pbar.set_description(message)

        start_time = time.time()

    accuracy = oa_metric.accuracy()

    message = 'Epoch {}, acc: {:.3f}'.format(epoch, accuracy)
    engine.log(message)
    if verbose:
        for i in range(config.num_class):
            class_name = config.class_names[i]
            accuracy = oa_metric.accuracy_per_class(i)
            engine.log('  {}: {:.3f}'.format(class_name, accuracy))

    return accuracy


def main():
    parser = make_parser()
    log_file = osp.join(config.logs_dir, 'test-{}.log'.format(time.strftime('%Y%m%d-%H%M%S')))
    with Engine(log_file=log_file, default_parser=parser, seed=config.seed) as engine:
        args = engine.args

        data_loader = test_data_loader(config)

        model = create_model(config.num_class)
        engine.register_state(model=model)

        if engine.snapshot is not None:
            engine.load_snapshot(engine.snapshot)
            test_epoch(engine, data_loader, epoch=engine.state.epoch, verbose=args.verbose)
        else:
            best_accuracy = 0
            best_epoch = -1
            for epoch in range(args.start_epoch, args.end_epoch + 1):
                snapshot = osp.join(config.snapshot_dir, 'epoch-{}.pth.tar'.format(epoch))
                engine.load_snapshot(snapshot)
                accuracy = test_epoch(engine, data_loader, epoch=epoch, verbose=args.verbose)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_epoch = epoch
            message = 'Best acc: {:.3f}, best epoch: {}'.format(best_accuracy, best_epoch)
            engine.log(message)


if __name__ == '__main__':
    main()
