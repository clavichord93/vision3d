import argparse
import os.path as osp
from tqdm import tqdm
import time

import torch

from vision3d.utils.metrics import AccuracyRecorderV1
from vision3d.engine.engine import Engine
from dataset import test_data_loader
from config import config
from model import create_model


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_epoch', metavar='S', required=True, type=int, help='start epoch')
    parser.add_argument('--end_epoch', metavar='E', required=True, type=int, help='end epoch')
    parser.add_argument('--verbose', action='store_true', help='verbose mode')
    parser.add_argument('--tensorboardx', action='store_true', help='use tensorboardX')
    return parser


def main():
    parser = make_parser()
    log_file = osp.join(config.PATH.logs_dir, 'test-{}.log'.format(time.strftime('%Y%m%d-%H%M%S')))
    with Engine(log_file=log_file, default_parser=parser, seed=config.seed) as engine:
        args = engine.args

        data_loader = test_data_loader(config.PATH.data_root, config.TEST)

        model = create_model(config.DATA.num_class)
        engine.register_state(model=model)

        num_iter_per_epoch = len(data_loader)

        best_accuracy = 0
        best_epoch = -1
        for epoch in range(args.start_epoch, args.end_epoch + 1):
            snapshot = osp.join(config.PATH.snapshot_dir, 'epoch-{}.pth.tar'.format(epoch))
            engine.resume_from_snapshot(snapshot)

            model = engine.get_model()
            model.eval()

            accuracy_recorder = AccuracyRecorderV1(config.DATA.num_class)

            start_time = time.time()

            pbar = tqdm(enumerate(data_loader), total=num_iter_per_epoch)
            for i, batch in pbar:
                points, labels = batch
                points = points.cuda()

                prepare_time = time.time() - start_time

                outputs = model(points)
                preds = outputs.argmax(dim=1).tolist()

                for pred, label in zip(preds, labels.tolist()):
                    accuracy_recorder.add_record(pred, label)

                process_time = time.time() - start_time - prepare_time

                message = "Epoch {}, ".format(epoch) + \
                          "data: {:.3f}s, proc: {:.3f}s".format(prepare_time, process_time)
                pbar.set_description(message)

                start_time = time.time()

            accuracy = accuracy_recorder.get_overall_accuracy()
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_epoch = epoch

            message = "Epoch {}, acc: {:.3f}".format(epoch, accuracy)
            engine.log(message)
            if args.verbose:
                for i in range(config.DATA.num_class):
                    class_name = config.DATA.class_names[i]
                    accuracy = accuracy_recorder.get_accuracy(i)
                    engine.log("  {}: {:.3f}".format(class_name, accuracy))

        message = "Best acc: {:.3f}, best epoch: {}".format(best_accuracy, best_epoch)
        engine.log(message)


if __name__ == '__main__':
    main()
