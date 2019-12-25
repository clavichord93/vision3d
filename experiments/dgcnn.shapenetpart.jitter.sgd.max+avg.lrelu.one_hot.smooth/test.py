import argparse
import os.path as osp
from tqdm import tqdm
import time

import torch
import numpy as np

from vision3d.utils.metrics import AccuracyRecorderV2, PartMeanIoURecorder
from vision3d.engine.engine import Engine
from dataset import test_data_loader
from config import config
from model import create_model


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', metavar='D', default='test', help='data split for testing')
    parser.add_argument('--start_epoch', metavar='S', required=True, type=int, help='start epoch')
    parser.add_argument('--end_epoch', metavar='E', required=True, type=int, help='end epoch')
    parser.add_argument('--steps', metavar='N', type=int, default=1, help='epoch steps for testing')
    parser.add_argument('--verbose', action='store_true', help='verbose mode')
    parser.add_argument('--tensorboardx', action='store_true', help='use tensorboardX')
    return parser


def main():
    parser = make_parser()
    log_file = osp.join(config.PATH.logs_dir, 'test-{}.log'.format(time.strftime('%Y%m%d-%H%M%S')))
    with Engine(log_file=log_file, default_parser=parser, seed=config.seed) as engine:
        args = engine.args

        data_loader = test_data_loader(config.PATH.data_root, args.split, config.TEST)

        model = create_model(config.DATA.num_class, config.DATA.num_part)
        engine.register_state(model=model)

        num_iter_per_epoch = len(data_loader)

        best_accuracy = 0
        best_accuracy_epoch = -1
        best_mean_iou_over_instance = 0
        best_mean_iou_over_instance_epoch = -1
        best_mean_iou_over_category = 0
        best_mean_iou_over_category_epoch = -1
        for epoch in range(args.start_epoch, args.end_epoch + 1, args.steps):
            snapshot = osp.join(config.PATH.snapshot_dir, 'epoch-{}.pth.tar'.format(epoch))
            engine.resume_from_snapshot(snapshot)

            model = engine.get_model()
            model.eval()

            accuracy_recorder = AccuracyRecorderV2(config.DATA.num_part)
            mean_iou_recorder = PartMeanIoURecorder(config.DATA.num_class,
                                                    config.DATA.num_part,
                                                    config.DATA.class_id_to_part_ids)

            start_time = time.time()

            pbar = tqdm(enumerate(data_loader), total=num_iter_per_epoch)
            for i, batch in pbar:
                points, normals, labels, class_ids = batch
                points = points.cuda()
                normals = normals.cuda()
                class_ids = class_ids.cuda()

                prepare_time = time.time() - start_time

                outputs = model(points, normals, class_ids)
                preds = outputs.argmax(dim=1)

                batch_size = preds.shape[0]
                preds = preds.detach().cpu().numpy()
                labels = labels.cpu().numpy()
                class_ids = class_ids.cpu().numpy()
                accuracy_recorder.add_records(preds, labels)
                for j in range(batch_size):
                    mean_iou_recorder.add_records(preds[j], labels[j], class_ids[j])

                process_time = time.time() - start_time - prepare_time

                message = "Epoch {}, ".format(epoch) + \
                          "data: {:.3f}s, proc: {:.3f}s".format(prepare_time, process_time)
                pbar.set_description(message)

                start_time = time.time()

            accuracy = accuracy_recorder.accuracy()
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_accuracy_epoch = epoch
            mean_iou_over_instance = mean_iou_recorder.mean_iou_over_instance()
            if mean_iou_over_instance > best_mean_iou_over_instance:
                best_mean_iou_over_instance = mean_iou_over_instance
                best_mean_iou_over_instance_epoch = epoch
            mean_iou_over_category = mean_iou_recorder.mean_iou_over_class()
            if mean_iou_over_category > best_mean_iou_over_category:
                best_mean_iou_over_category = mean_iou_over_category
                best_mean_iou_over_category_epoch = epoch

            message = 'Epoch {}, '.format(epoch) + \
                      'acc: {:.3f}, '.format(accuracy) + \
                      'mIoU (instance): {:.3f}, '.format(mean_iou_over_instance) + \
                      'mIoU (category): {:.3f}'.format(mean_iou_over_category)
            engine.log(message)
            if args.verbose:
                for i in range(config.DATA.num_part):
                    part_name = config.DATA.part_names[i]
                    accuracy = accuracy_recorder.accuracy_per_class(i)
                    engine.log("  {}, acc: {:.3f}".format(part_name, accuracy))
                for i in range(config.DATA.num_class):
                    class_name = config.DATA.class_names[i]
                    mean_iou = mean_iou_recorder.mean_iou_per_class(i)
                    engine.log("  {}, mIoU: {:.3f}".format(class_name, mean_iou))

        message = "Best acc: {:.3f}, best epoch: {}".format(best_accuracy, best_accuracy_epoch)
        engine.log(message)
        message = "Best mIoU (instance): {:.3f}, best epoch: {}".format(best_mean_iou_over_instance,
                                                                        best_mean_iou_over_instance_epoch)
        engine.log(message)
        message = "Best mIoU (category): {:.3f}, best epoch: {}".format(best_mean_iou_over_category,
                                                                        best_mean_iou_over_category_epoch)
        engine.log(message)


if __name__ == '__main__':
    main()
