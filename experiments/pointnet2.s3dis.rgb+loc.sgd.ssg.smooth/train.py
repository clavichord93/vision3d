import argparse
import os
import os.path as osp
import time

import torch
import torch.nn as nn
import torch.optim as optim

from vision3d.utils.metrics import AccuracyMeter, MeanIoUMeter, AverageMeter
from vision3d.engine import Engine
from vision3d.utils.pytorch_utils import SmoothCrossEntropyLoss, CosineAnnealingFunction
from dataset import train_data_loader
from config import config
from model import create_model


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_area', metavar='D', type=int, required=True, help='area for testing')
    parser.add_argument('--steps', metavar='N', type=int, default=10, help='iteration steps for logging')
    parser.add_argument('--tensorboardx', action='store_true', help='use tensorboardX')
    return parser


def train_one_epoch(engine, data_loader, model, loss_func, optimizer, scheduler, epoch):
    model.train()
    num_iter_per_epoch = len(data_loader)
    accuracy_meter = AccuracyMeter(config.num_class)
    mean_iou_meter = MeanIoUMeter(config.num_class)
    loss_meter = AverageMeter()
    start_time = time.time()

    for i, batch in enumerate(data_loader):
        points, features, labels = batch
        points = points.cuda()
        features = features.cuda()
        labels = labels.cuda()

        prepare_time = time.time() - start_time

        outputs = model(points, features)
        loss = loss_func(outputs, labels)
        loss_val = loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1).detach().cpu().numpy()
        labels = labels.cpu().numpy()
        accuracy_meter.add_results(preds, labels)
        mean_iou_meter.add_results(preds, labels)
        loss_meter.add_results(loss_val)

        process_time = time.time() - start_time - prepare_time

        if (i + 1) % engine.args.steps == 0:
            learning_rate = scheduler.get_lr()[0]
            message = 'Epoch {}/{}, '.format(epoch + 1, config.max_epoch) + \
                      'iter {}/{}, '.format(i + 1, num_iter_per_epoch) + \
                      'loss: {:.3f}, '.format(loss_val) + \
                      'lr: {:.3e}, '.format(learning_rate) + \
                      'prep: {:.3f}s, proc: {:.3f}s'.format(prepare_time, process_time)
            engine.logger.info(message)

        engine.step()

        start_time = time.time()

    message = 'Epoch {}, '.format(epoch) + \
              'acc: {:.3f}, '.format(accuracy_meter.overall_accuracy()) + \
              'mIoU: {:.3f}, '.format(mean_iou_meter.mean_iou()) + \
              'loss: {:.3f}'.format(loss_meter.average())
    engine.logger.info(message)

    engine.register_state(epoch=epoch)
    scheduler.step()

    snapshot = osp.join(config.snapshot_dir, 'epoch-{}.pth.tar'.format(epoch))
    engine.save_snapshot(snapshot)


def main():
    parser = make_parser()
    log_file = osp.join(config.logs_dir, 'train-{}.log'.format(time.strftime('%Y%m%d-%H%M%S')))
    with Engine(log_file=log_file, default_parser=parser, seed=config.seed) as engine:
        start_time = time.time()
        data_loader = train_data_loader(config, engine.args.test_area)
        loading_time = time.time() - start_time
        message = 'Data loader created: {:.3f}s collapsed.'.format(loading_time)
        engine.logger.info(message)

        model = create_model(config.num_class).cuda()
        optimizer = optim.SGD(model.parameters(),
                              lr=config.learning_rate,
                              weight_decay=config.weight_decay,
                              momentum=config.momentum)
        loss_func = SmoothCrossEntropyLoss(eps=config.label_smoothing_eps).cuda()
        if engine.data_parallel:
            model = nn.DataParallel(model)
        engine.register_state(model=model, optimizer=optimizer)

        if engine.snapshot is not None:
            engine.load_snapshot(engine.snapshot)

        cosine_annealing_fn = CosineAnnealingFunction(config.max_epoch, eta_min=config.eta_min)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, cosine_annealing_fn, last_epoch=engine.state.epoch)

        for epoch in range(engine.state.epoch + 1, config.max_epoch):
            train_one_epoch(engine, data_loader, model, loss_func, optimizer, scheduler, epoch)


if __name__ == '__main__':
    main()
