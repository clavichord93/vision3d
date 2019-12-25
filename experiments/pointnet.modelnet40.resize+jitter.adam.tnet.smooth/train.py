import argparse
import os
import os.path as osp
import time

import torch
import torch.nn as nn
import torch.optim as optim

from vision3d.utils.metrics import AccuracyRecorderV1
from vision3d.engine.engine import Engine
from vision3d.modules.pointnet import PointNetLoss
from dataset import train_data_loader
from config import config
from model import create_model


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', metavar='N', type=int, default=10, help='iteration steps for logging')
    parser.add_argument('--tensorboardx', action='store_true', help='use tensorboardX')
    return parser


def main():
    parser = make_parser()
    log_file = osp.join(config.PATH.logs_dir, 'train-{}.log'.format(time.strftime('%Y%m%d-%H%M%S')))
    with Engine(log_file=log_file, default_parser=parser, seed=config.seed) as engine:
        args = engine.args

        data_loader = train_data_loader(config.PATH.data_root, config.TRAIN)

        model = create_model(config.DATA.num_class)
        optimizer = optim.Adam(model.parameters(),
                               lr=config.OPTIMIZER.learning_rate,
                               weight_decay=config.OPTIMIZER.weight_decay)
        engine.register_state(model=model, optimizer=optimizer)

        if engine.resume_training:
            engine.resume_from_snapshot()

        model = engine.get_model()
        model.train()

        loss_func = PointNetLoss(alpha=config.MODEL.alpha, eps=config.MODEL.eps).cuda()

        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              config.OPTIMIZER.steps,
                                              gamma=config.OPTIMIZER.gamma,
                                              last_epoch=engine.state.epoch)

        num_iter_per_epoch = len(data_loader)

        for epoch in range(engine.state.epoch + 1, config.OPTIMIZER.max_epoch):
            accuracy_recorder = AccuracyRecorderV1(config.DATA.num_class)

            start_time = time.time()

            for i, batch in enumerate(data_loader):
                points, labels = batch
                points = points.cuda()
                labels = labels.cuda()

                prepare_time = time.time() - start_time

                outputs, transforms = model(points)
                loss, cls_loss, tnet_loss = loss_func(outputs, labels, transforms)

                loss_val = loss.item()
                cls_loss_val = cls_loss.item()
                tnet_loss_val = tnet_loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                preds = outputs.argmax(dim=1).tolist()
                for pred, label in zip(preds, labels.tolist()):
                    accuracy_recorder.add_record(pred, label)

                process_time = time.time() - start_time - prepare_time

                if (i + 1) % args.steps == 0:
                    learning_rate = scheduler.get_lr()[0]
                    message = 'Epoch {}/{}, '.format(epoch + 1, config.OPTIMIZER.max_epoch) + \
                              'iter {}/{}, '.format(i + 1, num_iter_per_epoch) + \
                              'loss: {:.3f}, '.format(loss_val) + \
                              'cls_loss: {:.3f}, '.format(cls_loss_val) + \
                              'tnet_loss: {:.3f}, '.format(tnet_loss_val) + \
                              'lr: {:.3e}, '.format(learning_rate) + \
                              'prep: {:.3f}s, proc: {:.3f}s'.format(prepare_time, process_time)
                    engine.log(message)

                engine.step()

                start_time = time.time()

            message = 'Epoch {}, acc: {:.3f}'.format(epoch, accuracy_recorder.get_overall_accuracy())
            engine.log(message)

            engine.register_state(epoch=epoch)
            scheduler.step()

            snapshot = osp.join(config.PATH.snapshot_dir, 'epoch-{}.pth.tar'.format(epoch))
            engine.save_snapshot(snapshot)


if __name__ == '__main__':
    main()
