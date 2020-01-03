import argparse
import os
import os.path as osp
import time

import numpy as np
import torch
import torch.optim as optim

from vision3d.utils.metrics import OverallAccuracy, PartMeanIoU
from vision3d.engine.engine import Engine
from vision3d.modules.pointnet import PointNetLoss
from dataset import train_data_loader
from config import config
from model import create_model


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', metavar='D', default='trainval', help='data split for training')
    parser.add_argument('--steps', metavar='N', type=int, default=10, help='steps for logs')
    parser.add_argument('--tensorboardx', action='store_true', help='use tensorboardX')
    return parser


def main():
    parser = make_parser()
    log_file = osp.join(config.logs_dir, 'train-{}.log'.format(time.strftime('%Y%m%d-%H%M%S')))
    with Engine(log_file=log_file, default_parser=parser, seed=config.seed) as engine:
        args = engine.args

        data_loader = train_data_loader(config, args.split)

        model = create_model(config.num_class, config.num_part)
        base_params = [parameter for name, parameter in model.named_parameters() if 'tnet' not in name]
        tnet_params = [parameter for name, parameter in model.named_parameters() if 'tnet' in name]
        optimizer = optim.SGD([{'params': base_params, 'lr': config.learning_rate},
                               {'params': tnet_params, 'lr': config.learning_rate * 0.1}],
                              lr=config.learning_rate,
                              weight_decay=config.weight_decay,
                              momentum=config.momentum)
        engine.register_state(model=model, optimizer=optimizer)

        if engine.snapshot is not None:
            engine.load_snapshot(engine.snapshot)

        last_epoch = engine.state.epoch
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                         config.max_epoch,
                                                         eta_min=config.eta_min,
                                                         last_epoch=last_epoch)

        model = engine.get_cuda_model()
        model.train()
        loss_func = PointNetLoss(alpha=config.tnet_loss_alpha, eps=config.label_smoothing_eps).cuda()

        num_iter_per_epoch = len(data_loader)

        for epoch in range(last_epoch + 1, config.max_epoch):
            oa_metric = OverallAccuracy(config.num_part)
            miou_metric = PartMeanIoU(config.num_class, config.num_part, config.class_id_to_part_ids)

            start_time = time.time()

            for i, batch in enumerate(data_loader):
                points, labels, class_ids = batch
                points = points.cuda()
                labels = labels.cuda()
                class_ids = class_ids.cuda()

                prepare_time = time.time() - start_time

                outputs, transforms = model(points, class_ids)
                loss, cls_loss, tnet_loss = loss_func(outputs, labels, transforms)

                loss_val = loss.item()
                cls_loss_val = cls_loss.item()
                tnet_loss_val = tnet_loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                preds = outputs.argmax(dim=1).detach().cpu().numpy()
                labels = labels.cpu().numpy()
                class_ids = class_ids.cpu().numpy()
                oa_metric.add_results(preds, labels)
                miou_metric.add_results(preds, labels, class_ids)

                process_time = time.time() - start_time - prepare_time

                if (i + 1) % args.steps == 0:
                    learning_rate = scheduler.get_lr()[0]
                    message = 'Epoch {}/{}, '.format(epoch + 1, config.max_epoch) + \
                              'iter {}/{}, '.format(i + 1, num_iter_per_epoch) + \
                              'loss: {:.3f}, '.format(loss_val) + \
                              'cls_loss: {:.3f}, '.format(cls_loss_val) + \
                              'tnet_loss: {:.3f}, '.format(tnet_loss_val) + \
                              'lr: {:.3e}, '.format(learning_rate) + \
                              'prep: {:.3f}s, proc: {:.3f}s'.format(prepare_time, process_time)
                    engine.log(message)

                engine.step()

                start_time = time.time()

            message = 'Epoch {}, '.format(epoch) + \
                      'acc: {:.3f}, '.format(oa_metric.accuracy()) + \
                      'mIoU (instance): {:.3f}, '.format(miou_metric.instance_miou()) + \
                      'mIoU (category): {:.3f}'.format(miou_metric.class_miou())
            engine.log(message)

            engine.register_state(epoch=epoch)
            scheduler.step()

            snapshot = osp.join(config.snapshot_dir, 'epoch-{}.pth.tar'.format(epoch))
            engine.save_snapshot(snapshot)


if __name__ == '__main__':
    main()
