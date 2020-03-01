from collections import OrderedDict

import torch
import torch.nn as nn

from vision3d.modules.pointcnn import XConv
import vision3d.modules.pointcnn.functional as F
from vision3d.utils.pytorch_utils import create_conv1d_blocks, create_linear_blocks


class PointCNN(nn.Module):
    def __init__(self, num_class):
        super(PointCNN, self).__init__()

        # stem block to initialize features
        layers = create_conv1d_blocks(3, 24, kernel_size=1, batch_norm_after_activation=True, activation='elu')
        self.stem = nn.Sequential(OrderedDict(layers))

        # XConv
        self.xconv1 = XConv(24, 48, hidden_dim=12, kernel_size=8, dilation=1, depth_multiplier=4)
        self.xconv2 = XConv(48, 96, hidden_dim=12, kernel_size=12, dilation=2, depth_multiplier=2)
        self.xconv3 = XConv(96, 192, hidden_dim=24, kernel_size=16, dilation=2, depth_multiplier=2)
        self.xconv4 = XConv(192, 384, hidden_dim=48, kernel_size=16, dilation=3, depth_multiplier=2, with_global=True)

        # classifier
        layers = create_linear_blocks(480, [384, 192], batch_norm_after_activation=True, activation='elu', dropout=0.5)
        layers.append(('fc3', nn.Linear(192, num_class)))
        self.classifier = nn.Sequential(OrderedDict(layers))

    def forward(self, points):
        # backbone
        features = self.stem(points)

        points, features = self.xconv1(points, features, points)

        centroids = F.random_point_sampling_and_gather(points, 384)
        points, features = self.xconv2(points, features, centroids)

        centroids = F.random_point_sampling_and_gather(points, 128)
        points, features = self.xconv3(points, features, centroids)

        points, features = self.xconv4(points, features, centroids)

        # classifier
        features, _ = features.max(dim=2)
        outputs = self.classifier(features)

        return outputs


def create_model(num_class):
    return PointCNN(num_class)


if __name__ == '__main__':
    model = PointCNN(40).cuda()
    print(model)
    print(model.state_dict().keys())

    points = torch.randn(32, 3, 1024).cuda()
    pred = model(points)
    print(pred.shape)
