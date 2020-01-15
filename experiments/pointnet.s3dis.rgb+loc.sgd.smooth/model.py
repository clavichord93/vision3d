from collections import OrderedDict

import torch
import torch.nn as nn

from vision3d.utils.pytorch_utils import create_conv1d_blocks


class PointNet(nn.Module):
    def __init__(self, num_class):
        super(PointNet, self).__init__()
        self.num_class = num_class

        # Shared MLP
        layers = create_conv1d_blocks(9, [64, 64, 64, 128, 1024], kernel_size=1)
        self.shared_mlp = nn.Sequential(OrderedDict(layers))

        # MLP
        layers = create_conv1d_blocks(1024, [256, 128], kernel_size=1)
        self.mlp = nn.Sequential(OrderedDict(layers))

        # classifier
        layers = create_conv1d_blocks(1152, [512, 256], kernel_size=1, dropout=0.5)
        layers.append(('conv3', nn.Conv1d(256, num_class, kernel_size=1)))
        self.classifier = nn.Sequential(OrderedDict(layers))

    def forward(self, points, extras):
        points = torch.cat([points, extras], dim=1)
        num_point = points.shape[2]

        # backbone
        points = self.shared_mlp(points)
        g_points, _ = points.max(dim=2, keepdim=True)
        g_points = self.mlp(g_points)
        g_points = g_points.repeat(1, 1, num_point)
        points = torch.cat([points, g_points], dim=1)

        # classifier
        outputs = self.classifier(points)

        return outputs


def create_model(num_class):
    return PointNet(num_class)


if __name__ == '__main__':
    model = create_model(13)
    print(model)
    print(model.state_dict().keys())

    points = torch.randn(32, 3, 4096)
    extras = torch.randn(32, 6, 4096)
    outputs = model(points, extras)
    print(outputs.shape)
