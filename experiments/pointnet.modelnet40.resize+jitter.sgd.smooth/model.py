from collections import OrderedDict

import torch
import torch.nn as nn

from vision3d.utils.pytorch_utils import create_conv1d_blocks, create_linear_blocks


class PointNet(nn.Module):
    def __init__(self, num_class):
        super(PointNet, self).__init__()

        # Shared MLP
        layers = create_conv1d_blocks(3, [64, 64, 64, 128, 1024], kernel_size=1)
        self.shared_mlp = nn.Sequential(OrderedDict(layers))

        # classifier
        layers = create_linear_blocks(1024, [512, 256], dropout=0.3)
        layers.append(('fc3', nn.Linear(256, num_class)))
        self.classifier = nn.Sequential(OrderedDict(layers))

    def forward(self, points):
        # backbone
        points = self.shared_mlp(points)

        # classifier
        features, _ = points.max(dim=2)
        outputs = self.classifier(features)

        return outputs


def create_model(num_class):
    return PointNet(num_class)


if __name__ == '__main__':
    model = create_model(40)
    print(model)
    print(model.state_dict().keys())

    points = torch.rand(32, 3, 1024)
    pred = model(points)
    print(pred.shape)
