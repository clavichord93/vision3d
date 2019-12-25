from collections import OrderedDict

import torch
import torch.nn as nn

from vision3d.utils.pytorch_utils import create_conv1d_blocks, create_fc_blocks
from vision3d.modules.pointnet import TNet


class PointNet(nn.Module):
    def __init__(self, num_class):
        super(PointNet, self).__init__()

        # TNet
        self.tnet1 = TNet(input_dim=3, output_dims1=[64, 128, 1024], output_dims2=[512, 256])
        self.tnet2 = TNet(input_dim=64, output_dims1=[64, 128, 1024], output_dims2=[512, 256])

        # Shared MLP
        layers = create_conv1d_blocks(3, [64, 64], kernel_size=1)
        self.shared_mlp1 = nn.Sequential(OrderedDict(layers))
        layers = create_conv1d_blocks(64, [64, 128, 1024], kernel_size=1)
        self.shared_mlp2 = nn.Sequential(OrderedDict(layers))

        # classifier
        layers = create_fc_blocks(1024, [512, 256], dropout_ratio=0.3)
        layers.append(('fc3', nn.Linear(256, num_class)))
        self.classifier = nn.Sequential(OrderedDict(layers))

    def forward(self, points):
        # backbone
        i_transform = self.tnet1(points)
        points = torch.matmul(i_transform, points)
        points = self.shared_mlp1(points)

        f_transform = self.tnet2(points)
        points = torch.matmul(f_transform, points)
        points = self.shared_mlp2(points)

        # classifier
        features, _ = points.max(dim=2)
        outputs = self.classifier(features)

        if self.training:
            return outputs, f_transform
        else:
            return outputs


def create_model(num_class):
    return PointNet(num_class)


if __name__ == '__main__':
    model = create_model(40)
    print(model)
    print(model.state_dict().keys())

    points = torch.rand(32, 3, 1024)
    pred, transform = model(points)
    print(pred.shape)
    print(transform.shape)
