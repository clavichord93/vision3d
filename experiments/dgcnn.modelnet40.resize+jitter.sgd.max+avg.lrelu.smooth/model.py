from collections import OrderedDict

import torch
import torch.nn as nn

from vision3d.modules.dgcnn import EdgeConv
from vision3d.utils.pytorch_utils import create_linear_blocks, create_conv1d_blocks


class DynamicGraphCNN(nn.Module):
    def __init__(self, num_class):
        super(DynamicGraphCNN, self).__init__()

        # EdgeConv
        self.edgeconv1 = EdgeConv(3, 64, num_neighbor=20, activation='lrelu', negative_slope=0.2)
        self.edgeconv2 = EdgeConv(64, 64, num_neighbor=20, activation='lrelu', negative_slope=0.2)
        self.edgeconv3 = EdgeConv(64, 128, num_neighbor=20, activation='lrelu', negative_slope=0.2)
        self.edgeconv4 = EdgeConv(128, 256, num_neighbor=20, activation='lrelu', negative_slope=0.2)

        # Shared MLP
        layers = create_conv1d_blocks(512, 1024, kernel_size=1, activation='lrelu', negative_slope=0.2)
        self.shared_mlp = nn.Sequential(OrderedDict(layers))

        # classifier
        layers = create_linear_blocks(1024 * 2, [512, 256], dropout=0.5, activation='lrelu', negative_slope=0.2)
        layers.append(('fc3', nn.Linear(256, num_class)))
        self.classifier = nn.Sequential(OrderedDict(layers))

    def forward(self, points):
        # backbone
        points1 = self.edgeconv1(points)
        points2 = self.edgeconv2(points1)
        points3 = self.edgeconv3(points2)
        points4 = self.edgeconv4(points3)
        points = torch.cat([points1, points2, points3, points4], dim=1)
        points = self.shared_mlp(points)

        # classifier
        max_agg, _ = points.max(dim=2)
        avg_agg = points.mean(dim=2)
        features = torch.cat([max_agg, avg_agg], dim=1)
        outputs = self.classifier(features)

        return outputs


def create_model(num_class):
    return DynamicGraphCNN(num_class)


if __name__ == '__main__':
    model = DynamicGraphCNN(40).cuda()
    print(model)
    print(model.state_dict().keys())

    points = torch.randn(8, 3, 1024).cuda()
    pred = model(points)
    print(pred.shape)
