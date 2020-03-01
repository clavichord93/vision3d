from collections import OrderedDict

import torch
import torch.nn as nn

from vision3d.modules.dgcnn import EdgeConv
from vision3d.utils.pytorch_utils import create_conv1d_blocks, create_linear_blocks


class DynamicGraphCNN(nn.Module):
    def __init__(self, num_class):
        super(DynamicGraphCNN, self).__init__()
        self.num_class = num_class

        # EdgeConv
        self.edgeconv1 = EdgeConv(9, [64, 64], num_neighbor=20)
        self.edgeconv2 = EdgeConv(64, [64, 64], num_neighbor=20)
        self.edgeconv3 = EdgeConv(64, 64, num_neighbor=20)

        # Shared MLP
        layers = create_conv1d_blocks(192, 1024, kernel_size=1)
        self.shared_mlp = nn.Sequential(OrderedDict(layers))

        # classifier
        layers = create_conv1d_blocks(1216, [256, 256, 128], kernel_size=1)
        layers.append(('conv4', nn.Conv1d(128, self.num_class, kernel_size=1)))
        self.classifier = nn.Sequential(OrderedDict(layers))

    def forward(self, points, features):
        points = torch.cat([points, features], dim=1)
        batch_size, _, num_point = points.shape

        # backbone
        points1 = self.edgeconv1(points)
        points2 = self.edgeconv2(points1)
        points3 = self.edgeconv3(points2)
        points = torch.cat([points1, points2, points3], dim=1)

        points = self.shared_mlp(points)
        points, _ = points.max(dim=2, keepdim=True)
        points = points.repeat(1, 1, num_point)
        points = torch.cat([points1, points2, points3, points], dim=1)

        # classifier
        outputs = self.classifier(points)

        return outputs


def create_model(num_class):
    return DynamicGraphCNN(num_class)


if __name__ == '__main__':
    model = DynamicGraphCNN(13).cuda()
    print(model)
    print(model.state_dict().keys())

    points = torch.randn(32, 3, 4096).cuda()
    features = torch.randn(32, 6, 4096).cuda()
    pred = model(points, features)
    print(pred.shape)
