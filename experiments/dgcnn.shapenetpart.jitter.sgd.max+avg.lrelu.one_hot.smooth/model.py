from collections import OrderedDict

import torch
import torch.nn as nn

from vision3d.modules.dgcnn import EdgeConv
from vision3d.utils.pytorch_utils import create_conv1d_blocks, create_linear_blocks


class DynamicGraphCNN(nn.Module):
    def __init__(self, num_class, num_part):
        super(DynamicGraphCNN, self).__init__()
        self.num_class = num_class
        self.num_part = num_part

        # EdgeConv
        self.edgeconv1 = EdgeConv(3, [64, 64], num_neighbor=20)
        self.edgeconv2 = EdgeConv(64, [64, 64], num_neighbor=20)
        self.edgeconv3 = EdgeConv(64, 64, num_neighbor=20)

        # Shared MLP
        layers = create_conv1d_blocks(192, 1024, kernel_size=1)
        self.shared_mlp = nn.Sequential(OrderedDict(layers))

        # MLP for categorical vector
        layers = create_linear_blocks(16, 64)
        self.mlp = nn.Sequential(OrderedDict(layers))

        # classifier
        layers = create_conv1d_blocks(1280, [256, 256, 128], kernel_size=1)
        layers.append(('conv4', nn.Conv1d(128, num_part, kernel_size=1)))
        self.classifier = nn.Sequential(OrderedDict(layers))

    def forward(self, *inputs):
        points, features, class_ids = inputs
        batch_size, _, num_point = points.shape
        one_hot = nn.functional.one_hot(class_ids, self.num_class).float()

        # backbone
        points1 = self.edgeconv1(points)
        points2 = self.edgeconv2(points1)
        points3 = self.edgeconv3(points2)
        points = torch.cat([points1, points2, points3], dim=1)

        points = self.shared_mlp(points)
        points, _ = points.max(dim=2)
        one_hot = self.mlp(one_hot)
        points = torch.cat([points, one_hot], dim=1).unsqueeze(2).repeat(1, 1, num_point)

        points = torch.cat([points1, points2, points3, points], dim=1)

        # classifier
        outputs = self.classifier(points)

        return outputs


def create_model(num_class, num_part):
    return DynamicGraphCNN(num_class, num_part)


if __name__ == '__main__':
    model = DynamicGraphCNN(16, 50).cuda()
    print(model)
    print(model.state_dict().keys())

    points = torch.randn(2, 3, 2048).cuda()
    normals = torch.randn(2, 3, 2048).cuda()
    class_ids = torch.randint(high=16, size=(2,)).cuda()
    pred = model(points, normals, class_ids)
    print(pred.shape)
