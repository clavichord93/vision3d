from collections import OrderedDict

import torch
import torch.nn as nn

from vision3d.utils.pytorch_utils import create_conv1d_blocks
from vision3d.modules.pointnet import TNet


class PointNet(nn.Module):
    def __init__(self, num_class, num_part):
        super(PointNet, self).__init__()
        self.num_class = num_class
        self.num_part = num_part

        # TNet
        self.tnet1 = TNet(input_dim=3, output_dims1=[64, 128, 1024], output_dims2=[512, 256])
        self.tnet2 = TNet(input_dim=128, output_dims1=[64, 128, 1024], output_dims2=[512, 256])

        # Shared MLP
        layers = create_conv1d_blocks(3, [64, 128, 128], kernel_size=1)
        self.shared_mlp1 = nn.Sequential(OrderedDict(layers))
        layers = create_conv1d_blocks(128, [128, 512, 2048], kernel_size=1)
        self.shared_mlp2 = nn.Sequential(OrderedDict(layers))

        # classifier
        layers = create_conv1d_blocks(3024, [256, 256, 128], kernel_size=1)
        layers.append(('conv4', nn.Conv1d(128, num_part, kernel_size=1)))
        self.classifier = nn.Sequential(OrderedDict(layers))

    def forward(self, points, class_ids):
        batch_size, _, num_point = points.shape
        one_hot = nn.functional.one_hot(class_ids, self.num_class).float()

        # backbone
        outputs = []

        i_transform = self.tnet1(points)
        points = torch.matmul(i_transform, points)

        num_layer = len(self.shared_mlp1)
        for i in range(num_layer):
            points = self.shared_mlp1[i](points)
            outputs.append(points)

        f_transform = self.tnet2(points)
        points = torch.matmul(f_transform, points)

        num_layer = len(self.shared_mlp2)
        for i in range(num_layer):
            points = self.shared_mlp2[i](points)
            if i < num_layer - 1:
                outputs.append(points)

        points, _ = points.max(dim=2, keepdim=True)
        points = points.repeat(1, 1, num_point)
        outputs.append(points)

        one_hot = one_hot.unsqueeze(2).repeat(1, 1, num_point)
        outputs.append(one_hot)

        points = torch.cat(outputs, dim=1)

        # classifier
        outputs = self.classifier(points)

        if self.training:
            return outputs, f_transform
        else:
            return outputs


def create_model(num_class, num_part):
    return PointNet(num_class, num_part)


if __name__ == '__main__':
    model = create_model(16, 50)
    print(model)
    print(model.state_dict().keys())

    points = torch.randn(2, 3, 2048)
    class_ids = torch.randint(high=16, size=(2,))
    outputs, transforms = model(points, class_ids)
    print(outputs.shape)
    print(transforms.shape)
