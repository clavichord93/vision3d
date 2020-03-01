from collections import OrderedDict

import torch
import torch.nn as nn

from vision3d.modules.pointnet2 import SetAbstractionModule, FeaturePropagationModule
from vision3d.utils.pytorch_utils import create_conv1d_blocks


class PointNet2(nn.Module):
    def __init__(self, num_class):
        super(PointNet2, self).__init__()
        self.num_class = num_class

        # SA module
        self.sa_module1 = SetAbstractionModule(input_dim=9,
                                               output_dims=[32, 32, 64],
                                               num_centroid=1024,
                                               num_sample=32,
                                               radius=0.1)
        self.sa_module2 = SetAbstractionModule(input_dim=67,
                                               output_dims=[64, 64, 128],
                                               num_centroid=256,
                                               num_sample=32,
                                               radius=0.2)
        self.sa_module3 = SetAbstractionModule(input_dim=131,
                                               output_dims=[128, 128, 256],
                                               num_centroid=64,
                                               num_sample=32,
                                               radius=0.4)
        self.sa_module4 = SetAbstractionModule(input_dim=259,
                                               output_dims=[256, 256, 512],
                                               num_centroid=16,
                                               num_sample=32,
                                               radius=0.8)

        # FP module
        self.fp_module1 = FeaturePropagationModule(input_dim=768, output_dims=[256, 256])
        self.fp_module2 = FeaturePropagationModule(input_dim=384, output_dims=[256, 256])
        self.fp_module3 = FeaturePropagationModule(input_dim=320, output_dims=[256, 128])
        self.fp_module4 = FeaturePropagationModule(input_dim=134, output_dims=[128, 128, 128])

        # classifier
        layers = create_conv1d_blocks(128, 128, kernel_size=1, dropout=0.5)
        layers.append(('conv2', nn.Conv1d(128, self.num_class, kernel_size=1)))
        self.classifier = nn.Sequential(OrderedDict(layers))

    def forward(self, points, features):
        batch_size, _, num_point = points.shape

        # backbone
        points1, features1 = self.sa_module1(points, features)
        points2, features2 = self.sa_module2(points1, features1)
        points3, features3 = self.sa_module3(points2, features2)
        points4, features4 = self.sa_module4(points3, features3)

        features3 = self.fp_module1(points3, features3, points4, features4)
        features2 = self.fp_module2(points2, features2, points3, features3)
        features1 = self.fp_module3(points1, features1, points2, features2)
        features = self.fp_module4(points, features, points1, features1)

        # classifier
        outputs = self.classifier(features)

        return outputs


def create_model(num_class):
    return PointNet2(num_class)


if __name__ == '__main__':
    model = create_model(13)
    print(model.state_dict().keys())

    model.cuda()
    batch_size = 32
    points = torch.randn(batch_size, 3, 4096).cuda()
    features = torch.randn(batch_size, 6, 4096).cuda()
    outputs = model(points, features)
    print(outputs.shape)
