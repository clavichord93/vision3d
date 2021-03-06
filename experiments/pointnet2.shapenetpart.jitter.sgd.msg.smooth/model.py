from collections import OrderedDict

import torch
import torch.nn as nn

from vision3d.modules.pointnet2 import (MultiScaleSetAbstractionModule,
                                        GlobalAbstractionModule,
                                        FeaturePropagationModule)
from vision3d.utils.pytorch_utils import create_conv1d_blocks


class PointNet2(nn.Module):
    def __init__(self, num_class, num_part):
        super(PointNet2, self).__init__()
        self.num_class = num_class
        self.num_part = num_part

        # SA module
        self.ms_sa_module1 = MultiScaleSetAbstractionModule(
            input_dim=6,
            output_dims_list=[[32, 32, 64], [64, 64, 128], [64, 96, 128]],
            num_centroid=512,
            num_samples=[32, 64, 128],
            radii=[0.1, 0.2, 0.4]
        )
        self.ms_sa_module2 = MultiScaleSetAbstractionModule(
            input_dim=323,
            output_dims_list=[[128, 128, 256], [128, 196, 256]],
            num_centroid=128,
            num_samples=[64, 128],
            radii=[0.4, 0.8]
        )
        self.gsa_module = GlobalAbstractionModule(input_dim=515, output_dims=[256, 512, 1024])

        # FP module
        self.fp_module1 = FeaturePropagationModule(input_dim=1536, output_dims=[256, 256])
        self.fp_module2 = FeaturePropagationModule(input_dim=576, output_dims=[256, 128])
        self.fp_module3 = FeaturePropagationModule(input_dim=150, output_dims=[128, 128])

        # classifier
        layers = create_conv1d_blocks(128, 128, kernel_size=1, dropout=0.5)
        layers.append(('conv2', nn.Conv1d(128, num_part, kernel_size=1)))
        self.classifier = nn.Sequential(OrderedDict(layers))

    def forward(self, points, features, class_ids):
        batch_size, _, num_point = points.shape
        one_hot = nn.functional.one_hot(class_ids, self.num_class).float()

        # backbone
        points1, features1 = self.ms_sa_module1(points, features)
        points2, features2 = self.ms_sa_module2(points1, features1)
        points3, features3 = self.gsa_module(points2, features2)

        features2 = self.fp_module1(points2, features2, points3, features3)
        features1 = self.fp_module2(points1, features1, points2, features2)
        one_hot = one_hot.unsqueeze(2).repeat(1, 1, num_point)
        features = torch.cat([one_hot, features, points], dim=1)
        features = self.fp_module3(points, features, points1, features1)

        # classifier
        outputs = self.classifier(features)

        return outputs


def create_model(num_class, num_part):
    return PointNet2(num_class, num_part)


if __name__ == '__main__':
    model = create_model(16, 50)
    print(model.state_dict().keys())

    model.cuda()
    points = torch.randn(2, 3, 2048).cuda()
    normals = torch.randn(2, 3, 2048).cuda()
    class_ids = torch.randint(high=16, size=(2,)).cuda()
    pred = model(points, normals, class_ids)
    print(pred.shape)
