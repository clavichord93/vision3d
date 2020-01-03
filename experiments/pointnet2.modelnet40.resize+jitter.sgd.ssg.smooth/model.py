from collections import OrderedDict

import torch
import torch.nn as nn

from vision3d.modules.pointnet2 import SetAbstractionModule, GlobalAbstractionModule
from vision3d.utils.pytorch_utils import create_linear_blocks


class PointNet2(nn.Module):
    def __init__(self, num_class):
        super(PointNet2, self).__init__()

        # SA module
        self.sa_module1 = SetAbstractionModule(input_dim=3,
                                               output_dims=[64, 64, 128],
                                               num_centroid=512,
                                               num_sample=32,
                                               radius=0.2)
        self.sa_module2 = SetAbstractionModule(input_dim=131,
                                               output_dims=[128, 128, 256],
                                               num_centroid=128,
                                               num_sample=64,
                                               radius=0.4)
        self.g_sa_module = GlobalAbstractionModule(input_dim=259, output_dims=[256, 512, 1024])

        # classifier
        layers = create_linear_blocks(1024, [512, 256], dropout=0.5)
        layers.append(('fc3', nn.Linear(256, num_class)))
        self.classifier = nn.Sequential(OrderedDict(layers))

    def forward(self, points):
        # backbone
        points, features = self.sa_module1(points, None)
        points, features = self.sa_module2(points, features)
        points, features = self.g_sa_module(points, features)

        # classifier
        features = features.squeeze(dim=2)
        outputs = self.classifier(features)

        return outputs


def create_model(num_class):
    return PointNet2(num_class)


if __name__ == '__main__':
    model = create_model(40).cuda()
    print(model)
    print(model.state_dict().keys())

    model.cuda()
    points = torch.rand(32, 3, 1024).cuda()
    pred = model(points)
    print(pred.shape)
