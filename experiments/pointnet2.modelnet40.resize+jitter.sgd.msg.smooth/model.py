from collections import OrderedDict

import torch
import torch.nn as nn

from vision3d.modules.pointnet2 import MultiScaleSetAbstractionModule, GlobalAbstractionModule
from vision3d.utils.pytorch_utils import create_linear_blocks


class PointNet2(nn.Module):
    def __init__(self, num_class):
        super(PointNet2, self).__init__()

        # SA module
        self.ms_sa_module1 = MultiScaleSetAbstractionModule(
            input_dim=3,
            output_dims_list=[[32, 32, 64], [64, 64, 128], [64, 96, 128]],
            num_centroid=512,
            num_samples=[16, 32, 128],
            radii=[0.1, 0.2, 0.4]
        )
        self.ms_sa_module2 = MultiScaleSetAbstractionModule(
            input_dim=323,
            output_dims_list=[[64, 64, 128], [128, 128, 256], [128, 128, 256]],
            num_centroid=128,
            num_samples=[32, 64, 128],
            radii=[0.2, 0.4, 0.8]
        )
        self.g_sa_module = GlobalAbstractionModule(input_dim=643, output_dims=[256, 512, 1024])

        # classifier
        layers = create_linear_blocks(1024, [512, 256], dropout=0.5)
        layers.append(('fc3', nn.Linear(256, num_class)))
        self.classifier = nn.Sequential(OrderedDict(layers))

    def forward(self, points):
        # backbone
        points, features = self.ms_sa_module1(points, None)
        points, features = self.ms_sa_module2(points, features)
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
