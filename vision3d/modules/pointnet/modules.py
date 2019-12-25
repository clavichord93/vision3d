from collections import OrderedDict

import torch
import torch.nn as nn

from vision3d.utils.pytorch_utils import create_conv1d_blocks, create_fc_blocks, SmoothCrossEntropyLoss


class TNet(nn.Module):
    def __init__(self, input_dim, output_dims1, output_dims2):
        super(TNet, self).__init__()
        self.input_dim = input_dim

        layers = create_conv1d_blocks(input_dim, output_dims1, kernel_size=1)
        self.shared_mlp = nn.Sequential(OrderedDict(layers))

        layers = create_fc_blocks(output_dims1[-1], output_dims2)
        self.mlp = nn.Sequential(OrderedDict(layers))

        self.weight = nn.Parameter(torch.zeros(output_dims2[-1], input_dim * input_dim), requires_grad=True)
        self.bias = nn.Parameter(torch.eye(input_dim).flatten(), requires_grad=True)

    def forward(self, points):
        batch_size = points.shape[0]
        points = self.shared_mlp(points)
        points, _ = points.max(dim=2)
        points = self.mlp(points)
        points = torch.matmul(points, self.weight) + self.bias
        points = points.view(batch_size, self.input_dim, self.input_dim)
        return points


class TNetLoss(nn.Module):
    def __init__(self):
        super(TNetLoss, self).__init__()

    def forward(self, transforms):
        assert transforms.dim() == 3, 'The dimension of the transform matrix is not 3!'
        assert transforms.shape[1] == transforms.shape[2], 'The transform matrix must be square!'
        device = transforms.device
        dim = transforms.shape[1]
        identity = torch.eye(dim).to(device)
        transforms = identity - torch.matmul(transforms, transforms.transpose(1, 2))
        loss = torch.sum(transforms ** 2) / 2
        return loss


class PointNetLoss(nn.Module):
    def __init__(self, alpha=0.001, eps=None):
        super(PointNetLoss, self).__init__()
        self.tnet_loss = TNetLoss()
        if eps is None:
            self.cls_loss = nn.CrossEntropyLoss()
        else:
            self.cls_loss = SmoothCrossEntropyLoss(eps=eps)
        self.alpha = alpha

    def forward(self, *inputs):
        outputs, labels, transforms = inputs
        cls_loss = self.cls_loss(outputs, labels)
        tnet_loss = self.alpha * self.tnet_loss(transforms)
        loss = cls_loss + tnet_loss
        return loss, cls_loss, tnet_loss