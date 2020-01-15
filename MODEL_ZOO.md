# Model Zoo

Current supported models include:

- PointNet (w/ and w/o TNet)
- PointNet++ (SSG and MSG)
- DGCNN

More models will be supported in the future:

- [ ] PAT
- [ ] PointConv
- [ ] PointCNN
- [ ] InterpCNN
- [ ] PointWeb

The random seed is fixed for reproducibility.
However, as `atomicAdd` is used in the CUDA operators, the training results may differ a little.

## ModelNet40

Most of the following results are trained using SGD with cosine annealing and label smoothing for 250 epochs.

Notes: 

1. `PointNet w/ TNet` is trained with Adam because it is hard to train the TNet with a large lr.
I tried to use a small lr for only the TNet but it is still worse than `PointNet w/o TNet`.

| Model | #Points | Acc |
| :--- | :---: | :---: |
| PointNet w/o T-Net | 1024 | 89.7 |
| PointNet w/ T-Net | 1024 | 90.1 |
| PointNet++ SSG | 1024 | 92.2 |
| PointNet++ MSG | 1024 | 92.7 |
| DGCNN | 1024 | 92.8 |

## ShapeNetPart

The training settings follow the same setting as in ModelNet40.

Note:

1. `PointNet w/ TNet` is trained with SGD as it gives better results than Adam.

| Model | #Points | mIoU@inst | mIoU@cat |
| :--- | :---: | :---: | :---: |
| PointNet w/ T-Net | 2048 | 84.5 | 81.3 |
| PointNet++ SSG | 2048 | 85.2 | 82.9 |
| PointNet++ MSG | 2048 | 85.2 | 82.8 |
| DGCNN | 2048 | 85.1 | 82.9 |

# S3DIS

The training settings follow the same setting as in ModelNet40.

| Model | #Points | OA@Area_5 | mIoU@Area_5 |
| :--- | :---: | :---: | :---: |
| PointNet w/o T-Net | 4096 | 79.0 | 43.4 |
| PointNet++ SSG | 4096 | 84.2 | 55.0 |
| DGCNN | 4096 |  |  |
