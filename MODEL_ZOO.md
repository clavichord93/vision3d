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

## ModelNet40

Most of the following results are trained using SGD with cosine annealing and label smoothing. The models are trained for 250 epochs. The first 1024 points in each shape are used for training and testing. Details can be found in the code. The random seed is fixed for reproducibility. However, as `atomicAdd` is used in the CUDA operators, the training results may differ a little.

Notes: 

1. `PointNet w/ TNet` is trained with Adam because it is hard to train the TNet with a large lr.
I tried to use a small lr for only the TNet but it is still worse than `PointNet w/o TNet`.

| Model | Acc@1024 |
| :--- | :---: |
| PointNet w/o T-Net | 89.7 |
| PointNet w/ T-Net | 90.1 |
| PointNet++ SSG | 92.2 |
| PointNet++ MSG | 92.7 |
| DGCNN | 92.8 |

## ShapeNetPart

The training settings follow the same setting as in ModelNet40. For each shape, 2048 points are randomly sampled during training and all points are used for testing.

Note:

1. `PointNet w/ TNet` is trained with SGD as it gives better results than Adam.

| Model | mIoU@inst | mIoU@cat |
| :--- | :---: | :---: |
| PointNet w/ T-Net | 84.5 | 81.3 |
| PointNet++ SSG | 85.2 | 82.9 |
| PointNet++ MSG | 85.2 | 82.8 |
| DGCNN | 85.1 | 82.9 |

# S3DIS

Results on S3DIS will be added soon.
