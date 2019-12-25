# Vision3D: A 3D Vision Library built with PyTorch

Vision3D is a 3D vision library built with PyTorch, in order to reproduce the state-of-the-art 3D vision models.

Current supported models are:

1. PointNet (w/ and w/o TNet).
2. PointNet++ (SSG and MSG).
3. DGCNN.

Current supported datasets are:

1. ModelNet40. Download from [here](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip).
2. ShapeNetPart. Download from [here](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip).

## Installation

Vision3D is tested on Python 3.7, PyTorch 1.3.1 and Ubuntu 18.04 with CUDA 10.2 and cuDNN 7.6.5, but it should work with other configurations.

Install Vision3D with the following command:

```bash
python setup.py build develop
```

All requirements will be automatically installed.

## Usage

The models are located in `experiments` and each model contains five files:

1. `config.py` contains overall configurations of the model.
2. `dataset.py` contains data loaders and data augmentations.
3. `model.py` contains the description of the model.
4. `train.py` contains the training code.
4. `test.py` contains the test code.

Run `train.py` and `test.py` with `--device` argument to specify the GPU(s) to use.
Currently CUDA devices are required to run the code.

## Model Zoo

The following results are trained using SGD with cosine annealing and label smoothing.
The models are trained for 250 epochs.
Details can be found in the code.
The random seed is fixed to ensure reproducibility.
However, as `atomicAdd` is used in the CUDA operators, the training results may differ a little.

Note: `PointNet w/ TNet` on ModelNet40 is trained with Adam because it is hard to train the TNet with a large lr.
I tried to use a small lr for only the TNet but it is still worse than `PointNet w/o TNet`.
However, training `PointNet w/ TNet` on ShapeNetPart with SGD is better than Adam.

| Model | ModelNet40 | ShapeNetPart<br>(mIoU@inst, mIoU@cat) |
| :--- | :---: | :---: |
| PointNet w/o T-Net | 89.7 | - |
| PointNet w/ T-Net | 90.1 | 84.5, 81.3 |
| PointNet++ SSG | 92.2 | 85.2, 82.9 |
| PointNet++ MSG | 92.7 | 85.2, 82.8 |
| DGCNN | 92.8 | 85.1, 82.9 |

## TODO

- [ ] S3DIS support.
- [ ] ScanNet support.
- [ ] PointCNN.
- [ ] InterpCNN.
- [ ] PointConv.
- [ ] Multi-process multi-GPU training.
- [ ] Object detection.
- [ ] Instance segmentation.

## Acknowledgement

- [pointnet](https://github.com/charlesq34/pointnet)
- [pointnet2](https://github.com/charlesq34/pointnet2)
- [Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch)
- [dgcnn](https://github.com/WangYueFt/dgcnn)
- [kaolin](https://github.com/NVIDIAGameWorks/kaolin)

## References

1. Charles R. Qi, Hao Su, Kaichun Mo, Leonidas J. Guibas. *PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation*. CVPR 2017.
2. Charles R. Qi, Li Yi, Hao Su, and Leonidas J. Guibas. *PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space*. NIPS 2017.
3. Yue Wang, Yongbin Sun, Ziwei Liu, Sanjay E. Sarma, Michael M. Bronstein, and Justin M. Solomon. *Dynamic Graph CNN for Learning on Point Clouds*. TOG.
