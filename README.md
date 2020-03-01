# Vision3D: A 3D Vision Library built with PyTorch

Vision3D is a 3D vision library built with PyTorch, in order to reproduce the state-of-the-art 3D vision models.

Current supported models are:

1. PointNet (w/ and w/o TNet).
2. PointNet++ (SSG and MSG).
3. DGCNN.

Current supported datasets are:

1. ModelNet40.
2. ShapeNetPart.
3. S3DIS.

## Installation

Vision3D is tested on Python 3.7, PyTorch 1.3.1, Ubuntu 18.04, CUDA 10.2 and cuDNN 7.6.5, but it should work with other configurations. Currently, Vision3d only support training and testing on GPUs.

Install Vision3D with the following command:

```bash
python setup.py build develop
```

All requirements will be automatically installed.

## Data Preparation

### ModelNet40

Vision3D uses pre-processed ModelNet40 dataset from PointNet++.
Download the dataset from [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip).
Modify `data_root` in `tools/datasets/modelnet40.py` and run the file for further pre-processing.
The script will generate 5 hdf5 files for training and 2 hdf5 files for testing with the corresponding id json files.
You can also use the dataset from [here](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip).


### ShapeNetPart

Vision3D uses pre-processed ShapeNetPart dataset with normal from PointNet++.
Download the dataset from [here](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip).
Modify `data_root` in `tools/datasets/shapenetpart.py` and run the file for further pre-processing.
The script will generate 4 pickle files for train, val, trainval and test, respectively.

### S3DIS

Vision3D supports both on-the-fly training augmentation and preprocessed data.
Before using S3DIS in Vision3D, you need to generate `.npy` files following the instructions in [pointnet](https://github.com/charlesq34/pointnet).
For preprocessed hdf5 training data, download from [here](https://shapenet.cs.stanford.edu/media/indoor3d_sem_seg_hdf5_data.zip).
Whole-scene evaluation should be used for testing.
Modify `data_root` in `tools/datastes/s3dis.py` and run the file to generate whole-scene testing hdf5 files for fast evaluation.

## Usage

The models are located in `experiments` and each model contains five files:

1. `config.py` contains overall configurations of the model.
2. `dataset.py` contains data loaders and data augmentations.
3. `model.py` contains the description of the model.
4. `train.py` contains the training code.
4. `test.py` contains the test code.

Before starting, modify `config.root_dir` and `config.data_root` in `config.py`. Run `train.py` and `test.py` with `--devices` argument to specify the GPU(s) to use. Please refer to the code for details.

Sample training and testing scripts are in `tools`.

## Model Zoo

Please refer to [MODEL_ZOO.md](MODEL_ZOO.md) for details.

## TODO

1. Support more datasets.
    - [x] S3DIS support.
    - [ ] ScanNet support.
2. Support more models.
    - [ ] PointCNN.
    - [ ] InterpCNN.
    - [ ] PointConv.
3. Support more tasks.
    - [ ] Object detection.
    - [ ] Instance segmentation.
4. Support more features.
    - [ ] Multi-process multi-GPU training (`DistributedDataParallel`).

## Acknowledgements

- [pointnet](https://github.com/charlesq34/pointnet)
- [pointnet2](https://github.com/charlesq34/pointnet2)
- [Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch)
- [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)
- [dgcnn](https://github.com/WangYueFt/dgcnn)
- [kaolin](https://github.com/NVIDIAGameWorks/kaolin)
- [TorchSeg](https://github.com/ycszen/TorchSeg)

## References

1. Charles R. Qi, Hao Su, Kaichun Mo, Leonidas J. Guibas. *PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation*. CVPR 2017.
2. Charles R. Qi, Li Yi, Hao Su, and Leonidas J. Guibas. *PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space*. NIPS 2017.
3. Yue Wang, Yongbin Sun, Ziwei Liu, Sanjay E. Sarma, Michael M. Bronstein, and Justin M. Solomon. *Dynamic Graph CNN for Learning on Point Clouds*. TOG.
