from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='vision3d',
    version='0.0.1',
    install_requires=[
        'torch',
        'tqdm',
        'coloredlogs',
        'easydict',
        'numpy',
        'h5py'
    ],
    ext_modules=[
        CUDAExtension('vision3d.ext', [
            'vision3d/csrc/ball_query.cpp',
            'vision3d/csrc/ball_query_cuda.cu',
            'vision3d/csrc/farthest_point_sampling.cpp',
            'vision3d/csrc/farthest_point_sampling_cuda.cu',
            'vision3d/csrc/gather_by_index.cpp',
            'vision3d/csrc/gather_by_index_cuda.cu',
            'vision3d/csrc/three_nearest_neighbors.cpp',
            'vision3d/csrc/three_nearest_neighbors_cuda.cu',
            'vision3d/csrc/extensions.cpp'
        ])
    ],
    cmdclass={'build_ext': BuildExtension}
)
