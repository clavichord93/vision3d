#include "cuda_util.h"
#include "gather_by_index.h"

// input: points(b, c, n) idx(b, m)
// output: out(b, c, m)
__global__ void gather_by_index_kernel(int b,
                                       int c,
                                       int n,
                                       int m,
                                       const float *__restrict__ points,
                                       const long *__restrict__ idx,
                                       float *__restrict__ out) {
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    for (int l = blockIdx.y; l < c; l += gridDim.y) {
      for (int j = threadIdx.x; j < m; j += blockDim.x) {
        int a = idx[i * m + j];
        out[(i * c + l) * m + j] = points[(i * c + l) * n + a];
      }
    }
  }
}

void gather_by_index_kernel_launcher(int b,
                                     int c,
                                     int n,
                                     int npoints,
                                     const float *points,
                                     const long *idx,
                                     float *out) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  gather_by_index_kernel<<<dim3(b, c, 1), opt_n_threads(npoints), 0, stream>>>(
      b, c, n, npoints, points, idx, out);

  CUDA_CHECK_ERRORS();
}

// input: grad_out(b, c, m) idx(b, m)
// output: grad_points(b, c, n)
__global__ void gather_by_index_grad_kernel(int b,
                                            int c,
                                            int n,
                                            int m,
                                            const float *__restrict__ grad_out,
                                            const long *__restrict__ idx,
                                            float *__restrict__ grad_points) {
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    for (int l = blockIdx.y; l < c; l += gridDim.y) {
      for (int j = threadIdx.x; j < m; j += blockDim.x) {
        int a = idx[i * m + j];
        atomicAdd(grad_points + (i * c + l) * n + a,
                  grad_out[(i * c + l) * m + j]);
      }
    }
  }
}

void gather_by_index_grad_kernel_launcher(int b,
                                          int c,
                                          int n,
                                          int npoints,
                                          const float *grad_out,
                                          const long *idx,
                                          float *grad_points) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  gather_by_index_grad_kernel<<<dim3(b, c, 1), opt_n_threads(npoints), 0,
                                stream>>>(
      b, c, n, npoints, grad_out, idx, grad_points);

  CUDA_CHECK_ERRORS();
}

// input: points(b, c, n) idx(b, npoints, nsample)
// output: out(b, c, npoints, nsample)
__global__ void group_gather_by_index_kernel(int b,
                                             int c,
                                             int n,
                                             int npoints,
                                             int nsample,
                                             const float *__restrict__ points,
                                             const long *__restrict__ idx,
                                             float *__restrict__ out) {
  int batch_index = blockIdx.x;
  points += batch_index * n * c;
  idx += batch_index * npoints * nsample;
  out += batch_index * npoints * nsample * c;

  const int index = threadIdx.y * blockDim.x + threadIdx.x;
  const int stride = blockDim.y * blockDim.x;
  for (int i = index; i < c * npoints; i += stride) {
    const int l = i / npoints;
    const int j = i % npoints;
    for (int k = 0; k < nsample; ++k) {
      int ii = idx[j * nsample + k];
      out[(l * npoints + j) * nsample + k] = points[l * n + ii];
    }
  }
}

void group_gather_by_index_kernel_launcher(int b,
                                           int c,
                                           int n,
                                           int npoints,
                                           int nsample,
                                           const float *points,
                                           const long *idx,
                                           float *out) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  group_gather_by_index_kernel<<<b, opt_block_config(npoints, c), 0, stream>>>(
      b, c, n, npoints, nsample, points, idx, out);

  CUDA_CHECK_ERRORS();
}

// input: grad_out(b, c, npoints, nsample), idx(b, npoints, nsample)
// output: grad_points(b, c, n)
__global__ void group_gather_by_index_grad_kernel(
    int b,
    int c,
    int n,
    int npoints,
    int nsample,
    const float *__restrict__ grad_out,
    const long *__restrict__ idx,
    float *__restrict__ grad_points) {
  int batch_index = blockIdx.x;
  grad_out += batch_index * npoints * nsample * c;
  idx += batch_index * npoints * nsample;
  grad_points += batch_index * n * c;

  const int index = threadIdx.y * blockDim.x + threadIdx.x;
  const int stride = blockDim.y * blockDim.x;
  for (int i = index; i < c * npoints; i += stride) {
    const int l = i / npoints;
    const int j = i % npoints;
    for (int k = 0; k < nsample; ++k) {
      int ii = idx[j * nsample + k];
      atomicAdd(grad_points + l * n + ii,
                grad_out[(l * npoints + j) * nsample + k]);
    }
  }
}

void group_gather_by_index_grad_kernel_launcher(int b,
                                                int c,
                                                int n,
                                                int npoints,
                                                int nsample,
                                                const float *grad_out,
                                                const long *idx,
                                                float *grad_points) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  group_gather_by_index_grad_kernel<<<b, opt_block_config(npoints, c), 0,
                                      stream>>>(
      b, c, n, npoints, nsample, grad_out, idx, grad_points);

  CUDA_CHECK_ERRORS();
}

