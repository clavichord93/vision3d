#include "cuda_util.h"
#include "farthest_point_sampling.h"

__device__ void __update(float* __restrict__ dists,
                         int* __restrict__ dists_i,
                         int idx1,
                         int idx2) {
  const float v1 = dists[idx1], v2 = dists[idx2];
  const int i1 = dists_i[idx1], i2 = dists_i[idx2];
  dists[idx1] = max(v1, v2);
  dists_i[idx1] = v2 > v1 ? i2 : i1;
}

// Input points: (b, c, n), tmp: (b, n)
// Ouput idxs (b, m)
template <unsigned int block_size>
__global__ void farthest_point_sampling_kernel(int b,
                                               int c,
                                               int n,
                                               int m,
                                               const float* __restrict__ points,
                                               float* __restrict__ tmp,
                                               long* __restrict__ idxs) {
  if (m <= 0) {
    return;
  }

  __shared__ float dists[block_size];
  __shared__ int dists_i[block_size];

  int batch_index = blockIdx.x;
  points += batch_index * c * n;
  tmp += batch_index * n;
  idxs += batch_index * m;

  int tid = threadIdx.x;
  const int stride = block_size;

  int old = 0;
  if (threadIdx.x == 0)
    idxs[0] = old;

  __syncthreads();
  for (int j = 1; j < m; j++) {
    int besti = 0;
    float best = -1;
    for (int k = tid; k < n; k += stride) {
      float mag = 0;
      float d = 0;
      for (int i = 0; i < c; ++i) {
        float point2 = points[i * n + k];
        float delta = point2 - points[i * n + old];
        mag += point2 * point2;
        d += delta * delta;
      }

      if (mag <= 1e-3)
        continue;

      float d2 = min(d, tmp[k]);
      tmp[k] = d2;
      besti = d2 > best ? k : besti;
      best = d2 > best ? d2 : best;
    }
    dists[tid] = best;
    dists_i[tid] = besti;
    __syncthreads();

    if (block_size >= 512) {
      if (tid < 256) {
        __update(dists, dists_i, tid, tid + 256);
      }
      __syncthreads();
    }
    if (block_size >= 256) {
      if (tid < 128) {
        __update(dists, dists_i, tid, tid + 128);
      }
      __syncthreads();
    }
    if (block_size >= 128) {
      if (tid < 64) {
        __update(dists, dists_i, tid, tid + 64);
      }
      __syncthreads();
    }
    if (block_size >= 64) {
      if (tid < 32) {
        __update(dists, dists_i, tid, tid + 32);
      }
      __syncthreads();
    }
    if (block_size >= 32) {
      if (tid < 16) {
        __update(dists, dists_i, tid, tid + 16);
      }
      __syncthreads();
    }
    if (block_size >= 16) {
      if (tid < 8) {
        __update(dists, dists_i, tid, tid + 8);
      }
      __syncthreads();
    }
    if (block_size >= 8) {
      if (tid < 4) {
        __update(dists, dists_i, tid, tid + 4);
      }
      __syncthreads();
    }
    if (block_size >= 4) {
      if (tid < 2) {
        __update(dists, dists_i, tid, tid + 2);
      }
      __syncthreads();
    }
    if (block_size >= 2) {
      if (tid < 1) {
        __update(dists, dists_i, tid, tid + 1);
      }
      __syncthreads();
    }

    old = dists_i[0];
    if (tid == 0)
      idxs[j] = old;
  }
}

void farthest_point_sampling_kernel_launcher(int b,
                                             int c,
                                             int n,
                                             int m,
                                             const float* points,
                                             float* tmp,
                                             long* idxs) {
  unsigned int n_threads = opt_n_threads(n);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  switch (n_threads) {
  case 512:
    farthest_point_sampling_kernel<512><<<b, n_threads, 0, stream>>>(
        b, c, n, m, points, tmp, idxs);
    break;
  case 256:
    farthest_point_sampling_kernel<256><<<b, n_threads, 0, stream>>>(
        b, c, n, m, points, tmp, idxs);
    break;
  case 128:
    farthest_point_sampling_kernel<128><<<b, n_threads, 0, stream>>>(
        b, c, n, m, points, tmp, idxs);
    break;
  case 64:
    farthest_point_sampling_kernel<64><<<b, n_threads, 0, stream>>>(
        b, c, n, m, points, tmp, idxs);
    break;
  case 32:
    farthest_point_sampling_kernel<32><<<b, n_threads, 0, stream>>>(
        b, c, n, m, points, tmp, idxs);
    break;
  case 16:
    farthest_point_sampling_kernel<16><<<b, n_threads, 0, stream>>>(
        b, c, n, m, points, tmp, idxs);
    break;
  case 8:
    farthest_point_sampling_kernel<8><<<b, n_threads, 0, stream>>>(
        b, c, n, m, points, tmp, idxs);
    break;
  case 4:
    farthest_point_sampling_kernel<4><<<b, n_threads, 0, stream>>>(
        b, c, n, m, points, tmp, idxs);
    break;
  case 2:
    farthest_point_sampling_kernel<2><<<b, n_threads, 0, stream>>>(
        b, c, n, m, points, tmp, idxs);
    break;
  case 1:
    farthest_point_sampling_kernel<1><<<b, n_threads, 0, stream>>>(
        b, c, n, m, points, tmp, idxs);
    break;
  default:
    farthest_point_sampling_kernel<512><<<b, n_threads, 0, stream>>>(
        b, c, n, m, points, tmp, idxs);
  }

  CUDA_CHECK_ERRORS();
}

