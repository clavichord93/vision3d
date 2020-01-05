#include <stdio.h>
#include <stdlib.h>

#include "cuda_util.h"
#include <curand.h>
#include <curand_kernel.h>

#include "ball_query.h"

// input: new_points(b, c, m) points(b, c, n)
// output: idx(b, m, nsample)
__global__ void ball_query_v1_kernel(int b,
                                     int c,
                                     int n,
                                     int m,
                                     float radius,
                                     int nsample,
                                     const float* __restrict__ new_points,
                                     const float* __restrict__ points,
                                     long* __restrict__ idx) {
  int batch_index = blockIdx.x;
  points += batch_index * c * n;
  new_points += batch_index * c * m;
  idx += m * nsample * batch_index;

  int index = threadIdx.x;
  int stride = blockDim.x;

  float radius2 = radius * radius;
  for (int j = index; j < m; j += stride) {
    for (int k = 0, cnt = 0; k < n && cnt < nsample; ++k) {
      float d2 = 0;
      for (int i = 0; i < c; ++i) {
        float delta = new_points[i * m + j] - points[i * n + k];
        d2 += delta * delta;
      }
      if (d2 < radius2) {
        if (cnt == 0) {
          for (int l = 0; l < nsample; ++l) {
            idx[j * nsample + l] = k;
          }
        }
        idx[j * nsample + cnt] = k;
        ++cnt;
      }
    }
  }
}

void ball_query_v1_kernel_launcher(int b,
                                   int c,
                                   int n,
                                   int m,
                                   float radius,
                                   int nsample,
                                   const float* new_points,
                                   const float* points,
                                   long* idx) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  ball_query_v1_kernel<<<b, opt_n_threads(m), 0, stream>>>(
      b, c, n, m, radius, nsample, new_points, points, idx);

  CUDA_CHECK_ERRORS();
}

// input: new_points(b, c, m) points(b, c, n)
// output: idx(b, m, nsample)
__global__ void ball_query_v2_kernel(int seed,
                                     curandState* rand_states,
                                     int b,
                                     int c,
                                     int n,
                                     int m,
                                     float radius,
                                     int nsample,
                                     const float* __restrict__ new_points,
                                     const float* __restrict__ points,
                                     long* __restrict__ idx) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  curandState* local_state = rand_states + id;

  // TODO: optimize: curand_init is slow.
  curand_init(seed, id, 0, local_state);
  // // A potentially faster but less accurate version:
  // curand_init(seed + id * 1337, 0, 0, &rand_states[id]);

  int batch_index = blockIdx.x;
  points += batch_index * c * n;
  new_points += batch_index * c * m;
  idx += m * nsample * batch_index;

  int index = threadIdx.x;
  int stride = blockDim.x;

  float radius2 = radius * radius;
  for (int j = index; j < m; j += stride) {
    for (int k = 0, cnt = 0; k < n; ++k) {
      float d2 = 0;
      for (int i = 0; i < c; ++i) {
        float new_point = new_points[m * i + j];
        float point = points[n * i + k];
        d2 += (new_point - point) * (new_point - point);
      }
      if (d2 < radius2) {
        if (cnt == 0) {
          for (int l = 0; l < nsample; ++l) {
            idx[j * nsample + l] = k;
          }
        } else if (cnt < nsample) {
          idx[j * nsample + cnt] = k;
        } else {
          unsigned int r = curand_uniform(local_state) * (cnt + 1);
          if (r < nsample) {
            idx[j * nsample + r] = k;
          }
        }
        ++cnt;
      }
    }
  }
}

void ball_query_v2_kernel_launcher(int seed,
                                   int b,
                                   int c,
                                   int n,
                                   int m,
                                   float radius,
                                   int nsample,
                                   const float* new_points,
                                   const float* points,
                                   long* idx) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int grid_dim = b;
  int block_dim = opt_n_threads(m);
  int num_threads = grid_dim * block_dim;

  curandState* rand_states;
  cudaMalloc((void**)&rand_states, num_threads * sizeof(curandState));

  ball_query_v2_kernel<<<grid_dim, block_dim, 0, stream>>>(
      seed, rand_states, b, c, n, m, radius, nsample, new_points, points, idx);

  cudaFree(rand_states);

  CUDA_CHECK_ERRORS();
}

