#pragma once

void ball_query_v1_kernel_launcher(int b, int n, int m, float radius,
                                   int nsample, const float *new_xyz,
                                   const float *xyz, long *idx);

void ball_query_v2_kernel_launcher(int seed, int b, int n, int m,
                                   float radius, int nsample,
                                   const float *new_xyz,
                                   const float *xyz, long *idx);

at::Tensor ball_query_v1(at::Tensor new_xyz, at::Tensor xyz,
                         const float radius, const int nsample);

at::Tensor ball_query_v2(int seed, at::Tensor new_xyz, at::Tensor xyz,
                         const float radius, const int nsample);

