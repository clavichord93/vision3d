#pragma once

void ball_query_v1_kernel_launcher(int b,
                                   int c,
                                   int n,
                                   int m,
                                   float radius,
                                   int nsample,
                                   const float* new_points,
                                   const float* points,
                                   long* idx);

void ball_query_v2_kernel_launcher(int seed,
                                   int b,
                                   int c,
                                   int n,
                                   int m,
                                   float radius,
                                   int nsample,
                                   const float* new_points,
                                   const float* points,
                                   long* idx);

at::Tensor ball_query_v1(at::Tensor new_points,
                         at::Tensor points,
                         const float radius,
                         const int nsample);

at::Tensor ball_query_v2(int seed,
                         at::Tensor new_points,
                         at::Tensor points,
                         const float radius,
                         const int nsample);

