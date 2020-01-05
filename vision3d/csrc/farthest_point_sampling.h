#pragma once

void farthest_point_sampling_kernel_launcher(int b,
                                             int c,
                                             int n,
                                             int m,
                                             const float* points,
                                             float* tmp,
                                             long* idxs);

at::Tensor farthest_point_sampling(at::Tensor points, const int nsamples);
