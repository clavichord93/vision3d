#pragma once

void farthest_point_sampling_kernel_launcher(int b, int n, int m,
                                             const float *dataset, float *temp,
                                             long *idxs);

at::Tensor farthest_point_sampling(at::Tensor points, const int nsamples);
