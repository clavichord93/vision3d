#pragma once

void gather_by_index_kernel_launcher(int b,
                                     int c,
                                     int n,
                                     int npoints,
                                     const float *points,
                                     const long *idx,
                                     float *out);

void gather_by_index_grad_kernel_launcher(int b,
                                          int c,
                                          int n,
                                          int npoints,
                                          const float *grad_out,
                                          const long *idx,
                                          float *grad_points);

void group_gather_by_index_kernel_launcher(int b,
                                           int c,
                                           int n,
                                           int npoints,
                                           int nsample,
                                           const float *points,
                                           const long *idx,
                                           float *out);

void group_gather_by_index_grad_kernel_launcher(int b,
                                                int c,
                                                int n,
                                                int npoints,
                                                int nsample,
                                                const float *grad_out,
                                                const long *idx,
                                                float *grad_points);

at::Tensor gather_by_index(at::Tensor points, at::Tensor idx);

at::Tensor gather_by_index_grad(at::Tensor grad_out,
                                at::Tensor idx,
                                const int n);

at::Tensor group_gather_by_index(at::Tensor points, at::Tensor idx);

at::Tensor group_gather_by_index_grad(at::Tensor grad_out,
                                      at::Tensor idx,
                                      const int n);

