#pragma once

void three_nearest_neighbors_kernel_launcher(int b,
                                             int c,
                                             int n,
                                             int m,
                                             const float* unknown,
                                             const float* known,
                                             float* dist2,
                                             long* idx);

void three_interpolate_kernel_launcher(int b,
                                       int c,
                                       int m,
                                       int n,
                                       const float* points,
                                       const long* idx,
                                       const float* weight,
                                       float* out);

void three_interpolate_grad_kernel_launcher(int b,
                                            int c,
                                            int n,
                                            int m,
                                            const float* grad_out,
                                            const long* idx,
                                            const float* weight,
                                            float* grad_points);

std::vector<at::Tensor> three_nearest_neighbors(at::Tensor unknown,
                                                at::Tensor known);

at::Tensor three_interpolate(at::Tensor points,
                             at::Tensor idx,
                             at::Tensor weight);

at::Tensor three_interpolate_grad(at::Tensor grad_out,
                                  at::Tensor idx,
                                  at::Tensor weight,
                                  const int m);

