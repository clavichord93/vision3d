#include <torch/extension.h>

#include "util.h"
#include "gather_by_index.h"

at::Tensor gather_by_index(at::Tensor points, at::Tensor idx) {
  CHECK_INPUT(points);
  CHECK_IS_FLOAT(points);
  CHECK_INPUT(idx);
  CHECK_IS_LONG(idx);

  at::Tensor output = torch::zeros(
      {points.size(0), points.size(1), idx.size(1)},
      at::device(points.device()).dtype(at::ScalarType::Float));

  gather_by_index_kernel_launcher(points.size(0),
                                  points.size(1),
                                  points.size(2),
                                  idx.size(1),
                                  points.data_ptr<float>(),
                                  idx.data_ptr<long>(),
                                  output.data_ptr<float>());

  return output;
}

at::Tensor gather_by_index_grad(at::Tensor grad_out,
                                at::Tensor idx,
                                const int n) {
  CHECK_INPUT(grad_out);
  CHECK_IS_FLOAT(grad_out);
  CHECK_INPUT(idx);
  CHECK_IS_LONG(idx);

  at::Tensor output = torch::zeros(
      {grad_out.size(0), grad_out.size(1), n},
      at::device(grad_out.device()).dtype(at::ScalarType::Float));

  gather_by_index_grad_kernel_launcher(grad_out.size(0),
                                       grad_out.size(1),
                                       n,
                                       idx.size(1),
                                       grad_out.data_ptr<float>(),
                                       idx.data_ptr<long>(),
                                       output.data_ptr<float>());

  return output;
}

at::Tensor group_gather_by_index(at::Tensor points, at::Tensor idx) {
  CHECK_INPUT(points);
  CHECK_IS_FLOAT(points);
  CHECK_INPUT(idx);
  CHECK_IS_LONG(idx);

  at::Tensor output = torch::zeros(
      {points.size(0), points.size(1), idx.size(1), idx.size(2)},
      at::device(points.device()).dtype(at::ScalarType::Float));

  group_gather_by_index_kernel_launcher(points.size(0),
                                        points.size(1),
                                        points.size(2),
                                        idx.size(1),
                                        idx.size(2),
                                        points.data_ptr<float>(),
                                        idx.data_ptr<long>(),
                                        output.data_ptr<float>());

  return output;
}

at::Tensor group_gather_by_index_grad(at::Tensor grad_out,
                                      at::Tensor idx,
                                      const int n) {
  CHECK_INPUT(grad_out);
  CHECK_IS_FLOAT(grad_out);
  CHECK_INPUT(idx);
  CHECK_IS_LONG(idx);

  at::Tensor output = torch::zeros(
      {grad_out.size(0), grad_out.size(1), n},
      at::device(grad_out.device()).dtype(at::ScalarType::Float));

  group_gather_by_index_grad_kernel_launcher(grad_out.size(0),
                                             grad_out.size(1),
                                             n,
                                             idx.size(1),
                                             idx.size(2),
                                             grad_out.data_ptr<float>(),
                                             idx.data_ptr<long>(),
                                             output.data_ptr<float>());

  return output;
}

