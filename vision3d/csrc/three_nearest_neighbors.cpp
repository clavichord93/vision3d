#include <torch/extension.h>

#include "util.h"
#include "three_nearest_neighbors.h"

std::vector<at::Tensor> three_nearest_neighbors(at::Tensor unknowns, at::Tensor knows) {
  CHECK_INPUT(unknowns);
  CHECK_IS_FLOAT(unknowns);
  CHECK_INPUT(knows);
  CHECK_IS_FLOAT(knows);

  at::Tensor idx =
      torch::zeros({unknowns.size(0), unknowns.size(2), 3},
                   at::device(unknowns.device()).dtype(at::ScalarType::Long));
  at::Tensor dist2 =
      torch::zeros({unknowns.size(0), unknowns.size(2), 3},
                   at::device(unknowns.device()).dtype(at::ScalarType::Float));

  three_nearest_neighbors_kernel_launcher(unknowns.size(0), unknowns.size(2), knows.size(2),
                           unknowns.data_ptr<float>(), knows.data_ptr<float>(),
                           dist2.data_ptr<float>(), idx.data_ptr<long>());

  return {dist2, idx};
}

at::Tensor three_interpolate(at::Tensor points, at::Tensor idx, at::Tensor weight) {
  CHECK_INPUT(points);
  CHECK_IS_FLOAT(points);
  CHECK_IS_LONG(idx);
  CHECK_INPUT(idx);
  CHECK_IS_FLOAT(weight);
  CHECK_INPUT(weight);

  at::Tensor output =
      torch::zeros({points.size(0), points.size(1), idx.size(1)},
                   at::device(points.device()).dtype(at::ScalarType::Float));

  three_interpolate_kernel_launcher(points.size(0), points.size(1),
                                    points.size(2), idx.size(1),
                                    points.data_ptr<float>(), idx.data_ptr<long>(),
                                    weight.data_ptr<float>(), output.data_ptr<float>());

  return output;
}

at::Tensor three_interpolate_grad(at::Tensor grad_out, at::Tensor idx,
                                  at::Tensor weight, const int m) {
  CHECK_INPUT(grad_out);
  CHECK_IS_FLOAT(grad_out);
  CHECK_INPUT(idx);
  CHECK_IS_LONG(idx);
  CHECK_INPUT(weight);
  CHECK_IS_FLOAT(weight);

  at::Tensor output =
      torch::zeros({grad_out.size(0), grad_out.size(1), m},
                   at::device(grad_out.device()).dtype(at::ScalarType::Float));

  three_interpolate_grad_kernel_launcher(grad_out.size(0), grad_out.size(1),
                                         grad_out.size(2), m, grad_out.data_ptr<float>(),
                                         idx.data_ptr<long>(), weight.data_ptr<float>(),
                                         output.data_ptr<float>());

  return output;
}

