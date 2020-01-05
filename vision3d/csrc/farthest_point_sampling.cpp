#include <torch/extension.h>

#include "util.h"
#include "farthest_point_sampling.h"

at::Tensor farthest_point_sampling(at::Tensor points, const int nsamples) {
  CHECK_INPUT(points);
  CHECK_IS_FLOAT(points);

  at::Tensor output = torch::zeros(
      {points.size(0), nsamples},
      at::device(points.device()).dtype(at::ScalarType::Long));

  at::Tensor tmp = torch::full(
      {points.size(0), points.size(2)},
      1e10,
      at::device(points.device()).dtype(at::ScalarType::Float));

  farthest_point_sampling_kernel_launcher(points.size(0),
                                          points.size(1),
                                          points.size(2),
                                          nsamples,
                                          points.data_ptr<float>(),
                                          tmp.data_ptr<float>(),
                                          output.data_ptr<long>());

  return output;
}

