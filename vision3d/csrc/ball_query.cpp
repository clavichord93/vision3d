#include <torch/extension.h>

#include "ball_query.h"
#include "util.h"

at::Tensor ball_query_v1(at::Tensor new_points,
                         at::Tensor points,
                         const float radius,
                         const int nsample) {
  CHECK_INPUT(new_points);
  CHECK_IS_FLOAT(new_points);
  CHECK_INPUT(points);
  CHECK_IS_FLOAT(points);

  at::Tensor idx = torch::zeros(
      {new_points.size(0), new_points.size(2), nsample},
      at::device(new_points.device()).dtype(at::ScalarType::Long));

  ball_query_v1_kernel_launcher(points.size(0),
                                points.size(1),
                                points.size(2),
                                new_points.size(2),
                                radius,
                                nsample,
                                new_points.data_ptr<float>(),
                                points.data_ptr<float>(),
                                idx.data_ptr<long>());

  return idx;
}

at::Tensor ball_query_v2(int seed,
                         at::Tensor new_points,
                         at::Tensor points,
                         const float radius,
                         const int nsample) {
  CHECK_INPUT(new_points);
  CHECK_IS_FLOAT(new_points);
  CHECK_INPUT(points);
  CHECK_IS_FLOAT(points);

  at::Tensor idx = torch::zeros(
      {new_points.size(0), new_points.size(2), nsample},
      at::device(new_points.device()).dtype(at::ScalarType::Long));

  ball_query_v2_kernel_launcher(seed,
                                points.size(0),
                                points.size(1),
                                points.size(2),
                                new_points.size(2),
                                radius,
                                nsample,
                                new_points.data_ptr<float>(),
                                points.data_ptr<float>(),
                                idx.data_ptr<long>());

  return idx;
}

