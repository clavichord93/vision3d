#include <torch/extension.h>

#include "ball_query.h"
#include "util.h"

at::Tensor ball_query_v1(at::Tensor new_xyz, at::Tensor xyz, const float radius,
                         const int nsample) {
  CHECK_INPUT(new_xyz);
  CHECK_IS_FLOAT(new_xyz);
  CHECK_INPUT(xyz);
  CHECK_IS_FLOAT(xyz);

  at::Tensor idx =
      torch::zeros({new_xyz.size(0), new_xyz.size(2), nsample},
                   at::device(new_xyz.device()).dtype(at::ScalarType::Long));

  ball_query_v1_kernel_launcher(xyz.size(0), xyz.size(2), new_xyz.size(2),
                                radius, nsample, new_xyz.data_ptr<float>(),
                                xyz.data_ptr<float>(), idx.data_ptr<long>());

  return idx;
}

at::Tensor ball_query_v2(int seed, at::Tensor new_xyz, at::Tensor xyz,
                         const float radius, const int nsample) {
  CHECK_INPUT(new_xyz);
  CHECK_IS_FLOAT(new_xyz);
  CHECK_INPUT(xyz);
  CHECK_IS_FLOAT(xyz);

  at::Tensor idx =
      torch::zeros({new_xyz.size(0), new_xyz.size(2), nsample},
                   at::device(new_xyz.device()).dtype(at::ScalarType::Long));

  ball_query_v2_kernel_launcher(
      seed, xyz.size(0), xyz.size(2), new_xyz.size(2), radius, nsample,
      new_xyz.data_ptr<float>(), xyz.data_ptr<float>(), idx.data_ptr<long>());

  return idx;
}

