#include <torch/extension.h>

#include "ball_query.h"
#include "farthest_point_sampling.h"
#include "gather_by_index.h"
#include "three_nearest_neighbors.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ball_query_v1", &ball_query_v1,
        "Ball Query (v1)");
  m.def("ball_query_v2", &ball_query_v2,
        "Ball Query With Random Sampling (v2)");

  m.def("group_gather_by_index", &group_gather_by_index,
        "Group Gather Points By Index");
  m.def("group_gather_by_index_grad", &group_gather_by_index_grad,
        "Group Gather Points By Index (Gradient)");

  m.def("farthest_point_sampling", &farthest_point_sampling,
        "Farthest Point Sampling");

  m.def("gather_by_index", &gather_by_index,
        "Gather Points By Index");
  m.def("gather_by_index_grad", &gather_by_index_grad,
        "Gather Points By Index (Gradient)");

  m.def("three_nearest_neighbors", &three_nearest_neighbors,
        "3 Nearest Neighbors");
  m.def("three_interpolate", &three_interpolate,
        "3 Nearest Neighbors Interpolate");
  m.def("three_interpolate_grad", &three_interpolate_grad,
        "3 Nearest Neighbors Interpolate (Gradient)");
}
