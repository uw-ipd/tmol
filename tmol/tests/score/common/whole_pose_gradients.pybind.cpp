#include <torch/extension.h>
#include <tmol/score/common/whole_pose_scoring.hh>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("accumulate", &tmol::score::common::accumulate_whole_pose_gradients);
}
