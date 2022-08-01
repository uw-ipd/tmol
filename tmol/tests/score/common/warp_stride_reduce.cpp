#include <torch/extension.h>
#include <Eigen/Core>

#include <tmol/utility/tensor/TensorCast.h>

#include <tmol/tests/score/common/warp_stride_reduce.hh>

torch::Tensor warp_stride_reduce_full(torch::Tensor values, int stride) {
  auto result = gpu_warp_stride_reduce_full(tmol::TCAST(values), stride);
  return result.tensor;
}

torch::Tensor warp_stride_reduce_full_vec3(torch::Tensor values, int stride) {
  auto result = gpu_warp_stride_reduce_full_vec3(tmol::TCAST(values), stride);
  return result.tensor;
}

torch::Tensor warp_stride_reduce_partial(torch::Tensor values, int stride) {
  auto result = gpu_warp_stride_reduce_partial(tmol::TCAST(values), stride);
  return result.tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "warp_stride_reduce_full",
      &warp_stride_reduce_full,
      "warp stride reduce full.");
  m.def(
      "warp_stride_reduce_full_vec3",
      &warp_stride_reduce_full_vec3,
      "warp stride reduce full vec3.");
  m.def(
      "warp_stride_reduce_partial",
      &warp_stride_reduce_partial,
      "warp stride reduce partial.");
}
