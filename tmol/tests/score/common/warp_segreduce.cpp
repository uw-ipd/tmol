#include <torch/extension.h>
#include <Eigen/Core>

#include <tmol/utility/tensor/TensorCast.h>

#include <tmol/tests/score/common/warp_segreduce.hh>

torch::Tensor warp_segreduce_full(torch::Tensor values, torch::Tensor flags) {
  auto result =
      gpu_warp_segreduce_full(tmol::TCAST(values), tmol::TCAST(flags));
  return result.tensor;
}

torch::Tensor warp_segreduce_full_vec3(
    torch::Tensor values, torch::Tensor flags) {
  auto result =
      gpu_warp_segreduce_full_vec3(tmol::TCAST(values), tmol::TCAST(flags));
  return result.tensor;
}

torch::Tensor warp_segreduce_vec3_benchmark(
    torch::Tensor values, torch::Tensor flags, int n_repeats) {
  auto result = gpu_warp_segreduce_vec3_benchmark(
      tmol::TCAST(values), tmol::TCAST(flags), n_repeats);
  return result.tensor;
}

torch::Tensor warp_segreduce_partial(
    torch::Tensor values, torch::Tensor flags) {
  auto result =
      gpu_warp_segreduce_partial(tmol::TCAST(values), tmol::TCAST(flags));
  return result.tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("warp_segreduce_full", &warp_segreduce_full, "warp segreduce full");
  m.def(
      "warp_segreduce_full_vec3",
      &warp_segreduce_full_vec3,
      "warp segreduce full vec3");
  m.def(
      "warp_segreduce_vec3_benchmark",
      &warp_segreduce_vec3_benchmark,
      "warp segreduce vec3 benchmark");
  m.def(
      "warp_segreduce_partial",
      &warp_segreduce_partial,
      "warp segreduce partial");
}
