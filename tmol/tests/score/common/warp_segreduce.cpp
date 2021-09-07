#include <torch/extension.h>
#include <Eigen/Core>

#include <tmol/utility/tensor/TensorCast.h>

#include <tmol/tests/score/common/warp_segreduce.hh>




torch::Tensor
warp_segreduce_1(
  torch::Tensor values,
  torch::Tensor flags
)
{
  auto result = warp_segreduce_gpu(
    tmol::TCAST(values),
    tmol::TCAST(flags)
  );
  return result.tensor;
}

torch::Tensor
warp_segreduce_2(
  torch::Tensor values,
  torch::Tensor flags
)
{
  auto result = warp_segreduce_gpu2(
    tmol::TCAST(values),
    tmol::TCAST(flags)
  );
  return result.tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "warp_segreduce_1", &warp_segreduce_1, "warp segreduce 1.");
  m.def(
      "warp_segreduce_2", &warp_segreduce_2, "warp segreduce 2.");
}
  
