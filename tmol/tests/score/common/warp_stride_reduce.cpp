#include <torch/extension.h>
#include <Eigen/Core>

#include <tmol/utility/tensor/TensorCast.h>

#include <tmol/tests/score/common/warp_stride_reduce.hh>




torch::Tensor
warp_stride_reduce_1(
  torch::Tensor values,
  int stride
)
{
  auto result = warp_stride_reduce_gpu(
    tmol::TCAST(values),
    stride
  );
  return result.tensor;
}

torch::Tensor
warp_stride_reduce_2(
  torch::Tensor values,
  int stride
)
{
  auto result = warp_stride_reduce_gpu2(
    tmol::TCAST(values),
    stride
  );
  return result.tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "warp_stride_reduce_1", &warp_stride_reduce_1, "warp segreduce 1.");
  m.def(
      "warp_stride_reduce_2", &warp_stride_reduce_2, "warp segreduce 2.");
}
  
