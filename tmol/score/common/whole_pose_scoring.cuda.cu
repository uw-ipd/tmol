#include <algorithm>
#include <type_traits>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace tmol {
namespace score {
namespace common {
namespace {

template <typename Real>
__device__ Real rounded_product(Real a, Real b);
template <>
__device__ float rounded_product(float a, float b) {
  return __fmul_rn(a, b);
}
template <>
__device__ double rounded_product(double a, double b) {
  return __dmul_rn(a, b);
}
template <typename Real>
__device__ Real rounded_sum(Real a, Real b);
template <>
__device__ float rounded_sum(float a, float b) {
  return __fadd_rn(a, b);
}
template <>
__device__ double rounded_sum(double a, double b) {
  return __dadd_rn(a, b);
}

template <typename Real, int NScoreTypes>
__global__ void weighted_gradient_sum(
    Real const* derivatives,
    Real const* weights,
    Real* output,
    int64_t n_values,
    int64_t values_per_pose,
    int64_t weight_type_stride,
    int64_t weight_pose_stride) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n_values;
       i += int64_t(blockDim.x) * gridDim.x) {
    int64_t const pose = i / values_per_pose;
    // Match Torch's four-accumulator reduction for this short, strided axis.
    // Explicit rounding prevents contraction of the separate multiply and sum.
    Real partial[4] = {0, 0, 0, 0};
#pragma unroll
    for (int channel = 0; channel < NScoreTypes; ++channel) {
      Real const product = rounded_product(
          derivatives[channel * n_values + i],
          weights[channel * weight_type_stride + pose * weight_pose_stride]);
      partial[channel % 4] = rounded_sum(partial[channel % 4], product);
    }
    output[i] = rounded_sum(
        rounded_sum(rounded_sum(partial[0], partial[1]), partial[2]),
        partial[3]);
  }
}

}  // namespace

at::Tensor accumulate_whole_pose_gradients_cuda(
    at::Tensor const& saved_grad, at::Tensor const& score_grad) {
  c10::cuda::CUDAGuard guard(saved_grad.device());
  auto result =
      at::empty({saved_grad.size(1), saved_grad.size(2)}, saved_grad.options());
  int64_t const n_values = result.numel();
  int64_t const values_per_pose = n_values / score_grad.size(1);
  int const blocks =
      static_cast<int>(std::min<int64_t>((n_values + 255) / 256, 65535));
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      saved_grad.scalar_type(), "whole_pose_gradients", [&] {
        auto launch = [&](auto channels) {
          weighted_gradient_sum<scalar_t, decltype(channels)::value>
              <<<blocks, 256, 0, stream>>>(
                  saved_grad.data_ptr<scalar_t>(),
                  score_grad.data_ptr<scalar_t>(),
                  result.data_ptr<scalar_t>(),
                  n_values,
                  values_per_pose,
                  score_grad.stride(0),
                  score_grad.stride(1));
        };
        switch (saved_grad.size(0)) {
          case 2:
            launch(std::integral_constant<int, 2>{});
            break;
          case 3:
            launch(std::integral_constant<int, 3>{});
            break;
          case 4:
            launch(std::integral_constant<int, 4>{});
            break;
          case 5:
            launch(std::integral_constant<int, 5>{});
            break;
          default:
            TORCH_INTERNAL_ASSERT(false, "unsupported score channel count");
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return result;
}

}  // namespace common
}  // namespace score
}  // namespace tmol
