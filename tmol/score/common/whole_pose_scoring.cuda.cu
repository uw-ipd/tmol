#include <tmol/utility/tensor/native_cuda.hh>
#include <tmol/utility/function_dispatch/aten.hh>
#include <algorithm>
#include <type_traits>

namespace tmol {
namespace score {
namespace common {
namespace {

template <typename Real>
__device__ Real rounded_product(Real a, Real b) {
  if constexpr (std::is_same_v<Real, float>) return __fmul_rn(a, b);
  return __dmul_rn(a, b);
}
template <typename Real>
__device__ Real rounded_sum(Real a, Real b) {
  if constexpr (std::is_same_v<Real, float>) return __fadd_rn(a, b);
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
    if constexpr (NScoreTypes == 1) {
      output[i] =
          rounded_product(derivatives[i], weights[pose * weight_pose_stride]);
      continue;
    }
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

template <typename Real>
struct ScoreGradientInputs {
  Real const* derivatives[16];
  int64_t offsets[16], channels[16];
  int count;
};
template <typename Real>
__global__ void score_gradient_sum(
    ScoreGradientInputs<Real> terms,
    Real const* weights,
    Real const* upstream,
    Real* output,
    int64_t n_values,
    int64_t values_per_pose,
    int64_t weight_stride,
    int64_t upstream_stride) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n_values;
       i += int64_t(blockDim.x) * gridDim.x) {
    int64_t pose = i / values_per_pose;
    Real total = 0;
    for (int term = terms.count - 1; term >= 0; --term) {
      Real partial[4] = {0, 0, 0, 0};
      for (int channel = 0; channel < terms.channels[term]; ++channel) {
        Real scale = rounded_product(
            weights[(terms.offsets[term] + channel) * weight_stride],
            upstream[pose * upstream_stride]);
        Real product = rounded_product(
            terms.derivatives[term][channel * n_values + i], scale);
        partial[channel % 4] = rounded_sum(partial[channel % 4], product);
      }
      Real sum = rounded_sum(
          rounded_sum(rounded_sum(partial[0], partial[1]), partial[2]),
          partial[3]);
      total = rounded_sum(total, sum);
    }
    output[i] = total;
  }
}

}  // namespace

tmol::native::Tensor accumulate_whole_pose_gradients_cuda(
    tmol::native::Tensor const& saved_grad,
    tmol::native::Tensor const& score_grad) {
  native::CUDAGuard guard(saved_grad.device());
  auto result = tmol::native::empty(
      {saved_grad.size(1), saved_grad.size(2)}, saved_grad.options());
  int64_t const n_values = result.numel();
  int64_t const values_per_pose = n_values / score_grad.size(1);
  int const blocks =
      static_cast<int>(std::min<int64_t>((n_values + 255) / 256, 65535));
  auto stream = native::current_cuda_stream();
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
          case 1:
            launch(std::integral_constant<int, 1>{});
            break;
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

tmol::native::Tensor accumulate_score_gradients_cuda(
    std::vector<tmol::native::Tensor> const& derivatives,
    std::vector<int64_t> const& offsets,
    tmol::native::Tensor const& weights,
    tmol::native::Tensor const& upstream) {
  TORCH_CHECK(!derivatives.empty() && derivatives.size() <= 16);
  TORCH_CHECK(offsets.size() == derivatives.size());
  auto const& first = derivatives[0];
  TORCH_CHECK(first.is_cuda() && first.dim() == 3 && first.size(2) == 3);
  TORCH_CHECK(weights.dim() == 2 && weights.size(1) == 1);
  native::CUDAGuard guard(first.device());
  auto result = native::empty({first.size(1), first.size(2)}, first.options());
  int64_t n_values = result.numel();
  TORCH_CHECK(
      upstream.dim() == 1 && upstream.numel() > 0
      && n_values % upstream.numel() == 0);
  auto compatible = [&](native::Tensor const& value) {
    return value.device() == first.device()
           && value.scalar_type() == first.scalar_type();
  };
  TORCH_CHECK(compatible(weights) && compatible(upstream));
  for (size_t i = 0; i < derivatives.size(); ++i) {
    auto const& value = derivatives[i];
    TORCH_CHECK(compatible(value) && value.dim() == 3 && value.is_contiguous());
    TORCH_CHECK(
        value.size(1) == first.size(1) && value.size(2) == first.size(2));
    TORCH_CHECK(
        value.size(0) >= 1 && value.size(0) <= 5 && offsets[i] >= 0
        && offsets[i] + value.size(0) <= weights.size(0));
  }
  if (!n_values) return result;
  auto stream = native::current_cuda_stream();
  AT_DISPATCH_FLOATING_TYPES(first.scalar_type(), "score_gradients", [&] {
    ScoreGradientInputs<scalar_t> inputs{};
    inputs.count = derivatives.size();
    for (size_t i = 0; i < derivatives.size(); ++i) {
      inputs.derivatives[i] = derivatives[i].data_ptr<scalar_t>();
      inputs.offsets[i] = offsets[i];
      inputs.channels[i] = derivatives[i].size(0);
    }
    int blocks =
        static_cast<int>(std::min<int64_t>((n_values + 255) / 256, 65535));
    score_gradient_sum<<<blocks, 256, 0, stream>>>(
        inputs,
        weights.data_ptr<scalar_t>(),
        upstream.data_ptr<scalar_t>(),
        result.data_ptr<scalar_t>(),
        n_values,
        n_values / upstream.numel(),
        weights.stride(0),
        upstream.stride(0));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return result;
}

}  // namespace common
}  // namespace score
}  // namespace tmol
