#pragma once

#include <ATen/Tensor.h>
#include "TensorAccessor.h"

namespace tmol {

template <typename ToT, typename FromT, int N>
tmol::PackedTensorAccessor<ToT, N> reinterpret_tensor(at::Tensor input_t) {
  auto input = input_t.accessor<FromT, N>();

  int64_t sizes[N];
  int64_t strides[N];
  static_assert(
      sizeof(ToT) % sizeof(FromT) == 0,
      "Cast target type must be even multiple size of source type.");

  int64_t stride_factor = sizeof(ToT) / sizeof(FromT);

  AT_ASSERT(
      input.size(N - 1) % stride_factor == 0,
      "Low-dimension shape must be even multiple of adjusted stride.")
  AT_ASSERT(input.stride(N - 1) == 1, "Must be c-contiguous.")

  for (int d = 0; d < N - 1; ++d) {
    sizes[d] = input.size(d);
    strides[d] = input.stride(d) / stride_factor;
  }

  sizes[N - 1] = input.size(N - 1) / stride_factor;
  strides[N - 1] = 1;

  return tmol::PackedTensorAccessor<ToT, N>(
      reinterpret_cast<ToT*>(input_t.data_ptr()), sizes, strides);
}

}  // namespace tmol
