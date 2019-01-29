#pragma once

#include <ATen/Device.h>
#include <stdint.h>
#include <algorithm>
#include <cstddef>
#include <iterator>

#include <ATen/Functions.h>
#include <ATen/Tensor.h>
#include <torch/torch.h>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorUtil.h>

namespace tmol {

template <typename T, size_t N, Device D, PtrTag P = PtrTag::Restricted>
struct TPack {
  at::Tensor tensor;
  tmol::TView<T, N, D, P> view;

  TPack(at::Tensor tensor, tmol::TView<T, N, D, P> view)
      : tensor(tensor), view(view){};

  TPack(at::Tensor tensor)
      : tensor(tensor), view(view_tensor<T, N, D, P>(tensor)){};

  template <
      typename std::enable_if<enable_tensor_view<T>::enabled>::type* = nullptr>
  static auto empty(at::IntList size) -> TPack<T, N, D, P> {
    return _allocate(
        [](at::IntList size, const at::TensorOptions& options) {
          return at::empty(size, options);
        },
        size);
  }

  template <
      typename std::enable_if<enable_tensor_view<T>::enabled>::type* = nullptr>
  static auto ones(at::IntList size) -> TPack<T, N, D, P> {
    return _allocate(
        [](at::IntList size, const at::TensorOptions& options) {
          return at::ones(size, options);
        },
        size);
  }

  template <
      typename std::enable_if<enable_tensor_view<T>::enabled>::type* = nullptr>
  static auto zeros(at::IntList size) -> TPack<T, N, D, P> {
    return _allocate(
        [](at::IntList size, const at::TensorOptions& options) {
          return at::zeros(size, options);
        },
        size);
  }

  template <
      typename AllocFun,
      typename std::enable_if<enable_tensor_view<T>::enabled>::type* = nullptr>
  static auto _allocate(AllocFun aten_alloc, at::IntList size)
      -> TPack<T, N, D, P> {
    typedef typename enable_tensor_view<T>::PrimitiveType BaseT;

    at::Type& target_type =
        (D == Device::CPU) ? torch::CPU(enable_tensor_view<T>::scalar_type)
                           : torch::CUDA(enable_tensor_view<T>::scalar_type);

    at::Tensor tensor;

    int64_t stride_factor = sizeof(T) / sizeof(BaseT);
    if (stride_factor == 1) {
      // The target type has a primitive layout, construct size N tensor.
      tensor = aten_alloc(size, target_type);
    } else {
      // The target type is composite, construct size N + 1 tensor with implicit
      // minor dimension for the composite type.
      std::array<int64_t, N + 1> composite_size;
      for (int i = 0; i < N; i++) {
        composite_size.at(i) = size.at(i);
      }
      composite_size.at(N) = stride_factor;

      tensor = aten_alloc(composite_size, target_type);
    }

    return {tensor, view_tensor<T, N, D, P>(tensor)};
  }
};

}  // namespace tmol
