#include <tmol/utility/tensor/native.hh>
#pragma once

#include <stdint.h>
#include <algorithm>
#include <cstddef>
#include <iterator>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorUtil.h>

namespace tmol {

template <typename T, size_t N, Device D, PtrTag P = PtrTag::Restricted>
struct TPack {
  tmol::native::Tensor tensor;
  tmol::TView<T, N, D, P> view;

  TPack(tmol::native::Tensor tensor, tmol::TView<T, N, D, P> view)
      : tensor(tensor), view(view) {};

  TPack(tmol::native::Tensor tensor)
      : tensor(tensor), view(tmol::view_tensor<T, N, D, P>(tensor)) {};

  TPack() : tensor(), view() {};

  int64_t dim() { return view.dim(); }
  int64_t size(int64_t d) { return view.size(d); }
  int64_t stride(int64_t d) { return view.stride(d); }

  // Empty
  template <
      typename std::enable_if<enable_tensor_view<T>::enabled>::type* = nullptr>
  static auto empty(tmol::native::IntList size) -> TPack<T, N, D, P> {
    return _allocate(
        [](tmol::native::IntList size,
           const tmol::native::TensorOptions& options) {
          return tmol::native::empty(size, options);
        },
        size);
  }

  template <
      typename Target,
      typename std::enable_if<enable_tensor_view<T>::enabled>::type* = nullptr>
  static auto empty_like(Target& other) -> TPack<T, N, D, P> {
    return _allocate_like(
        [](tmol::native::IntList size,
           const tmol::native::TensorOptions& options) {
          return tmol::native::empty(size, options);
        },
        other);
  }

  // Ones
  template <
      typename std::enable_if<enable_tensor_view<T>::enabled>::type* = nullptr>
  static auto ones(tmol::native::IntList size) -> TPack<T, N, D, P> {
    return _allocate(
        [](tmol::native::IntList size,
           const tmol::native::TensorOptions& options) {
          return tmol::native::ones(size, options);
        },
        size);
  }

  template <
      typename Target,
      typename std::enable_if<enable_tensor_view<T>::enabled>::type* = nullptr>
  static auto ones_like(Target& other) -> TPack<T, N, D, P> {
    return _allocate_like(
        [](tmol::native::IntList size,
           const tmol::native::TensorOptions& options) {
          return tmol::native::ones(size, options);
        },
        other);
  }

  // Zeros
  template <
      typename std::enable_if<enable_tensor_view<T>::enabled>::type* = nullptr>
  static auto zeros(tmol::native::IntList size) -> TPack<T, N, D, P> {
    return _allocate(
        [](tmol::native::IntList size,
           const tmol::native::TensorOptions& options) {
          return tmol::native::zeros(size, options);
        },
        size);
  }

  template <
      typename Target,
      typename std::enable_if<enable_tensor_view<T>::enabled>::type* = nullptr>
  static auto zeros_like(Target& other) -> TPack<T, N, D, P> {
    return _allocate_like(
        [](tmol::native::IntList size,
           const tmol::native::TensorOptions& options) {
          return tmol::native::zeros(size, options);
        },
        other);
  }

  // Full
  template <
      typename Scalar,
      typename std::enable_if<enable_tensor_view<T>::enabled>::type* = nullptr>
  static auto full(tmol::native::IntList size, Scalar value)
      -> TPack<T, N, D, P> {
    return _allocate(
        [value = value](
            tmol::native::IntList size,
            const tmol::native::TensorOptions& options) {
          return tmol::native::full(size, value, options);
        },
        size);
  }

  template <
      typename Scalar,
      typename Target,
      typename std::enable_if<enable_tensor_view<T>::enabled>::type* = nullptr>
  static auto full_like(Target& other, Scalar value) -> TPack<T, N, D, P> {
    return _allocate_like(
        [value = value](
            tmol::native::IntList size,
            const tmol::native::TensorOptions& options) {
          return tmol::native::full(size, value, options);
        },
        other);
  }

  // Tensor allocation interface
  template <
      typename AllocFun,
      typename std::enable_if<enable_tensor_view<T>::enabled>::type* = nullptr>
  static auto _allocate(AllocFun aten_alloc, tmol::native::IntList size)
      -> TPack<T, N, D, P> {
    typedef typename enable_tensor_view<T>::PrimitiveType BaseT;

    tmol::native::TensorOptions target_type =
        tmol::native::TensorOptions()
            .dtype(enable_tensor_view<T>::scalar_type())
            .device(
                (D == Device::CPU) ? tmol::native::kCPU : tmol::native::kCUDA);

    constexpr int nconsumed_dims = enable_tensor_view<T>::nconsumed_dims;
    auto consumed_dims = enable_tensor_view<T>::consumed_dims;

    tmol::native::Tensor tensor;

    if (nconsumed_dims == 0) {
      tensor = aten_alloc(size, target_type);
    } else {
      std::array<int64_t, N + nconsumed_dims> composite_size;
      for (int i = 0; i < N; i++) {
        composite_size.at(i) = size.at(i);
      }
      for (int i = N; i < N + nconsumed_dims; i++) {
        composite_size.at(i) = consumed_dims(i - N);
      }
      tensor = aten_alloc(composite_size, target_type);
    }

    return TPack<T, N, D, P>(tensor);
  }

  template <
      typename AllocFun,
      typename Target,
      typename std::enable_if<enable_tensor_view<T>::enabled>::type* = nullptr>
  static auto _allocate_like(AllocFun aten_alloc, Target& other)
      -> TPack<T, N, D, P> {
    AT_ASSERTM(
        other.dim() == N, "TPack:::_allocate_like mismatched dimensionality.");

    // Via individual access to dims to support TView. (Does not exposes
    // "sizes", only dim & size.)
    int64_t dims[N];
    for (int i = 0; i < N; i++) {
      dims[i] = other.size(i);
    };

    return _allocate(aten_alloc, dims);
  }
};

}  // namespace tmol
