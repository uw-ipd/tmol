#pragma once

#include <array>

#include <ATen/Error.h>
#include <ATen/Functions.h>
#include <ATen/ScalarType.h>
#include <ATen/Tensor.h>

#include <torch/torch.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>

namespace tmol {

inline bool operator==(const tmol::Device& lhs, const at::Device::Type& rhs) {
  if (lhs == tmol::Device::CPU) {
    return rhs == at::Device::Type::CPU;
  } else if (lhs == tmol::Device::CUDA) {
    return rhs == at::Device::Type::CUDA;
  } else {
    AT_ERROR("Unknown tmol::Device type.");
  }
};

inline bool operator==(const at::Device::Type& lhs, const tmol::Device& rhs) {
  return rhs == lhs;
};

template <typename ToT>
struct enable_tensor_view {
  static const bool enabled = false;
};

#define FORALL_SCALAR_TYPES_EXCEPT_HALF(_) \
  _(uint8_t, at::ScalarType::Byte)         \
  _(int8_t, at::ScalarType::Char)          \
  _(int16_t, at::ScalarType::Short)        \
  _(int, at::ScalarType::Int)              \
  _(int64_t, at::ScalarType::Long)         \
  _(float, at::ScalarType::Float)          \
  _(double, at::ScalarType::Double)

#define SCALAR_VIEW(ctype, stype)                    \
  template <>                                        \
  struct enable_tensor_view<ctype> {                 \
    static const bool enabled = true;                \
    static const at::ScalarType scalar_type = stype; \
    typedef ctype PrimitiveType;                     \
  };

FORALL_SCALAR_TYPES_EXCEPT_HALF(SCALAR_VIEW)
#undef SCALAR_VIEW

template <>
struct enable_tensor_view<bool> {
  static const bool enabled = enable_tensor_view<uint8_t>::enabled;
  static const at::ScalarType scalar_type =
      enable_tensor_view<uint8_t>::scalar_type;
  typedef typename enable_tensor_view<uint8_t>::PrimitiveType PrimitiveType;
};

template <typename T, int N>
struct enable_tensor_view<Eigen::Matrix<T, N, 1>> {
  static const bool enabled = enable_tensor_view<T>::enabled;
  static const at::ScalarType scalar_type = enable_tensor_view<T>::scalar_type;
  typedef typename enable_tensor_view<T>::PrimitiveType PrimitiveType;
};

template <typename T, int N>
struct enable_tensor_view<Eigen::AlignedBox<T, N>> {
  static const bool enabled = enable_tensor_view<T>::enabled;
  static const at::ScalarType scalar_type = enable_tensor_view<T>::scalar_type;
  typedef typename enable_tensor_view<T>::PrimitiveType PrimitiveType;
};

template <
    typename T,
    int N,
    Device D,
    PtrTag P = PtrTag::Restricted,
    typename std::enable_if<enable_tensor_view<T>::enabled>::type* = nullptr>
auto _view_tensor(at::Tensor input_t) -> tmol::TView<T, N, D, P> {
  typedef typename enable_tensor_view<T>::PrimitiveType FromT;

  static_assert(
      sizeof(T) % sizeof(FromT) == 0,
      "Cast target type must be even multiple size of source type.");

  int64_t stride_factor = sizeof(T) / sizeof(FromT);

  AT_ASSERTM(
      input_t.size(N - 1) % stride_factor == 0,
      "Low-dimension shape must be even multiple of adjusted stride.")

  if (stride_factor != 1) {
    AT_ASSERTM(input_t.stride(N - 1) == 1, "Must be c-contiguous.")
  }

  AT_ASSERTM(
      input_t.device().type() == D, "_view_tensor of incorrect device type.")

  auto input = input_t.accessor<FromT, N>();

  int64_t sizes[N];
  int64_t strides[N];

  for (int d = 0; d < N - 1; ++d) {
    sizes[d] = input.size(d);
    strides[d] = input.stride(d) / stride_factor;
  }

  sizes[N - 1] = input.size(N - 1) / stride_factor;
  if (stride_factor != 1) {
    // If composite type is not same size as input type then is c-contig in
    // minor dimension.
    strides[N - 1] = 1;
  } else {
    // The types are the same size, and can be non-contiguous in minor dim.
    strides[N - 1] = input.stride(N - 1);
  }

  return tmol::TView<T, N, D, P>(
      reinterpret_cast<T*>(input_t.data_ptr()), sizes, strides);
}

template <
    typename T,
    int N,
    Device D,
    PtrTag P = PtrTag::Restricted,
    typename std::enable_if<enable_tensor_view<T>::enabled>::type* = nullptr>
auto view_tensor(at::Tensor input_t) -> tmol::TView<T, N, D, P> {
  typedef typename enable_tensor_view<T>::PrimitiveType FromT;
  int64_t stride_factor = sizeof(T) / sizeof(FromT);

  if (input_t.dim() == N + 1 && input_t.size(N) == stride_factor) {
    // Implicitly convert an input tensor of result dims [..., 1]
    // into a dim-1 view, squeezing off the last dimension.
    auto full_view = _view_tensor<T, N + 1, D, P>(input_t);

    AT_ASSERTM(
        full_view.size(N) == 1, "Expected low-dimension result shape 1.");

    return tmol::TView<T, N, D, P>(
        full_view.data(), &full_view.size(0), &full_view.stride(0));
  } else {
    return _view_tensor<T, N, D, P>(input_t);
  }
};

template <
    typename T,
    int N,
    Device D,
    PtrTag P = PtrTag::Restricted,
    typename std::enable_if<enable_tensor_view<T>::enabled>::type* = nullptr>
auto view_tensor(at::Tensor tensor, std::string name)
    -> tmol::TView<T, N, D, P> {
  try {
    return view_tensor<T, N, D, P>(tensor);
  } catch (at::Error err) {
    AT_ERROR(
        "Error viewing tensor '" + name + "': " + err.what_without_backtrace());
  }
}

}  // namespace tmol
