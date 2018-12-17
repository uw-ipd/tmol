#pragma once

#include <ATen/Error.h>
#include <ATen/Tensor.h>
#include "ATen/ScalarType.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "TensorAccessor.h"

#include <map>
#include <string>

namespace tmol {

template <typename ToT>
struct can_view_tensor {
  static const bool value = false;
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
  struct can_view_tensor<ctype> {                    \
    static const bool value = true;                  \
    static const at::ScalarType scalar_type = stype; \
    typedef ctype PType;                             \
  };

FORALL_SCALAR_TYPES_EXCEPT_HALF(SCALAR_VIEW)
#undef SCALAR_VIEW

template <typename T, int N>
struct can_view_tensor<Eigen::Matrix<T, N, 1>> {
  static const bool value = can_view_tensor<T>::value;
  static const at::ScalarType scalar_type = can_view_tensor<T>::scalar_type;
  typedef typename can_view_tensor<T>::PType PType;
};

template <typename T, int N>
struct can_view_tensor<Eigen::AlignedBox<T, N>> {
  static const bool value = can_view_tensor<T>::value;
  static const at::ScalarType scalar_type = can_view_tensor<T>::scalar_type;
  typedef typename can_view_tensor<T>::PType PType;
};

template <
    typename T,
    int N,
    template <typename U> class PtrTraits = DefaultPtrTraits,
    typename std::enable_if<can_view_tensor<T>::value>::type* = nullptr>
tmol::TView<T, N, PtrTraits> view_tensor(at::Tensor input_t) {
  typedef typename can_view_tensor<T>::PType FromT;

  auto input = input_t.accessor<FromT, N>();

  int64_t sizes[N];
  int64_t strides[N];
  static_assert(
      sizeof(T) % sizeof(FromT) == 0,
      "Cast target type must be even multiple size of source type.");

  int64_t stride_factor = sizeof(T) / sizeof(FromT);

  AT_ASSERTM(
      input.size(N - 1) % stride_factor == 0,
      "Low-dimension shape must be even multiple of adjusted stride.")
  AT_ASSERTM(input.stride(N - 1) == 1, "Must be c-contiguous.")

  for (int d = 0; d < N - 1; ++d) {
    sizes[d] = input.size(d);
    strides[d] = input.stride(d) / stride_factor;
  }

  sizes[N - 1] = input.size(N - 1) / stride_factor;
  strides[N - 1] = 1;

  return tmol::TView<T, N, PtrTraits>(
      reinterpret_cast<T*>(input_t.data_ptr()), sizes, strides);
}

template <
    typename T,
    int N,
    template <typename U> class PtrTraits = DefaultPtrTraits,
    typename std::enable_if<can_view_tensor<T>::value>::type* = nullptr>
tmol::TView<T, N, PtrTraits> view_tensor(at::Tensor tensor, std::string name) {
  try {
    return view_tensor<T, N, PtrTraits>(tensor);
  } catch (at::Error err) {
    AT_ERROR(
        "Error viewing tensor '" + name + "': " + err.what_without_backtrace());
  }
}

template <
    typename T,
    int N,
    template <typename U> class PtrTraits = DefaultPtrTraits,
    typename std::enable_if<can_view_tensor<T>::value>::type* = nullptr>
tmol::TView<T, N, PtrTraits> view_tensor(
    std::map<std::string, at::Tensor> input_map, std::string member) {
  auto member_t = input_map.find(member);

  AT_ASSERTM(
      member_t != input_map.end(),
      "Map does not contain key: '" + member + "'");

  try {
    return view_tensor<T, N, PtrTraits>(member_t->second, member);
  } catch (at::Error err) {
    AT_ERROR(
        "Error viewing tensor map member '" + member
        + "': " + err.what_without_backtrace());
  }
}

}  // namespace tmol
