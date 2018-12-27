#pragma once

#include <ATen/Error.h>
#include <ATen/ScalarType.h>
#include <ATen/Tensor.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>

#include <map>
#include <string>

namespace tmol {

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
    template <typename U> class PtrTraits = DefaultPtrTraits,
    typename std::enable_if<enable_tensor_view<T>::enabled>::type* = nullptr>
tmol::TView<T, N, PtrTraits> _view_tensor(at::Tensor input_t) {
  typedef typename enable_tensor_view<T>::PrimitiveType FromT;

  static_assert(
      sizeof(T) % sizeof(FromT) == 0,
      "Cast target type must be even multiple size of source type.");

  int64_t stride_factor = sizeof(T) / sizeof(FromT);

  AT_ASSERTM(
      input_t.size(N - 1) % stride_factor == 0,
      "Low-dimension shape must be even multiple of adjusted stride.")
  AT_ASSERTM(input_t.stride(N - 1) == 1, "Must be c-contiguous.")

  auto input = input_t.accessor<FromT, N>();

  int64_t sizes[N];
  int64_t strides[N];

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
    typename std::enable_if<enable_tensor_view<T>::enabled>::type* = nullptr>
tmol::TView<T, N, PtrTraits> view_tensor(at::Tensor input_t) {
  typedef typename enable_tensor_view<T>::PrimitiveType FromT;
  int64_t stride_factor = sizeof(T) / sizeof(FromT);

  if (input_t.dim() == N + 1 && input_t.size(N) == stride_factor) {
    // Implicitly convert an input tensor of result dims [..., 1]
    // into a dim-1 view, squeezing off the last dimension.
    auto full_view = _view_tensor<T, N + 1, PtrTraits>(input_t);

    AT_ASSERTM(
        full_view.size(N) == 1, "Expected low-dimension result shape 1.");

    return tmol::TView<T, N, PtrTraits>(
        full_view.data(), &full_view.size(0), &full_view.stride(0));
  } else {
    return _view_tensor<T, N, PtrTraits>(input_t);
  }
};

template <
    typename T,
    int N,
    template <typename U> class PtrTraits = DefaultPtrTraits,
    typename std::enable_if<enable_tensor_view<T>::enabled>::type* = nullptr>
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
    typename std::enable_if<enable_tensor_view<T>::enabled>::type* = nullptr>
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

namespace pybind11 {
namespace detail {

template <typename T, size_t N, template <typename U> class P>
struct type_caster<tmol::TView<T, N, P>> {
 public:
  typedef tmol::TView<T, N, P> ViewType;
  PYBIND11_TYPE_CASTER(ViewType, _<ViewType>());

  bool load(handle src, bool convert) {
    type_caster<at::Tensor> conv;

    if (!conv.load(src, convert)) {
      return false;
    }

    try {
      value = tmol::view_tensor<T, N, P>(conv);
      return true;
    } catch (at::Error err) {
      return false;
    }
  }

  // C++ -> Python cast operation not supported.
};

}  // namespace detail
}  // namespace pybind11
