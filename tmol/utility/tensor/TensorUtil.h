#include <tmol/utility/tensor/native.hh>
#pragma once

#include <array>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>

namespace tmol {

inline bool operator==(const tmol::Device& lhs, const native::DeviceType& rhs) {
  if (lhs == tmol::Device::CPU) {
    return rhs == native::DeviceType::CPU;
  } else if (lhs == tmol::Device::CUDA) {
    return rhs == native::DeviceType::CUDA;
  } else {
    AT_ERROR("Unknown tmol::Device type.");
  }
};

inline bool operator==(const native::DeviceType& lhs, const tmol::Device& rhs) {
  return rhs == lhs;
};

template <typename ToT>
struct enable_tensor_view {
  static const bool enabled = false;
};

// bool size is ostensibly implementation dependent
static_assert(sizeof(bool) == 1);

#define FORALL_SCALAR_TYPES_EXCEPT_HALF(_)    \
  _(bool, tmol::native::ScalarType::Bool)     \
  _(uint8_t, tmol::native::ScalarType::Byte)  \
  _(int8_t, tmol::native::ScalarType::Char)   \
  _(int16_t, tmol::native::ScalarType::Short) \
  _(int, tmol::native::ScalarType::Int)       \
  _(int64_t, tmol::native::ScalarType::Long)  \
  _(float, tmol::native::ScalarType::Float)   \
  _(double, tmol::native::ScalarType::Double)

#define SCALAR_VIEW(ctype, stype)                                   \
  template <>                                                       \
  struct enable_tensor_view<ctype> {                                \
    static const bool enabled = true;                               \
    static tmol::native::ScalarType scalar_type() { return stype; } \
    static const int nconsumed_dims = 0;                            \
    static int consumed_dims(int) { return 0; }                     \
    typedef ctype PrimitiveType;                                    \
  };

FORALL_SCALAR_TYPES_EXCEPT_HALF(SCALAR_VIEW)
#undef SCALAR_VIEW

// Eigen Matrix/Vector Conversions
// Matrix is defined as <T, M, N>, consume two [M, N] dimensions.
// Vector is defined as <T, M, 1>, consume one [M] dimension.
template <typename T, int M, int N>
struct enable_tensor_view<Eigen::Matrix<T, M, N>> {
  static const bool enabled = enable_tensor_view<T>::enabled;
  static const tmol::native::ScalarType scalar_type() {
    return enable_tensor_view<T>::scalar_type();
  }
  static constexpr int nconsumed_dims = ((N > 1) ? 2 : 1);
  static int consumed_dims(int i) {
    if (i == 0) {
      return M;
    } else if (i == 1 && N > 1) {
      return N;
    } else {
      return 0;
    }
  }
  typedef typename enable_tensor_view<T>::PrimitiveType PrimitiveType;
};

template <typename T, int N>
struct enable_tensor_view<Eigen::AlignedBox<T, N>> {
  static const bool enabled = enable_tensor_view<T>::enabled;
  static const tmol::native::ScalarType scalar_type() {
    return enable_tensor_view<T>::scalar_type();
  }
  static const int nconsumed_dims = 1;
  static int consumed_dims(int i) { return (i == 0) ? N : 0; }
  typedef typename enable_tensor_view<T>::PrimitiveType PrimitiveType;
};

template <
    typename T,
    size_t N,
    Device D,
    PtrTag P = PtrTag::Restricted,
    typename std::enable_if<enable_tensor_view<T>::enabled>::type* = nullptr>
auto view_tensor(tmol::native::Tensor input_t) -> tmol::TView<T, N, D, P> {
  typedef typename enable_tensor_view<T>::PrimitiveType FromT;
  int64_t stride_factor = sizeof(T) / sizeof(FromT);

  constexpr int nconsumed_dims = enable_tensor_view<T>::nconsumed_dims;
  auto consumed_dims = enable_tensor_view<T>::consumed_dims;

  TORCH_CHECK(
      input_t.dim() == N + nconsumed_dims,
      "view_tensor of wrong dimensionality.",
      " dim: ",
      input_t.dim(),
      " expected: ",
      N + nconsumed_dims);
  for (int d = N; d < input_t.dim(); ++d) {
    TORCH_CHECK(
        input_t.size(d) == consumed_dims(d - N),
        "squeezed dimension mismatch in view_tensor.",
        " d: ",
        d,
        " size: ",
        input_t.size(d),
        " expected: ",
        consumed_dims(d - N));
  }
  for (int d = N; d < input_t.dim(); ++d) {
    TORCH_CHECK(
        input_t.stride(d) != 0,
        "stride of zero for view_tensor is incompatible for consumed "
        "dimension. Did torch.sum() or torch.expand() yeild a tensor with a "
        "stride of 0?",
        " d: ",
        d);
  }

  // All classes w/ an associated enable_tensor_view must
  // consume contiguous blocks of memory
  int64_t target_stride = 1;
  for (int d = input_t.dim() - 1; d >= N; --d) {
    TORCH_CHECK(
        input_t.stride(d) == target_stride,
        " stride for input tensor at dimension ",
        d,
        " must match the target stride of ",
        target_stride,
        " to ensure that memory is allocated in a ",
        "contiguous block, but a stride of ",
        input_t.stride(d),
        " was found instead.");
    target_stride *= input_t.size(d);
  }

  TORCH_CHECK(
      input_t.device().type() == D,
      "view_tensor of incorrect device type.",
      " device: ",
      static_cast<int>(input_t.device().type()));

  TORCH_CHECK(
      input_t.scalar_type() == enable_tensor_view<FromT>::scalar_type(),
      "view_tensor dtype mismatch");

  int64_t sizes[N];
  int64_t strides[N];

  for (int d = 0; d < N; ++d) {
    sizes[d] = input_t.size(d);
    strides[d] = input_t.stride(d) / stride_factor;
  }

  return tmol::TView<T, N, D, P>(
      reinterpret_cast<T*>(input_t.data_ptr()), sizes, strides);
};

template <
    typename T,
    size_t N,
    Device D,
    PtrTag P = PtrTag::Restricted,
    typename std::enable_if<enable_tensor_view<T>::enabled>::type* = nullptr>
auto view_tensor(tmol::native::Tensor tensor, const std::string& name)
    -> tmol::TView<T, N, D, P> {
  try {
    return view_tensor<T, N, D, P>(tensor);
  } catch (tmol::native::Error err) {
    AT_ERROR("Error viewing tensor '" + name + "': " + err.what());
  }
}

}  // namespace tmol
