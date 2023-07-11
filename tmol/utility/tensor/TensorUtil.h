#pragma once

#include <array>

#include <ATen/Functions.h>
#include <ATen/ScalarType.h>
#include <ATen/Tensor.h>

#include <torch/extension.h>

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

// bool size is ostensibly implementation dependent
static_assert(sizeof(bool) == 1);

#define FORALL_SCALAR_TYPES_EXCEPT_HALF(_) \
  _(bool, at::ScalarType::Bool)            \
  _(uint8_t, at::ScalarType::Byte)         \
  _(int8_t, at::ScalarType::Char)          \
  _(int16_t, at::ScalarType::Short)        \
  _(int, at::ScalarType::Int)              \
  _(int64_t, at::ScalarType::Long)         \
  _(float, at::ScalarType::Float)          \
  _(double, at::ScalarType::Double)

#define SCALAR_VIEW(ctype, stype)                         \
  template <>                                             \
  struct enable_tensor_view<ctype> {                      \
    static const bool enabled = true;                     \
    static at::ScalarType scalar_type() { return stype; } \
    static const int nconsumed_dims = 0;                  \
    static int consumed_dims(int) { return 0; }           \
    typedef ctype PrimitiveType;                          \
  };

FORALL_SCALAR_TYPES_EXCEPT_HALF(SCALAR_VIEW)
#undef SCALAR_VIEW

// Eigen Matrix/Vector Conversions
// Matrix is defined as <T, M, N>, consume two [M, N] dimensions.
// Vector is defined as <T, M, 1>, consume one [M] dimension.
template <typename T, int M, int N>
struct enable_tensor_view<Eigen::Matrix<T, M, N>> {
  static const bool enabled = enable_tensor_view<T>::enabled;
  static const at::ScalarType scalar_type() {
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
  static const at::ScalarType scalar_type() {
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
auto view_tensor(at::Tensor input_t) -> tmol::TView<T, N, D, P> {
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
      input_t.device().type());

  auto input = input_t.accessor<FromT, N + nconsumed_dims>();

  int64_t sizes[N];
  int64_t strides[N];

  for (int d = 0; d < N; ++d) {
    sizes[d] = input.size(d);
    strides[d] = input.stride(d) / stride_factor;
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
auto view_tensor(at::Tensor tensor, const std::string& name)
    -> tmol::TView<T, N, D, P> {
  try {
    return view_tensor<T, N, D, P>(tensor);
  } catch (at::Error err) {
    AT_ERROR(
        "Error viewing tensor '" + name + "': " + err.what_without_backtrace());
  }
}

}  // namespace tmol
