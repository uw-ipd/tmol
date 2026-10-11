#pragma once
#ifdef TORCH_TARGET_VERSION
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/util/Exception.h>
#include <memory>
#include <stdexcept>
#include <vector>

namespace tmol::native {
namespace stable = torch::stable;
using ScalarType = torch::headeronly::ScalarType;
using Device = stable::Device;
using DeviceType = torch::headeronly::DeviceType;
using IntList = torch::headeronly::IntHeaderOnlyArrayRef;
using Error = std::runtime_error;
constexpr auto kCPU = torch::headeronly::DeviceType::CPU;
constexpr auto kCUDA = torch::headeronly::DeviceType::CUDA;
constexpr auto kInt32 = ScalarType::Int;
constexpr auto kFloat32 = ScalarType::Float;
constexpr auto kFloat64 = ScalarType::Double;

struct TensorOptions {
  ScalarType scalar = ScalarType::Float;
  Device target{kCPU};
  TensorOptions dtype(ScalarType value) const {
    auto o = *this;
    o.scalar = value;
    return o;
  }
  ScalarType dtype() const { return scalar; }
  TensorOptions device(Device value) const {
    auto o = *this;
    o.target = value;
    return o;
  }
  Device device() const { return target; }
};

inline ScalarType scalar_type_from_options(const TensorOptions& o) {
  return o.dtype();
}

struct Tensor {
  stable::Tensor value;
  bool needs_grad = false;
  // Python interop keeps the original Tensor for saved-variable version checks.
  std::shared_ptr<std::shared_ptr<void>> original =
      std::make_shared<std::shared_ptr<void>>();
  Tensor() = default;
  Tensor(stable::Tensor tensor) : value(std::move(tensor)) {}
  bool defined() const { return value.defined(); }
  int64_t dim() const { return value.dim(); }
  int64_t size(int64_t d) const { return value.size(d); }
  int64_t stride(int64_t d) const { return value.stride(d); }
  int64_t numel() const { return value.numel(); }
  auto sizes() const { return value.sizes(); }
  auto strides() const { return value.strides(); }
  auto device() const { return value.device(); }
  auto scalar_type() const { return value.scalar_type(); }
  bool requires_grad() const { return needs_grad; }
  TensorOptions options() const {
    return TensorOptions().dtype(scalar_type()).device(device());
  }
  void* data_ptr() const { return value.mutable_data_ptr(); }
  template <class T>
  const T* const_data_ptr() const {
    return value.const_data_ptr<T>();
  }
  template <class T>
  T* data_ptr() const {
    return value.mutable_data_ptr<T>();
  }
  Tensor reshape(IntList shape) const;
  Tensor select(int64_t d, int64_t i) const;
  Tensor narrow(int64_t d, int64_t start, int64_t length) const;
  bool is_cuda() const { return device().is_cuda(); }
  bool is_contiguous() const { return value.is_contiguous(); }
  bool is_neg() const {
    return false;
  }  // DLPack imports resolve lazy negative views.
  Tensor contiguous() const;
  Tensor unsqueeze(int64_t d) const;
  Tensor squeeze(int64_t d) const;
  template <class T>
  T item() const {
    static_assert(std::is_same_v<T, int64_t>);
    int64_t result;
    STABLE_TORCH_ERROR_CODE_CHECK(aoti_torch_item_int64(value.get(), &result));
    return result;
  }
  Tensor sum(ScalarType dtype) const {
    return stable::sum(value, std::nullopt, false, dtype);
  }
  Tensor sum(IntList dims) const;
  Tensor sum(int64_t d) const { return sum(IntList(&d, 1)); }
  Tensor index_select(int64_t d, const Tensor& index) const;
  Tensor to(Device device) const {
    return stable::to(value, scalar_type(), std::nullopt, device);
  }
};
inline Tensor empty(IntList shape, TensorOptions options) {
  return stable::empty(shape, options.scalar, std::nullopt, options.target);
}
template <class S>
Tensor full(IntList shape, S fill, TensorOptions options) {
  Tensor result = empty(shape, options);
  stable::fill_(result.value, fill);
  return result;
}
inline Tensor zeros(IntList shape, TensorOptions options) {
  return full(shape, 0, options);
}
inline Tensor ones(IntList shape, TensorOptions options) {
  return full(shape, 1, options);
}
Tensor operator*(const Tensor& a, const Tensor& b);
template <class F>
void parallel_for(int64_t begin, int64_t end, int64_t grain, F fn) {
  STABLE_TORCH_ERROR_CODE_CHECK(torch_parallel_for(
      begin,
      end,
      grain,
      [](int64_t b, int64_t e, void* ctx) { (*static_cast<F*>(ctx))(b, e); },
      &fn));
}
inline Tensor random_seed_tensor(IntList shape, Device device) {
  auto result =
      empty(shape, TensorOptions().dtype(ScalarType::Long).device(device));
  // Use the dispatcher to acquire/advance Torch's generator under its own lock.
  // This also preserves CUDA Graph replay semantics without GeneratorImpl ABI.
  std::array<StableIValue, 4> stack{
      stable::detail::from(result.value),
      stable::detail::from(std::numeric_limits<int64_t>::min()),
      stable::detail::from(std::optional<int64_t>()),
      stable::detail::from(std::optional<stable::Tensor>())};
  STABLE_TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::random_", "from", stack.data(), TORCH_ABI_VERSION));
  // The dispatcher transfers an owning tensor result back through the stack.
  return stable::detail::to<stable::Tensor>(stack[0]);
}
struct GradMode {
  static bool is_enabled() { return aoti_torch_grad_mode_is_enabled(); }
};
}  // namespace tmol::native
#ifndef TORCH_CHECK
#define TORCH_CHECK STD_TORCH_CHECK
#define TORCH_INTERNAL_ASSERT STD_TORCH_CHECK
#define AT_ASSERTM STD_TORCH_CHECK
#define AT_ERROR(...) STD_TORCH_CHECK(false, __VA_ARGS__)
#endif

#else
// JIT and test-only extensions are built against their local Torch runtime.
// Published modules always define TORCH_TARGET_VERSION and use the C ABI above.
#include <torch/torch.h>
#include <torch/script.h>
#include <ATen/Parallel.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/macros.h>
#include <torch/headeronly/util/shim_utils.h>
namespace tmol::native {
using at::Device;
using at::empty;
using at::Error;
using at::full;
using at::IntList;
using at::ones;
using at::parallel_for;
using at::ScalarType;
using at::Tensor;
using at::TensorOptions;
using at::zeros;
using c10::DeviceType;
using torch::GradMode;
using torch::kCPU;
using torch::kCUDA;
using torch::kFloat32;
using torch::kFloat64;
using torch::kInt32;
namespace autograd = torch::autograd;
inline ScalarType scalar_type_from_options(const TensorOptions& o) {
  return c10::typeMetaToScalarType(o.dtype());
}
inline Tensor random_seed_tensor(IntList shape, Device device) {
  return at::empty(shape, at::TensorOptions().dtype(at::kLong).device(device))
      .random_(
          std::numeric_limits<int64_t>::min(),
          std::optional<int64_t>(),
          std::nullopt);
}
}  // namespace tmol::native
#endif
