#pragma once
#ifdef TORCH_TARGET_VERSION
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/util/Exception.h>
#include <memory>
#include <optional>
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

// Opaque Python ownership keeps Python headers out of kernel translation units.
struct PythonOwner {
  void* object = nullptr;
  void (*release)(void*) = nullptr;
  ~PythonOwner() {
    if (object) release(object);
  }
};

struct Tensor {
  std::optional<stable::Tensor> value;
  bool needs_grad = false;
  bool defined_ = false;
  void* data_ = nullptr;
  IntList sizes_, strides_;
  ScalarType dtype_ = ScalarType::Float;
  Device device_{kCPU};
  // Resolved negative views retain the original for saved-variable version
  // checks.
  std::shared_ptr<PythonOwner> original;
  Tensor() = default;
  Tensor(stable::Tensor tensor) : value(std::move(tensor)) { cache_metadata(); }
  // Tensor metadata is immutable during a native call. Read it once rather
  // than crossing the C ABI for every accessor and dimension check.
  void cache_metadata() {
    defined_ = value && value->defined();
    if (!defined_) return;
    sizes_ = value->sizes();
    strides_ = value->strides();
    data_ = value->mutable_data_ptr();
    dtype_ = value->scalar_type();
    device_ = value->device();
  }
  stable::Tensor handle() const { return value.value(); }
  bool defined() const { return defined_; }
  int64_t dim() const { return sizes_.size(); }
  int64_t size(int64_t d) const { return sizes_.at(d < 0 ? d + dim() : d); }
  int64_t stride(int64_t d) const { return strides_.at(d < 0 ? d + dim() : d); }
  int64_t numel() const {
    int64_t count = 1;
    for (auto size : sizes_) count *= size;
    return count;
  }
  IntList sizes() const { return sizes_; }
  IntList strides() const { return strides_; }
  Device device() const { return device_; }
  ScalarType scalar_type() const { return dtype_; }
  bool requires_grad() const { return needs_grad; }
  TensorOptions options() const {
    return TensorOptions().dtype(scalar_type()).device(device());
  }
  void* data_ptr() const { return data_; }
  template <class T>
  const T* const_data_ptr() const {
    return data_ptr<T>();
  }
  template <class T>
  T* data_ptr() const {
    STD_TORCH_CHECK(
        scalar_type() == torch::headeronly::CppTypeToScalarType<T>::value,
        "Tensor data pointer requested with a different dtype");
    return static_cast<T*>(data_ptr());
  }
  Tensor reshape(IntList shape) const;
  Tensor select(int64_t d, int64_t i) const;
  Tensor narrow(int64_t d, int64_t start, int64_t length) const;
  bool is_cuda() const { return device().is_cuda(); }
  bool is_contiguous() const {
    if (numel() == 0) return true;
    int64_t expected = 1;
    for (int64_t d = dim() - 1; d >= 0; --d) {
      if (size(d) == 1) continue;
      if (stride(d) != expected) return false;
      expected *= size(d);
    }
    return true;
  }
  bool is_neg() const {
    return false;
  }  // The Negative dispatch path resolves lazy negative views.
  Tensor contiguous() const;
  Tensor unsqueeze(int64_t d) const;
  Tensor squeeze(int64_t d) const;
  template <class T>
  T item() const {
    static_assert(std::is_same_v<T, int64_t>);
    int64_t result;
    STABLE_TORCH_ERROR_CODE_CHECK(
        aoti_torch_item_int64(handle().get(), &result));
    return result;
  }
  Tensor sum(ScalarType dtype) const {
    return stable::sum(handle(), std::nullopt, false, dtype);
  }
  Tensor sum(IntList dims) const;
  Tensor sum(int64_t d) const { return sum(IntList(&d, 1)); }
  Tensor index_select(int64_t d, const Tensor& index) const;
  Tensor to(Device device) const {
    return stable::to(handle(), scalar_type(), std::nullopt, device);
  }
};
inline Tensor empty(IntList shape, TensorOptions options) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t d = static_cast<int64_t>(shape.size()) - 1; d > 0; --d)
    strides[d - 1] = strides[d] * std::max<int64_t>(shape[d], 1);
  AtenTensorHandle result;
  STABLE_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(
      shape.size(),
      shape.data(),
      strides.data(),
      stable::detail::to<int32_t>(stable::detail::from(options.scalar)),
      stable::detail::to<int32_t>(stable::detail::from(options.target.type())),
      options.target.index(),
      &result));
  return stable::Tensor(result);
}
template <class S>
Tensor full(IntList shape, S fill, TensorOptions options) {
  Tensor result = empty(shape, options);
  stable::fill_(*result.value, fill);
  return result;
}
inline Tensor zeros(IntList shape, TensorOptions options) {
  Tensor result = empty(shape, options);
  STABLE_TORCH_ERROR_CODE_CHECK(aoti_torch_zero_(result.value->get()));
  return result;
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
      stable::detail::from(*result.value),
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
#ifndef STABLE_TORCH_ERROR_CODE_CHECK
#define STABLE_TORCH_ERROR_CODE_CHECK TORCH_ERROR_CODE_CHECK
#endif
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
