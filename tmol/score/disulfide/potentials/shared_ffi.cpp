#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <utility>

#include "xla/ffi/api/ffi.h"

#include <tmol/score/disulfide/potentials/shared_cpu.h>

namespace ffi = xla::ffi;

namespace {

using Buffer = ffi::AnyBuffer;
using Result = ffi::Result<Buffer>;

ffi::Error invalid(std::string message) {
  return ffi::Error::InvalidArgument(std::move(message));
}

bool shape_is(Buffer buffer, std::initializer_list<int64_t> dimensions) {
  auto actual = buffer.dimensions();
  return actual.size() == dimensions.size()
         && std::equal(actual.begin(), actual.end(), dimensions.begin());
}

ffi::Error check_real_dtype(
    Buffer buffer, ffi::DataType dtype, char const* name) {
  if (buffer.element_type() != dtype) {
    return invalid(std::string(name) + " must match the coordinate dtype");
  }
  return ffi::Error::Success();
}

ffi::Error check_inputs(Buffer coords, Buffer pairs, Buffer params) {
  auto coord_shape = coords.dimensions();
  auto pair_shape = pairs.dimensions();
  auto param_shape = params.dimensions();
  if (coord_shape.size() != 2 || coord_shape[1] != 3) {
    return invalid("coords must have shape [n_atoms, 3]");
  }
  if (pair_shape.size() != 2 || pair_shape[1] != 8) {
    return invalid("pairs must have shape [n_pairs, 8]");
  }
  if (param_shape.size() != 2 || param_shape[1] != 33) {
    return invalid("params must have shape [n_params, 33]");
  }
  if (coords.element_type() != ffi::DataType::F32
      && coords.element_type() != ffi::DataType::F64) {
    return invalid("coords must have dtype float32 or float64");
  }
  if (pairs.element_type() != ffi::DataType::S32) {
    return invalid("pairs must have dtype int32");
  }
  return check_real_dtype(params, coords.element_type(), "params");
}

ffi::Error check_forward_results(
    Buffer coords,
    Buffer pairs,
    Result energies,
    Result* derivatives = nullptr) {
  int64_t n_pairs = pairs.dimensions()[0];
  if (!shape_is(*energies, {n_pairs})) {
    return invalid("energies must have shape [n_pairs]");
  }
  if (auto error =
          check_real_dtype(*energies, coords.element_type(), "energies");
      error.failure()) {
    return error;
  }
  if (derivatives != nullptr) {
    if (!shape_is(**derivatives, {n_pairs, 6, 3})) {
      return invalid("derivatives must have shape [n_pairs, 6, 3]");
    }
    return check_real_dtype(
        **derivatives, coords.element_type(), "derivatives");
  }
  return ffi::Error::Success();
}

ffi::Error check_vjp_buffers(
    Buffer pairs, Buffer derivatives, Buffer cotangents, Result gradient) {
  auto pair_shape = pairs.dimensions();
  auto gradient_shape = gradient->dimensions();
  if (pair_shape.size() != 2 || pair_shape[1] != 8) {
    return invalid("pairs must have shape [n_pairs, 8]");
  }
  if (pairs.element_type() != ffi::DataType::S32) {
    return invalid("pairs must have dtype int32");
  }
  int64_t n_pairs = pair_shape[0];
  if (!shape_is(derivatives, {n_pairs, 6, 3})) {
    return invalid("derivatives must have shape [n_pairs, 6, 3]");
  }
  if (!shape_is(cotangents, {n_pairs})) {
    return invalid("cotangents must have shape [n_pairs]");
  }
  if (gradient_shape.size() != 2 || gradient_shape[1] != 3) {
    return invalid("gradient must have shape [n_atoms, 3]");
  }
  auto dtype = derivatives.element_type();
  if (dtype != ffi::DataType::F32 && dtype != ffi::DataType::F64) {
    return invalid("derivatives must have dtype float32 or float64");
  }
  if (auto error = check_real_dtype(cotangents, dtype, "cotangents");
      error.failure()) {
    return error;
  }
  return check_real_dtype(*gradient, dtype, "gradient");
}

ffi::Error check_recompute_buffers(
    Buffer coords,
    Buffer pairs,
    Buffer params,
    Buffer cotangents,
    Result gradient,
    Result workspace) {
  if (auto error = check_inputs(coords, pairs, params); error.failure()) {
    return error;
  }
  if (!shape_is(cotangents, {pairs.dimensions()[0]})) {
    return invalid("cotangents must have shape [n_pairs]");
  }
  if (!shape_is(*gradient, {coords.dimensions()[0], 3})) {
    return invalid("gradient must have shape [n_atoms, 3]");
  }
  if (!shape_is(*workspace, {pairs.dimensions()[0], 6, 3})) {
    return invalid("workspace must have shape [n_pairs, 6, 3]");
  }
  if (auto error =
          check_real_dtype(cotangents, coords.element_type(), "cotangents");
      error.failure()) {
    return error;
  }
  if (auto error =
          check_real_dtype(*gradient, coords.element_type(), "gradient");
      error.failure()) {
    return error;
  }
  return check_real_dtype(*workspace, coords.element_type(), "workspace");
}

ffi::Error common_status(int32_t status) {
  if (status == TMOL_DISULFIDE_CPU_SUCCESS) return ffi::Error::Success();
  std::string message = "shared disulfide CPU call failed: ";
  message += tmol_disulfide_cpu_status_message(status);
  switch (status) {
    case TMOL_DISULFIDE_CPU_INVALID_MODE:
    case TMOL_DISULFIDE_CPU_INVALID_SHAPE:
    case TMOL_DISULFIDE_CPU_INVALID_INDEX:
    case TMOL_DISULFIDE_CPU_MISSING_DERIVATIVE_OUTPUT:
    case TMOL_DISULFIDE_CPU_INSUFFICIENT_WORKSPACE:
      return ffi::Error::InvalidArgument(message);
    default:
      return ffi::Error::Internal(message);
  }
}

ffi::Error check_abi() {
  uint32_t actual = tmol_disulfide_cpu_abi_version();
  if (actual == TMOL_DISULFIDE_CPU_ABI_VERSION) {
    return ffi::Error::Success();
  }
  return ffi::Error::Internal(
      "shared disulfide CPU ABI mismatch; reinstall matching native "
      "components (expected "
      + std::to_string(TMOL_DISULFIDE_CPU_ABI_VERSION) + ", found "
      + std::to_string(actual) + ")");
}

template <typename Real>
struct Core;

#define TMOL_DEFINE_CORE(Real, Suffix)                                       \
  template <>                                                                \
  struct Core<Real> {                                                        \
    using Inputs = tmol_disulfide_cpu_##Suffix##_inputs;                     \
    using Outputs = tmol_disulfide_cpu_##Suffix##_forward_outputs;           \
    static Inputs inputs(Buffer coords, Buffer pairs, Buffer params) {       \
      return {                                                               \
          static_cast<Real const*>(coords.untyped_data()),                   \
          coords.dimensions()[0],                                            \
          1,                                                                 \
          static_cast<int32_t const*>(pairs.untyped_data()),                 \
          pairs.dimensions()[0],                                             \
          1,                                                                 \
          static_cast<Real const*>(params.untyped_data()),                   \
          params.dimensions()[0],                                            \
          1};                                                                \
    }                                                                        \
    static int32_t forward(                                                  \
        Buffer coords,                                                       \
        Buffer pairs,                                                        \
        Buffer params,                                                       \
        int32_t mode,                                                        \
        Result energies,                                                     \
        Result* derivatives) {                                               \
      Outputs outputs{                                                       \
          static_cast<Real*>(energies->untyped_data()),                      \
          1,                                                                 \
          derivatives == nullptr                                             \
              ? nullptr                                                      \
              : static_cast<Real*>((*derivatives)->untyped_data()),          \
          derivatives == nullptr ? 0 : 6,                                    \
          derivatives == nullptr ? 0 : 1};                                   \
      return tmol_disulfide_cpu_##Suffix##_forward(                          \
          tmol_disulfide_cpu_serial_executor(),                              \
          inputs(coords, pairs, params),                                     \
          mode,                                                              \
          outputs);                                                          \
    }                                                                        \
    static int32_t vjp(                                                      \
        Buffer pairs,                                                        \
        Buffer derivatives,                                                  \
        Buffer cotangents,                                                   \
        Result gradient) {                                                   \
      return tmol_disulfide_cpu_##Suffix##_vjp(                              \
          tmol_disulfide_cpu_serial_executor(),                              \
          TMOL_DISULFIDE_CPU_MAY_OVERLAP,                                    \
          static_cast<int32_t const*>(pairs.untyped_data()),                 \
          pairs.dimensions()[0],                                             \
          1,                                                                 \
          static_cast<Real const*>(derivatives.untyped_data()),              \
          6,                                                                 \
          1,                                                                 \
          static_cast<Real const*>(cotangents.untyped_data()),               \
          1,                                                                 \
          static_cast<Real*>(gradient->untyped_data()),                      \
          gradient->dimensions()[0],                                         \
          1);                                                                \
    }                                                                        \
    static tmol_disulfide_cpu_workspace workspace(int64_t n_pairs) {         \
      return tmol_disulfide_cpu_##Suffix##_recompute_vjp_workspace(n_pairs); \
    }                                                                        \
    static int32_t recompute_vjp(                                            \
        Buffer coords,                                                       \
        Buffer pairs,                                                        \
        Buffer params,                                                       \
        Buffer cotangents,                                                   \
        Result gradient,                                                     \
        void* workspace,                                                     \
        size_t workspace_bytes) {                                            \
      return tmol_disulfide_cpu_##Suffix##_recompute_vjp(                    \
          tmol_disulfide_cpu_serial_executor(),                              \
          TMOL_DISULFIDE_CPU_MAY_OVERLAP,                                    \
          inputs(coords, pairs, params),                                     \
          static_cast<Real const*>(cotangents.untyped_data()),               \
          1,                                                                 \
          static_cast<Real*>(gradient->untyped_data()),                      \
          1,                                                                 \
          workspace,                                                         \
          workspace_bytes);                                                  \
    }                                                                        \
  }

TMOL_DEFINE_CORE(float, f32);
TMOL_DEFINE_CORE(double, f64);
#undef TMOL_DEFINE_CORE

ffi::Error forward(
    Buffer coords, Buffer pairs, Buffer params, Result energies) {
  if (auto error = check_abi(); error.failure()) return error;
  if (auto error = check_inputs(coords, pairs, params); error.failure()) {
    return error;
  }
  if (auto error = check_forward_results(coords, pairs, energies);
      error.failure()) {
    return error;
  }
  if (coords.element_type() == ffi::DataType::F32) {
    return common_status(
        Core<float>::forward(
            coords,
            pairs,
            params,
            TMOL_DISULFIDE_CPU_INFERENCE,
            energies,
            nullptr));
  }
  return common_status(
      Core<double>::forward(
          coords,
          pairs,
          params,
          TMOL_DISULFIDE_CPU_INFERENCE,
          energies,
          nullptr));
}

ffi::Error forward_tracked(
    Buffer coords,
    Buffer pairs,
    Buffer params,
    Result energies,
    Result derivatives) {
  if (auto error = check_abi(); error.failure()) return error;
  if (auto error = check_inputs(coords, pairs, params); error.failure()) {
    return error;
  }
  if (auto error = check_forward_results(coords, pairs, energies, &derivatives);
      error.failure()) {
    return error;
  }
  if (coords.element_type() == ffi::DataType::F32) {
    return common_status(
        Core<float>::forward(
            coords,
            pairs,
            params,
            TMOL_DISULFIDE_CPU_TRACKED,
            energies,
            &derivatives));
  }
  return common_status(
      Core<double>::forward(
          coords,
          pairs,
          params,
          TMOL_DISULFIDE_CPU_TRACKED,
          energies,
          &derivatives));
}

ffi::Error vjp(
    Buffer pairs, Buffer derivatives, Buffer cotangents, Result gradient) {
  if (auto error = check_abi(); error.failure()) return error;
  if (auto error = check_vjp_buffers(pairs, derivatives, cotangents, gradient);
      error.failure()) {
    return error;
  }
  return derivatives.element_type() == ffi::DataType::F32
             ? common_status(
                   Core<float>::vjp(pairs, derivatives, cotangents, gradient))
             : common_status(
                   Core<double>::vjp(pairs, derivatives, cotangents, gradient));
}

ffi::Error recompute_vjp(
    Buffer coords,
    Buffer pairs,
    Buffer params,
    Buffer cotangents,
    Result gradient,
    Result workspace) {
  if (auto error = check_abi(); error.failure()) return error;
  if (auto error = check_recompute_buffers(
          coords, pairs, params, cotangents, gradient, workspace);
      error.failure()) {
    return error;
  }
  auto invoke = [&](auto real_tag) -> ffi::Error {
    using Real = decltype(real_tag);
    auto required = Core<Real>::workspace(pairs.dimensions()[0]);
    return common_status(
        Core<Real>::recompute_vjp(
            coords,
            pairs,
            params,
            cotangents,
            gradient,
            required.bytes == 0 ? nullptr : workspace->untyped_data(),
            required.bytes));
  };
  return coords.element_type() == ffi::DataType::F32 ? invoke(float{})
                                                     : invoke(double{});
}

}  // namespace

#if defined(__GNUC__)
#define TMOL_DISULFIDE_FFI_EXPORT __attribute__((visibility("default")))
#else
#define TMOL_DISULFIDE_FFI_EXPORT
#endif

extern "C" TMOL_DISULFIDE_FFI_EXPORT XLA_FFI_Error*
tmol_disulfide_cpu_ffi_forward(XLA_FFI_CallFrame*);
extern "C" TMOL_DISULFIDE_FFI_EXPORT XLA_FFI_Error*
tmol_disulfide_cpu_ffi_forward_tracked(XLA_FFI_CallFrame*);
extern "C" TMOL_DISULFIDE_FFI_EXPORT XLA_FFI_Error* tmol_disulfide_cpu_ffi_vjp(
    XLA_FFI_CallFrame*);
extern "C" TMOL_DISULFIDE_FFI_EXPORT XLA_FFI_Error*
tmol_disulfide_cpu_ffi_recompute_vjp(XLA_FFI_CallFrame*);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_disulfide_cpu_ffi_forward,
    forward,
    ffi::Ffi::Bind().Arg<Buffer>().Arg<Buffer>().Arg<Buffer>().Ret<Buffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_disulfide_cpu_ffi_forward_tracked,
    forward_tracked,
    ffi::Ffi::Bind()
        .Arg<Buffer>()
        .Arg<Buffer>()
        .Arg<Buffer>()
        .Ret<Buffer>()
        .Ret<Buffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_disulfide_cpu_ffi_vjp,
    vjp,
    ffi::Ffi::Bind().Arg<Buffer>().Arg<Buffer>().Arg<Buffer>().Ret<Buffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_disulfide_cpu_ffi_recompute_vjp,
    recompute_vjp,
    ffi::Ffi::Bind()
        .Arg<Buffer>()
        .Arg<Buffer>()
        .Arg<Buffer>()
        .Arg<Buffer>()
        .Ret<Buffer>()
        .Ret<Buffer>());

#undef TMOL_DISULFIDE_FFI_EXPORT
