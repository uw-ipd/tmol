#include <dlfcn.h>

#include <mutex>
#include <string>

#include <tmol/utility/tensor/native_library.hh>

#include <tmol/score/disulfide/potentials/shared_cpu.h>
#include <tmol/score/disulfide/potentials/shared_cuda.h>

namespace tmol::score::disulfide::potentials {
namespace {

using tmol::native::Tensor;
using tmol::native::autograd::AutogradContext;
using tmol::native::autograd::Function;
using tmol::native::autograd::tensor_list;

struct CudaAPI {
  void* library = nullptr;
  decltype(&tmol_disulfide_cuda_abi_version) abi_version = nullptr;
  decltype(&tmol_disulfide_cuda_status_message) status_message = nullptr;
  decltype(&tmol_disulfide_cuda_f32_recompute_vjp_workspace)
      f32_recompute_vjp_workspace = nullptr;
  decltype(&tmol_disulfide_cuda_f64_recompute_vjp_workspace)
      f64_recompute_vjp_workspace = nullptr;
  decltype(&tmol_disulfide_cuda_f32_forward) f32_forward = nullptr;
  decltype(&tmol_disulfide_cuda_f64_forward) f64_forward = nullptr;
  decltype(&tmol_disulfide_cuda_f32_vjp) f32_vjp = nullptr;
  decltype(&tmol_disulfide_cuda_f64_vjp) f64_vjp = nullptr;
  decltype(&tmol_disulfide_cuda_f32_recompute_vjp) f32_recompute_vjp = nullptr;
  decltype(&tmol_disulfide_cuda_f64_recompute_vjp) f64_recompute_vjp = nullptr;
};

CudaAPI* active_cuda_api = nullptr;
std::mutex cuda_registration_mutex;
std::string cuda_library_path;

CudaAPI& cuda_api() {
  STD_TORCH_CHECK(
      active_cuda_api, "shared disulfide CUDA adapter is not registered");
  return *active_cuda_api;
}

template <typename Symbol>
Symbol load_cuda_symbol(void* library, char const* name) {
  dlerror();
  void* address = dlsym(library, name);
  char const* error = dlerror();
  STD_TORCH_CHECK(
      !error && address,
      "shared disulfide CUDA library is missing ",
      name,
      error ? std::string(": ") + error : std::string());
  return reinterpret_cast<Symbol>(address);
}

uintptr_t current_cuda_stream(Tensor const& tensor) {
  using DirectGetStream = AOTITorchError (*)(int32_t, void**);
  static auto direct = reinterpret_cast<DirectGetStream>(
      dlsym(RTLD_DEFAULT, "aoti_torch_get_current_cuda_stream"));
  void* native_stream = nullptr;
  if (direct) {
    STABLE_TORCH_ERROR_CODE_CHECK(
        direct(tensor.device().index(), &native_stream));
    return reinterpret_cast<uintptr_t>(native_stream);
  }

  using NativeHandle = AOTITorchError (*)(StreamHandle, void**);
  static auto native_handle = reinterpret_cast<NativeHandle>(
      dlsym(RTLD_DEFAULT, "torch_stream_native_handle"));
  STD_TORCH_CHECK(
      native_handle,
      "Torch does not expose a compatible current-CUDA-stream stable ABI");
  StreamHandle stream = nullptr;
  auto get_status =
      aoti_torch_get_current_stream(tensor.device().index(), &stream);
  if (get_status != TORCH_SUCCESS) {
    STABLE_TORCH_ERROR_CODE_CHECK(get_status);
  }
  auto native_status = native_handle(stream, &native_stream);
  auto delete_status = aoti_torch_delete_stream(stream);
  STABLE_TORCH_ERROR_CODE_CHECK(native_status);
  STABLE_TORCH_ERROR_CODE_CHECK(delete_status);
  return reinterpret_cast<uintptr_t>(native_stream);
}

void torch_parallel_for(
    void*, int64_t count, tmol_disulfide_cpu_chunk chunk, void* task_context) {
  tmol::native::parallel_for(0, count, 1, [=](int64_t begin, int64_t end) {
    chunk(task_context, begin, end);
  });
}
tmol_disulfide_cpu_executor torch_executor() {
  return {nullptr, torch_parallel_for};
}

uint32_t common_abi_version() {
  static uint32_t const version = tmol_disulfide_cpu_abi_version();
  return version;
}

void check_common_abi() {
  uint32_t const actual = common_abi_version();
  STD_TORCH_CHECK(
      actual == TMOL_DISULFIDE_CPU_ABI_VERSION,
      "shared disulfide CPU ABI mismatch: expected ",
      TMOL_DISULFIDE_CPU_ABI_VERSION,
      ", found ",
      actual,
      "; reinstall matching tmol native components");
}

void check_inputs(
    Tensor const& coords, Tensor const& pairs, Tensor const& params) {
  STD_TORCH_CHECK(
      coords.device() == pairs.device() && coords.device() == params.device(),
      "coords, pairs, and params must be on the same device");
  STD_TORCH_CHECK(
      coords.device().is_cpu() || coords.device().is_cuda(),
      "shared prepared disulfide adapter supports CPU and CUDA only");
  STD_TORCH_CHECK(
      coords.scalar_type() == tmol::native::kFloat32
          || coords.scalar_type() == tmol::native::kFloat64,
      "coords must have dtype float32 or float64");
  STD_TORCH_CHECK(
      params.scalar_type() == coords.scalar_type(),
      "params must have the same dtype as coords");
  STD_TORCH_CHECK(
      pairs.scalar_type() == tmol::native::kInt32,
      "pairs must have dtype int32");
  STD_TORCH_CHECK(
      coords.dim() == 2 && coords.size(1) == 3,
      "coords must have shape [n_atoms, 3]");
  STD_TORCH_CHECK(
      pairs.dim() == 2 && pairs.size(1) == 8,
      "pairs must have shape [n_pairs, 8]");
  STD_TORCH_CHECK(
      params.dim() == 2 && params.size(1) == 33,
      "params must have shape [n_params, 33]");
  STD_TORCH_CHECK(
      coords.stride(0) > 0 && coords.stride(1) == 1
          && coords.stride(0) % 3 == 0,
      "coords must contain tightly packed xyz records");
  STD_TORCH_CHECK(
      pairs.stride(0) > 0 && pairs.stride(1) == 1 && pairs.stride(0) % 8 == 0,
      "pairs must contain tightly packed eight-int records");
  STD_TORCH_CHECK(
      params.stride(0) > 0 && params.stride(1) == 1
          && params.stride(0) % 33 == 0,
      "params must contain tightly packed 33-scalar records");
  STD_TORCH_CHECK(
      !params.requires_grad(), "parameter gradients are not supported");
}

template <typename Real>
struct CPUInputs;
template <>
struct CPUInputs<float> {
  using type = tmol_disulfide_cpu_f32_inputs;
};
template <>
struct CPUInputs<double> {
  using type = tmol_disulfide_cpu_f64_inputs;
};

template <typename Real>
typename CPUInputs<Real>::type make_inputs(
    Tensor const&, Tensor const&, Tensor const&);

template <>
tmol_disulfide_cpu_f32_inputs make_inputs<float>(
    Tensor const& coords, Tensor const& pairs, Tensor const& params) {
  return {
      coords.const_data_ptr<float>(),
      coords.size(0),
      coords.stride(0) / 3,
      pairs.const_data_ptr<int32_t>(),
      pairs.size(0),
      pairs.stride(0) / 8,
      params.const_data_ptr<float>(),
      params.size(0),
      params.stride(0) / 33};
}

template <>
tmol_disulfide_cpu_f64_inputs make_inputs<double>(
    Tensor const& coords, Tensor const& pairs, Tensor const& params) {
  return {
      coords.const_data_ptr<double>(),
      coords.size(0),
      coords.stride(0) / 3,
      pairs.const_data_ptr<int32_t>(),
      pairs.size(0),
      pairs.stride(0) / 8,
      params.const_data_ptr<double>(),
      params.size(0),
      params.stride(0) / 33};
}

template <typename Real>
int32_t forward_call(
    Tensor const& coords,
    Tensor const& pairs,
    Tensor const& params,
    Tensor const& energies,
    Tensor const& derivatives,
    int32_t mode);

#define TMOL_DEFINE_FORWARD_CALL(Real, Suffix)                            \
  template <>                                                             \
  int32_t forward_call<Real>(                                             \
      Tensor const& coords,                                               \
      Tensor const& pairs,                                                \
      Tensor const& params,                                               \
      Tensor const& energies,                                             \
      Tensor const& derivatives,                                          \
      int32_t mode) {                                                     \
    if (coords.is_cuda()) {                                               \
      auto inputs = tmol_disulfide_cuda_##Suffix##_inputs{                \
          coords.const_data_ptr<Real>(),                                  \
          coords.size(0),                                                 \
          coords.stride(0) / 3,                                           \
          pairs.const_data_ptr<int32_t>(),                                \
          pairs.size(0),                                                  \
          pairs.stride(0) / 8,                                            \
          params.const_data_ptr<Real>(),                                  \
          params.size(0),                                                 \
          params.stride(0) / 33};                                         \
      auto outputs = tmol_disulfide_cuda_##Suffix##_forward_outputs{      \
          energies.data_ptr<Real>(),                                      \
          energies.stride(0),                                             \
          derivatives.defined() ? derivatives.data_ptr<Real>() : nullptr, \
          derivatives.defined() ? derivatives.stride(0) / 3 : 0,          \
          derivatives.defined() ? derivatives.stride(1) / 3 : 0};         \
      return cuda_api().Suffix##_forward(                                 \
          current_cuda_stream(coords), inputs, mode, outputs, nullptr);   \
    }                                                                     \
    auto outputs = tmol_disulfide_cpu_##Suffix##_forward_outputs{         \
        energies.data_ptr<Real>(),                                        \
        energies.stride(0),                                               \
        derivatives.defined() ? derivatives.data_ptr<Real>() : nullptr,   \
        derivatives.defined() ? derivatives.stride(0) / 3 : 0,            \
        derivatives.defined() ? derivatives.stride(1) / 3 : 0};           \
    return tmol_disulfide_cpu_##Suffix##_forward(                         \
        torch_executor(),                                                 \
        make_inputs<Real>(coords, pairs, params),                         \
        mode,                                                             \
        outputs);                                                         \
  }

TMOL_DEFINE_FORWARD_CALL(float, f32)
TMOL_DEFINE_FORWARD_CALL(double, f64)
#undef TMOL_DEFINE_FORWARD_CALL

template <typename Real>
int32_t retained_vjp_call(
    Tensor const& pairs,
    Tensor const& derivatives,
    Tensor const& cotangents,
    Tensor const& gradient);

#define TMOL_DEFINE_RETAINED_VJP_CALL(Real, Suffix) \
  template <>                                       \
  int32_t retained_vjp_call<Real>(                  \
      Tensor const& pairs,                          \
      Tensor const& derivatives,                    \
      Tensor const& cotangents,                     \
      Tensor const& gradient) {                     \
    if (pairs.is_cuda()) {                          \
      return cuda_api().Suffix##_vjp(               \
          current_cuda_stream(pairs),               \
          TMOL_DISULFIDE_CUDA_MAY_OVERLAP,          \
          pairs.const_data_ptr<int32_t>(),          \
          pairs.size(0),                            \
          pairs.stride(0) / 8,                      \
          derivatives.const_data_ptr<Real>(),       \
          derivatives.stride(0) / 3,                \
          derivatives.stride(1) / 3,                \
          cotangents.const_data_ptr<Real>(),        \
          cotangents.stride(0),                     \
          gradient.data_ptr<Real>(),                \
          gradient.size(0),                         \
          gradient.stride(0) / 3,                   \
          nullptr);                                 \
    }                                               \
    return tmol_disulfide_cpu_##Suffix##_vjp(       \
        torch_executor(),                           \
        TMOL_DISULFIDE_CPU_MAY_OVERLAP,             \
        pairs.const_data_ptr<int32_t>(),            \
        pairs.size(0),                              \
        pairs.stride(0) / 8,                        \
        derivatives.const_data_ptr<Real>(),         \
        derivatives.stride(0) / 3,                  \
        derivatives.stride(1) / 3,                  \
        cotangents.const_data_ptr<Real>(),          \
        cotangents.stride(0),                       \
        gradient.data_ptr<Real>(),                  \
        gradient.size(0),                           \
        gradient.stride(0) / 3);                    \
  }

TMOL_DEFINE_RETAINED_VJP_CALL(float, f32)
TMOL_DEFINE_RETAINED_VJP_CALL(double, f64)
#undef TMOL_DEFINE_RETAINED_VJP_CALL

template <typename Real>
int32_t recompute_vjp_call(
    Tensor const& coords,
    Tensor const& pairs,
    Tensor const& params,
    Tensor const& cotangents,
    Tensor const& gradient,
    Tensor const& workspace);

#define TMOL_DEFINE_RECOMPUTE_VJP_CALL(Real, Suffix)                          \
  template <>                                                                 \
  int32_t recompute_vjp_call<Real>(                                           \
      Tensor const& coords,                                                   \
      Tensor const& pairs,                                                    \
      Tensor const& params,                                                   \
      Tensor const& cotangents,                                               \
      Tensor const& gradient,                                                 \
      Tensor const& workspace) {                                              \
    if (coords.is_cuda()) {                                                   \
      auto inputs = tmol_disulfide_cuda_##Suffix##_inputs{                    \
          coords.const_data_ptr<Real>(),                                      \
          coords.size(0),                                                     \
          coords.stride(0) / 3,                                               \
          pairs.const_data_ptr<int32_t>(),                                    \
          pairs.size(0),                                                      \
          pairs.stride(0) / 8,                                                \
          params.const_data_ptr<Real>(),                                      \
          params.size(0),                                                     \
          params.stride(0) / 33};                                             \
      auto required =                                                         \
          cuda_api().Suffix##_recompute_vjp_workspace(pairs.size(0));         \
      STD_TORCH_CHECK(                                                        \
          static_cast<size_t>(workspace.numel() * sizeof(Real))               \
              >= required.bytes,                                              \
          "internal disulfide CUDA workspace is too small");                  \
      return cuda_api().Suffix##_recompute_vjp(                               \
          current_cuda_stream(coords),                                        \
          TMOL_DISULFIDE_CUDA_MAY_OVERLAP,                                    \
          inputs,                                                             \
          cotangents.const_data_ptr<Real>(),                                  \
          cotangents.stride(0),                                               \
          gradient.data_ptr<Real>(),                                          \
          gradient.stride(0) / 3,                                             \
          workspace.data_ptr<Real>(),                                         \
          workspace.numel() * sizeof(Real),                                   \
          nullptr);                                                           \
    }                                                                         \
    auto required =                                                           \
        tmol_disulfide_cpu_##Suffix##_recompute_vjp_workspace(pairs.size(0)); \
    STD_TORCH_CHECK(                                                          \
        static_cast<size_t>(workspace.numel() * sizeof(Real))                 \
            >= required.bytes,                                                \
        "internal disulfide workspace is too small");                         \
    return tmol_disulfide_cpu_##Suffix##_recompute_vjp(                       \
        torch_executor(),                                                     \
        TMOL_DISULFIDE_CPU_MAY_OVERLAP,                                       \
        make_inputs<Real>(coords, pairs, params),                             \
        cotangents.const_data_ptr<Real>(),                                    \
        cotangents.stride(0),                                                 \
        gradient.data_ptr<Real>(),                                            \
        gradient.stride(0) / 3,                                               \
        workspace.data_ptr<Real>(),                                           \
        workspace.numel() * sizeof(Real));                                    \
  }

TMOL_DEFINE_RECOMPUTE_VJP_CALL(float, f32)
TMOL_DEFINE_RECOMPUTE_VJP_CALL(double, f64)
#undef TMOL_DEFINE_RECOMPUTE_VJP_CALL

void check_status(int32_t status, bool cuda) {
  if (cuda) {
    STD_TORCH_CHECK(
        status == TMOL_DISULFIDE_CUDA_SUCCESS,
        "shared disulfide CUDA call failed: ",
        cuda_api().status_message(status));
    return;
  }
  STD_TORCH_CHECK(
      status == TMOL_DISULFIDE_CPU_SUCCESS,
      "shared disulfide CPU call failed: ",
      tmol_disulfide_cpu_status_message(status));
}

class PreparedDisulfideOp : public Function<PreparedDisulfideOp> {
 public:
  static Tensor forward(
      AutogradContext* ctx,
      Tensor coords,
      Tensor pairs,
      Tensor params,
      bool recompute_backward,
      bool needs_backward) {
    check_common_abi();
    check_inputs(coords, pairs, params);
    auto energies = tmol::native::empty({pairs.size(0)}, coords.options());
    bool tracked = needs_backward && !recompute_backward;
    Tensor derivatives;
    if (tracked) {
      derivatives =
          tmol::native::empty({pairs.size(0), 6, 3}, coords.options());
    }
    int32_t status = 0;
    if (coords.scalar_type() == tmol::native::kFloat32) {
      status = forward_call<float>(
          coords,
          pairs,
          params,
          energies,
          derivatives,
          tracked ? TMOL_DISULFIDE_CPU_TRACKED : TMOL_DISULFIDE_CPU_INFERENCE);
    } else {
      status = forward_call<double>(
          coords,
          pairs,
          params,
          energies,
          derivatives,
          tracked ? TMOL_DISULFIDE_CPU_TRACKED : TMOL_DISULFIDE_CPU_INFERENCE);
    }
    check_status(status, coords.is_cuda());
    if (needs_backward) {
      ctx->saved_data["recompute"] = recompute_backward;
      ctx->saved_data["n_atoms"] = coords.size(0);
      if (recompute_backward) {
        ctx->save_for_backward({coords, pairs, params});
      } else {
        ctx->save_for_backward({pairs, derivatives});
      }
    }
    return energies;
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    tensor_list result(5);
    if (!grad_outputs[0].defined()) return result;
    check_common_abi();
    auto saved = ctx->get_saved_variables();
    bool recompute = ctx->saved_data["recompute"].toBool();
    int64_t n_atoms = ctx->saved_data["n_atoms"].toInt();
    Tensor coords = recompute ? saved[0] : Tensor();
    Tensor pairs = recompute ? saved[1] : saved[0];
    Tensor params = recompute ? saved[2] : Tensor();
    Tensor derivatives = recompute ? Tensor() : saved[1];
    auto gradient =
        tmol::native::empty({n_atoms, 3}, grad_outputs[0].options());
    auto cotangents = grad_outputs[0].contiguous();
    int32_t status = 0;
    if (recompute) {
      auto workspace =
          tmol::native::empty({pairs.size(0), 6, 3}, coords.options());
      status =
          coords.scalar_type() == tmol::native::kFloat32
              ? recompute_vjp_call<float>(
                    coords, pairs, params, cotangents, gradient, workspace)
              : recompute_vjp_call<double>(
                    coords, pairs, params, cotangents, gradient, workspace);
    } else {
      status = derivatives.scalar_type() == tmol::native::kFloat32
                   ? retained_vjp_call<float>(
                         pairs, derivatives, cotangents, gradient)
                   : retained_vjp_call<double>(
                         pairs, derivatives, cotangents, gradient);
    }
    check_status(status, pairs.is_cuda());
    result[0] = gradient;
    return result;
  }
};

Tensor prepared_disulfide(
    Tensor coords, Tensor pairs, Tensor params, bool recompute_backward) {
  // PyTorch disables GradMode while invoking a custom Function forward.
  bool needs_backward =
      tmol::native::GradMode::is_enabled() && coords.requires_grad();
  return PreparedDisulfideOp::apply(
      coords, pairs, params, recompute_backward, needs_backward);
}

}  // namespace
void register_cuda_common(std::string const& path) {
  std::lock_guard<std::mutex> guard(cuda_registration_mutex);
  if (active_cuda_api) {
    STD_TORCH_CHECK(
        path == cuda_library_path,
        "shared disulfide CUDA adapter is already registered from ",
        cuda_library_path);
    return;
  }
  STD_TORCH_CHECK(!path.empty(), "shared disulfide CUDA library path is empty");

  void* library = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  STD_TORCH_CHECK(
      library,
      "failed to load shared disulfide CUDA library ",
      path,
      ": ",
      dlerror());
  try {
    CudaAPI loaded;
    loaded.library = library;
    loaded.abi_version = load_cuda_symbol<decltype(loaded.abi_version)>(
        library, "tmol_disulfide_cuda_abi_version");
    loaded.status_message = load_cuda_symbol<decltype(loaded.status_message)>(
        library, "tmol_disulfide_cuda_status_message");
    loaded.f32_recompute_vjp_workspace =
        load_cuda_symbol<decltype(loaded.f32_recompute_vjp_workspace)>(
            library, "tmol_disulfide_cuda_f32_recompute_vjp_workspace");
    loaded.f64_recompute_vjp_workspace =
        load_cuda_symbol<decltype(loaded.f64_recompute_vjp_workspace)>(
            library, "tmol_disulfide_cuda_f64_recompute_vjp_workspace");
    loaded.f32_forward = load_cuda_symbol<decltype(loaded.f32_forward)>(
        library, "tmol_disulfide_cuda_f32_forward");
    loaded.f64_forward = load_cuda_symbol<decltype(loaded.f64_forward)>(
        library, "tmol_disulfide_cuda_f64_forward");
    loaded.f32_vjp = load_cuda_symbol<decltype(loaded.f32_vjp)>(
        library, "tmol_disulfide_cuda_f32_vjp");
    loaded.f64_vjp = load_cuda_symbol<decltype(loaded.f64_vjp)>(
        library, "tmol_disulfide_cuda_f64_vjp");
    loaded.f32_recompute_vjp =
        load_cuda_symbol<decltype(loaded.f32_recompute_vjp)>(
            library, "tmol_disulfide_cuda_f32_recompute_vjp");
    loaded.f64_recompute_vjp =
        load_cuda_symbol<decltype(loaded.f64_recompute_vjp)>(
            library, "tmol_disulfide_cuda_f64_recompute_vjp");
    uint32_t const actual = loaded.abi_version();
    STD_TORCH_CHECK(
        actual == TMOL_DISULFIDE_CUDA_ABI_VERSION,
        "shared disulfide CUDA ABI mismatch: expected ",
        TMOL_DISULFIDE_CUDA_ABI_VERSION,
        ", found ",
        actual,
        "; reinstall matching tmol native components");

    using StableLibrary = tmol::native::stable::detail::StableLibrary;
    auto kernel = [](StableIValue* stack, uint64_t, uint64_t) {
      tmol::native::invoke_boxed<&prepared_disulfide>(
          stack,
          &prepared_disulfide,
          std::index_sequence_for<Tensor, Tensor, Tensor, bool>{});
    };
    auto cuda = std::make_unique<StableLibrary>(
        StableLibrary::Kind::IMPL,
        "tmol_disulfide_shared",
        "CUDA",
        __FILE__,
        __LINE__);
    auto autograd_cuda = std::make_unique<StableLibrary>(
        StableLibrary::Kind::IMPL,
        "tmol_disulfide_shared",
        "AutogradCUDA",
        __FILE__,
        __LINE__);
    cuda->impl("prepared", kernel);
    autograd_cuda->impl("prepared", kernel);

    static CudaAPI registered;
    static std::vector<std::unique_ptr<StableLibrary>> dispatch_libraries;
    registered = loaded;
    active_cuda_api = &registered;
    cuda_library_path = path;
    dispatch_libraries.push_back(std::move(cuda));
    dispatch_libraries.push_back(std::move(autograd_cuda));
  } catch (...) {
    dlclose(library);
    throw;
  }
}

TMOL_CPU_AUTOGRAD_LIBRARY(tmol_disulfide_shared, m) {
  m.def("prepared", TMOL_FN(&prepared_disulfide));
}

}  // namespace tmol::score::disulfide::potentials

#ifdef TMOL_SHARED_DISULFIDE_TEST_MODULE
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  tmol::native::bind_libraries(m);
  m.def(
      "_register_cuda_common",
      &tmol::score::disulfide::potentials::register_cuda_common);
}
#endif
