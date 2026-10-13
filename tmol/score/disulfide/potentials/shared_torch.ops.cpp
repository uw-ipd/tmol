#include <tmol/utility/tensor/native_library.hh>

#include <tmol/score/disulfide/potentials/shared_cpu.h>

namespace tmol::score::disulfide::potentials {
namespace {

using tmol::native::Tensor;
using tmol::native::autograd::AutogradContext;
using tmol::native::autograd::Function;
using tmol::native::autograd::tensor_list;

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
      coords.device().is_cpu() && pairs.device().is_cpu()
          && params.device().is_cpu(),
      "shared prepared disulfide adapter is CPU-only");
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

#define TMOL_DEFINE_FORWARD_CALL(Real, Suffix)                          \
  template <>                                                           \
  int32_t forward_call<Real>(                                           \
      Tensor const& coords,                                             \
      Tensor const& pairs,                                              \
      Tensor const& params,                                             \
      Tensor const& energies,                                           \
      Tensor const& derivatives,                                        \
      int32_t mode) {                                                   \
    auto outputs = tmol_disulfide_cpu_##Suffix##_forward_outputs{       \
        energies.data_ptr<Real>(),                                      \
        energies.stride(0),                                             \
        derivatives.defined() ? derivatives.data_ptr<Real>() : nullptr, \
        derivatives.defined() ? derivatives.stride(0) / 3 : 0,          \
        derivatives.defined() ? derivatives.stride(1) / 3 : 0};         \
    return tmol_disulfide_cpu_##Suffix##_forward(                       \
        torch_executor(),                                               \
        make_inputs<Real>(coords, pairs, params),                       \
        mode,                                                           \
        outputs);                                                       \
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

void check_status(int32_t status) {
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
    check_status(status);
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
    check_status(status);
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

TMOL_CPU_AUTOGRAD_LIBRARY(tmol_disulfide_shared, m) {
  m.def("prepared", TMOL_FN(&prepared_disulfide));
}

}  // namespace tmol::score::disulfide::potentials

#ifdef TMOL_SHARED_DISULFIDE_TEST_MODULE
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { tmol::native::bind_libraries(m); }
#endif
