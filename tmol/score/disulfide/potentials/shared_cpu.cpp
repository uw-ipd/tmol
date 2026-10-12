#include "shared_cpu.hh"

#include <limits>

#include "potentials.hh"

namespace tmol {
namespace score {
namespace disulfide {
namespace potentials {
namespace {

void serial_parallel_for(
    void*, int64_t count, CPUChunk chunk, void* task_context) {
  chunk(task_context, 0, count);
}

template <typename Real>
bool valid_pair_indices(DisulfideCPUInputs<Real> const& inputs) {
  for (int64_t i = 0; i < inputs.pairs.size(0); ++i) {
    auto const pair = inputs.pairs[i];
    for (int j = 0; j < 6; ++j) {
      if (pair[j] < 0 || pair[j] >= inputs.coords.size(0)) return false;
    }
    if (pair[6] < 0 || pair[6] >= inputs.params.size(0) || pair[7] < 0
        || pair[7] >= inputs.params.size(0)) {
      return false;
    }
  }
  return true;
}

template <typename Real>
struct ForwardTask {
  DisulfideCPUInputs<Real> inputs;
  DisulfideForwardMode mode;
  DisulfideCPUForwardOutputs<Real> outputs;
  bool write_energies;
};

template <typename Real>
void forward_chunk(void* opaque, int64_t begin, int64_t end) {
  auto& task = *static_cast<ForwardTask<Real>*>(opaque);
  int64_t local_sizes[1] = {6};
  int64_t local_strides[1] = {1};
  int64_t deriv_sizes[2] = {1, 6};
  int64_t deriv_strides[2] = {6, 1};
  for (int64_t i = begin; i < end; ++i) {
    auto const pair = task.inputs.pairs[i];
    DisulfideVec3<Real> local_coords_storage[6];
    for (int j = 0; j < 6; ++j) {
      local_coords_storage[j] = task.inputs.coords[pair[j]];
    }
    TView<DisulfideVec3<Real>, 1, Device::CPU> local_coords(
        local_coords_storage, local_sizes, local_strides);
    if (task.mode == DisulfideForwardMode::inference) {
      task.outputs.energies[i] = disulfide_energy<Real, Device::CPU>(
          local_coords,
          0,
          1,
          2,
          3,
          4,
          5,
          task.inputs.params[pair[6]],
          task.inputs.params[pair[7]]);
      continue;
    }
    DisulfideVec3<Real> local_derivs_storage[6];
    for (auto& derivative : local_derivs_storage) derivative.setZero();
    TView<DisulfideVec3<Real>, 2, Device::CPU> local_derivs(
        local_derivs_storage, deriv_sizes, deriv_strides);
    Real energy = 0;
    accumulate_disulfide<Real, Device::CPU>(
        local_coords,
        0,
        1,
        2,
        3,
        4,
        5,
        task.inputs.params[pair[6]],
        task.inputs.params[pair[7]],
        local_derivs,
        Real(1),
        &energy);
    if (task.write_energies) task.outputs.energies[i] = energy;
    for (int j = 0; j < 6; ++j) {
      task.outputs.coordinate_derivatives[i][j] = local_derivs_storage[j];
    }
  }
}

template <typename Real>
struct VJPTask {
  TView<PreparedDisulfidePair, 1, Device::CPU> pairs;
  TView<DisulfideVec3<Real>, 2, Device::CPU> derivatives;
  TView<Real, 1, Device::CPU> cotangents;
  TView<DisulfideVec3<Real>, 1, Device::CPU> gradient;
};

template <typename Real>
void vjp_chunk(void* opaque, int64_t begin, int64_t end) {
  auto& task = *static_cast<VJPTask<Real>*>(opaque);
  for (int64_t i = begin; i < end; ++i) {
    auto const pair = task.pairs[i];
    for (int j = 0; j < 6; ++j) {
      task.gradient[pair[j]] += task.cotangents[i] * task.derivatives[i][j];
    }
  }
}

template <typename Real>
void vjp_task(void* opaque, int64_t begin, int64_t end) {
  auto& task = *static_cast<VJPTask<Real>*>(opaque);
  for (int64_t worker = begin; worker < end; ++worker) {
    if (worker == 0) {
      vjp_chunk<Real>(opaque, 0, task.pairs.size(0));
    }
  }
}

}  // namespace

CPUExecutor serial_cpu_executor() { return {nullptr, serial_parallel_for}; }

extern "C" TMOL_DISULFIDE_CPU_API uint32_t tmol_disulfide_cpu_abi_version() {
  return TMOL_DISULFIDE_CPU_ABI_VERSION;
}

extern "C" TMOL_DISULFIDE_CPU_API char const* tmol_disulfide_cpu_status_message(
    int32_t status) {
  switch (static_cast<DisulfideCPUStatus>(status)) {
    case DisulfideCPUStatus::success:
      return "success";
    case DisulfideCPUStatus::invalid_executor:
      return "invalid CPU executor";
    case DisulfideCPUStatus::invalid_mode:
      return "invalid execution mode";
    case DisulfideCPUStatus::invalid_shape:
      return "invalid borrowed-view shape or pointer";
    case DisulfideCPUStatus::invalid_index:
      return "prepared pair index out of bounds";
    case DisulfideCPUStatus::missing_derivative_output:
      return "tracked forward requires pair coordinate derivatives";
    case DisulfideCPUStatus::insufficient_workspace:
      return "insufficient or misaligned caller workspace";
  }
  return "unknown disulfide CPU status";
}

template <typename Real>
WorkspaceRequirements disulfide_cpu_forward_workspace(int64_t) {
  return {0, alignof(DisulfideVec3<Real>)};
}

template <typename Real>
WorkspaceRequirements disulfide_cpu_recompute_vjp_workspace(int64_t n_pairs) {
  if (n_pairs <= 0) return {0, alignof(DisulfideVec3<Real>)};
  constexpr size_t row_bytes = 6 * sizeof(DisulfideVec3<Real>);
  constexpr auto max = std::numeric_limits<size_t>::max();
  if (static_cast<uint64_t>(n_pairs) > max / row_bytes) {
    return {max, alignof(DisulfideVec3<Real>)};
  }
  return {
      static_cast<size_t>(n_pairs) * row_bytes, alignof(DisulfideVec3<Real>)};
}

template <typename Real>
DisulfideCPUStatus disulfide_cpu_forward_impl(
    CPUExecutor executor,
    DisulfideCPUInputs<Real> inputs,
    DisulfideForwardMode mode,
    DisulfideCPUForwardOutputs<Real> outputs,
    bool write_energies) {
  if (!executor.parallel_for) return DisulfideCPUStatus::invalid_executor;
  if (mode != DisulfideForwardMode::inference
      && mode != DisulfideForwardMode::tracked) {
    return DisulfideCPUStatus::invalid_mode;
  }
  if (inputs.coords.size(0) < 0 || inputs.pairs.size(0) < 0
      || inputs.params.size(0) < 0
      || (write_energies && outputs.energies.size(0) != inputs.pairs.size(0))
      || (inputs.pairs.size(0) != 0
          && (!inputs.coords.data() || !inputs.pairs.data()
              || !inputs.params.data()
              || (write_energies && !outputs.energies.data())))) {
    return DisulfideCPUStatus::invalid_shape;
  }
  if (mode == DisulfideForwardMode::tracked
      && (outputs.coordinate_derivatives.size(0) != inputs.pairs.size(0)
          || outputs.coordinate_derivatives.size(1) != 6
          || (inputs.pairs.size(0) != 0
              && !outputs.coordinate_derivatives.data()))) {
    return DisulfideCPUStatus::missing_derivative_output;
  }
  if (!valid_pair_indices(inputs)) return DisulfideCPUStatus::invalid_index;
  ForwardTask<Real> task{inputs, mode, outputs, write_energies};
  executor.parallel_for(
      executor.context, inputs.pairs.size(0), forward_chunk<Real>, &task);
  return DisulfideCPUStatus::success;
}

template <typename Real>
DisulfideCPUStatus disulfide_cpu_forward(
    CPUExecutor executor,
    DisulfideCPUInputs<Real> inputs,
    DisulfideForwardMode mode,
    DisulfideCPUForwardOutputs<Real> outputs) {
  return disulfide_cpu_forward_impl<Real>(
      executor, inputs, mode, outputs, true);
}

template <typename Real>
DisulfideCPUStatus disulfide_cpu_vjp(
    CPUExecutor executor,
    DisulfideVJPMode mode,
    TView<PreparedDisulfidePair, 1, Device::CPU> pairs,
    TView<DisulfideVec3<Real>, 2, Device::CPU> coordinate_derivatives,
    TView<Real, 1, Device::CPU> cotangents,
    TView<DisulfideVec3<Real>, 1, Device::CPU> coordinate_gradient) {
  if (!executor.parallel_for) return DisulfideCPUStatus::invalid_executor;
  if (mode != DisulfideVJPMode::may_overlap
      && mode != DisulfideVJPMode::disjoint_pairs) {
    return DisulfideCPUStatus::invalid_mode;
  }
  if (pairs.size(0) < 0 || coordinate_gradient.size(0) < 0
      || coordinate_derivatives.size(0) != pairs.size(0)
      || coordinate_derivatives.size(1) != 6
      || cotangents.size(0) != pairs.size(0)
      || (pairs.size(0) != 0
          && (!pairs.data() || !coordinate_derivatives.data()
              || !cotangents.data()))
      || (coordinate_gradient.size(0) != 0 && !coordinate_gradient.data())) {
    return DisulfideCPUStatus::invalid_shape;
  }
  for (int64_t i = 0; i < pairs.size(0); ++i) {
    for (int j = 0; j < 6; ++j) {
      if (pairs[i][j] < 0 || pairs[i][j] >= coordinate_gradient.size(0)) {
        return DisulfideCPUStatus::invalid_index;
      }
    }
  }
  for (int64_t i = 0; i < coordinate_gradient.size(0); ++i) {
    coordinate_gradient[i].setZero();
  }
  VJPTask<Real> task{
      pairs, coordinate_derivatives, cotangents, coordinate_gradient};
  if (mode == DisulfideVJPMode::disjoint_pairs) {
    executor.parallel_for(
        executor.context, pairs.size(0), vjp_chunk<Real>, &task);
  } else {
    // Keep overlapping scatter race-free without allocating an atom-sized
    // reduction buffer; the adapter still controls the one scheduled task.
    executor.parallel_for(executor.context, 1, vjp_task<Real>, &task);
  }
  return DisulfideCPUStatus::success;
}

template <typename Real>
DisulfideCPUStatus disulfide_cpu_recompute_vjp(
    CPUExecutor executor,
    DisulfideVJPMode mode,
    DisulfideCPUInputs<Real> inputs,
    TView<Real, 1, Device::CPU> cotangents,
    TView<DisulfideVec3<Real>, 1, Device::CPU> coordinate_gradient,
    void* workspace,
    size_t workspace_bytes) {
  auto required =
      disulfide_cpu_recompute_vjp_workspace<Real>(inputs.pairs.size(0));
  if (workspace_bytes < required.bytes || (required.bytes && !workspace)
      || (required.bytes
          && reinterpret_cast<uintptr_t>(workspace) % required.alignment
                 != 0)) {
    return DisulfideCPUStatus::insufficient_workspace;
  }
  int64_t deriv_sizes[2] = {inputs.pairs.size(0), 6};
  int64_t deriv_strides[2] = {6, 1};
  TView<DisulfideVec3<Real>, 2, Device::CPU> derivatives(
      static_cast<DisulfideVec3<Real>*>(workspace), deriv_sizes, deriv_strides);
  auto status = disulfide_cpu_forward_impl<Real>(
      executor,
      inputs,
      DisulfideForwardMode::tracked,
      {{}, derivatives},
      false);
  if (status != DisulfideCPUStatus::success) return status;
  return disulfide_cpu_vjp<Real>(
      executor,
      mode,
      inputs.pairs,
      derivatives,
      cotangents,
      coordinate_gradient);
}

#define TMOL_INSTANTIATE_DISULFIDE_CPU(Real)                                  \
  template WorkspaceRequirements disulfide_cpu_forward_workspace<Real>(       \
      int64_t);                                                               \
  template WorkspaceRequirements disulfide_cpu_recompute_vjp_workspace<Real>( \
      int64_t);                                                               \
  template DisulfideCPUStatus disulfide_cpu_forward<Real>(                    \
      CPUExecutor,                                                            \
      DisulfideCPUInputs<Real>,                                               \
      DisulfideForwardMode,                                                   \
      DisulfideCPUForwardOutputs<Real>);                                      \
  template DisulfideCPUStatus disulfide_cpu_vjp<Real>(                        \
      CPUExecutor,                                                            \
      DisulfideVJPMode,                                                       \
      TView<PreparedDisulfidePair, 1, Device::CPU>,                           \
      TView<DisulfideVec3<Real>, 2, Device::CPU>,                             \
      TView<Real, 1, Device::CPU>,                                            \
      TView<DisulfideVec3<Real>, 1, Device::CPU>);                            \
  template DisulfideCPUStatus disulfide_cpu_recompute_vjp<Real>(              \
      CPUExecutor,                                                            \
      DisulfideVJPMode,                                                       \
      DisulfideCPUInputs<Real>,                                               \
      TView<Real, 1, Device::CPU>,                                            \
      TView<DisulfideVec3<Real>, 1, Device::CPU>,                             \
      void*,                                                                  \
      size_t)

TMOL_INSTANTIATE_DISULFIDE_CPU(float);
TMOL_INSTANTIATE_DISULFIDE_CPU(double);
#undef TMOL_INSTANTIATE_DISULFIDE_CPU

namespace {

template <typename Real, typename CInputs>
DisulfideCPUInputs<Real> cpp_inputs(CInputs inputs) {
  static_assert(sizeof(DisulfideVec3<Real>) == 3 * sizeof(Real));
  static_assert(sizeof(PreparedDisulfidePair) == 8 * sizeof(int32_t));
  static_assert(sizeof(DisulfideGlobalParams<Real>) == 33 * sizeof(Real));
  int64_t coord_sizes[1] = {inputs.n_atoms};
  int64_t coord_strides[1] = {inputs.coord_stride};
  int64_t pair_sizes[1] = {inputs.n_pairs};
  int64_t pair_strides[1] = {inputs.pair_stride};
  int64_t param_sizes[1] = {inputs.n_params};
  int64_t param_strides[1] = {inputs.param_stride};
  return {
      {reinterpret_cast<DisulfideVec3<Real>*>(const_cast<Real*>(inputs.coords)),
       coord_sizes,
       coord_strides},
      {reinterpret_cast<PreparedDisulfidePair*>(
           const_cast<int32_t*>(inputs.pairs)),
       pair_sizes,
       pair_strides},
      {reinterpret_cast<DisulfideGlobalParams<Real>*>(
           const_cast<Real*>(inputs.params)),
       param_sizes,
       param_strides}};
}

template <typename Real, typename COutputs>
DisulfideCPUForwardOutputs<Real> cpp_outputs(
    int64_t n_pairs, COutputs outputs) {
  int64_t energy_sizes[1] = {n_pairs};
  int64_t energy_strides[1] = {outputs.energy_stride};
  int64_t derivative_sizes[2] = {
      outputs.coordinate_derivatives ? n_pairs : 0,
      outputs.coordinate_derivatives ? 6 : 0};
  int64_t derivative_strides[2] = {
      outputs.derivative_pair_stride, outputs.derivative_atom_stride};
  return {
      {outputs.energies, energy_sizes, energy_strides},
      {reinterpret_cast<DisulfideVec3<Real>*>(outputs.coordinate_derivatives),
       derivative_sizes,
       derivative_strides}};
}

template <typename Real>
DisulfideCPUStatus c_vjp(
    CPUExecutor executor,
    int32_t mode,
    int32_t const* pairs,
    int64_t n_pairs,
    int64_t pair_stride,
    Real const* derivatives,
    int64_t derivative_pair_stride,
    int64_t derivative_atom_stride,
    Real const* cotangents,
    int64_t cotangent_stride,
    Real* gradient,
    int64_t n_atoms,
    int64_t gradient_stride) {
  int64_t pair_sizes[1] = {n_pairs}, pair_strides[1] = {pair_stride};
  int64_t derivative_sizes[2] = {n_pairs, 6};
  int64_t derivative_strides[2] = {
      derivative_pair_stride, derivative_atom_stride};
  int64_t cotangent_sizes[1] = {n_pairs};
  int64_t cotangent_strides[1] = {cotangent_stride};
  int64_t gradient_sizes[1] = {n_atoms};
  int64_t gradient_strides[1] = {gradient_stride};
  return disulfide_cpu_vjp<Real>(
      executor,
      static_cast<DisulfideVJPMode>(mode),
      {reinterpret_cast<PreparedDisulfidePair*>(const_cast<int32_t*>(pairs)),
       pair_sizes,
       pair_strides},
      {reinterpret_cast<DisulfideVec3<Real>*>(const_cast<Real*>(derivatives)),
       derivative_sizes,
       derivative_strides},
      {const_cast<Real*>(cotangents), cotangent_sizes, cotangent_strides},
      {reinterpret_cast<DisulfideVec3<Real>*>(gradient),
       gradient_sizes,
       gradient_strides});
}

template <typename Real, typename CInputs>
DisulfideCPUStatus c_recompute_vjp(
    CPUExecutor executor,
    int32_t mode,
    CInputs inputs,
    Real const* cotangents,
    int64_t cotangent_stride,
    Real* gradient,
    int64_t gradient_stride,
    void* workspace,
    size_t workspace_bytes) {
  int64_t cotangent_sizes[1] = {inputs.n_pairs};
  int64_t cotangent_strides[1] = {cotangent_stride};
  int64_t gradient_sizes[1] = {inputs.n_atoms};
  int64_t gradient_strides[1] = {gradient_stride};
  return disulfide_cpu_recompute_vjp<Real>(
      executor,
      static_cast<DisulfideVJPMode>(mode),
      cpp_inputs<Real>(inputs),
      {const_cast<Real*>(cotangents), cotangent_sizes, cotangent_strides},
      {reinterpret_cast<DisulfideVec3<Real>*>(gradient),
       gradient_sizes,
       gradient_strides},
      workspace,
      workspace_bytes);
}

}  // namespace

extern "C" TMOL_DISULFIDE_CPU_API tmol_disulfide_cpu_executor
tmol_disulfide_cpu_serial_executor() {
  return serial_cpu_executor();
}

#define TMOL_DEFINE_DISULFIDE_C_ABI(Suffix, Real)                              \
  extern "C" TMOL_DISULFIDE_CPU_API tmol_disulfide_cpu_workspace               \
  tmol_disulfide_cpu_##Suffix##_forward_workspace(int64_t n_pairs) {           \
    return disulfide_cpu_forward_workspace<Real>(n_pairs);                     \
  }                                                                            \
  extern "C" TMOL_DISULFIDE_CPU_API tmol_disulfide_cpu_workspace               \
  tmol_disulfide_cpu_##Suffix##_recompute_vjp_workspace(int64_t n_pairs) {     \
    return disulfide_cpu_recompute_vjp_workspace<Real>(n_pairs);               \
  }                                                                            \
  extern "C" TMOL_DISULFIDE_CPU_API int32_t                                    \
  tmol_disulfide_cpu_##Suffix##_forward(                                       \
      tmol_disulfide_cpu_executor executor,                                    \
      tmol_disulfide_cpu_##Suffix##_inputs inputs,                             \
      int32_t mode,                                                            \
      tmol_disulfide_cpu_##Suffix##_forward_outputs outputs) {                 \
    return static_cast<int32_t>(disulfide_cpu_forward<Real>(                   \
        executor,                                                              \
        cpp_inputs<Real>(inputs),                                              \
        static_cast<DisulfideForwardMode>(mode),                               \
        cpp_outputs<Real>(inputs.n_pairs, outputs)));                          \
  }                                                                            \
  extern "C" TMOL_DISULFIDE_CPU_API int32_t tmol_disulfide_cpu_##Suffix##_vjp( \
      tmol_disulfide_cpu_executor executor,                                    \
      int32_t mode,                                                            \
      int32_t const* pairs,                                                    \
      int64_t n_pairs,                                                         \
      int64_t pair_stride,                                                     \
      Real const* coordinate_derivatives,                                      \
      int64_t derivative_pair_stride,                                          \
      int64_t derivative_atom_stride,                                          \
      Real const* cotangents,                                                  \
      int64_t cotangent_stride,                                                \
      Real* coordinate_gradient,                                               \
      int64_t n_atoms,                                                         \
      int64_t gradient_stride) {                                               \
    return static_cast<int32_t>(c_vjp<Real>(                                   \
        executor,                                                              \
        mode,                                                                  \
        pairs,                                                                 \
        n_pairs,                                                               \
        pair_stride,                                                           \
        coordinate_derivatives,                                                \
        derivative_pair_stride,                                                \
        derivative_atom_stride,                                                \
        cotangents,                                                            \
        cotangent_stride,                                                      \
        coordinate_gradient,                                                   \
        n_atoms,                                                               \
        gradient_stride));                                                     \
  }                                                                            \
  extern "C" TMOL_DISULFIDE_CPU_API int32_t                                    \
  tmol_disulfide_cpu_##Suffix##_recompute_vjp(                                 \
      tmol_disulfide_cpu_executor executor,                                    \
      int32_t mode,                                                            \
      tmol_disulfide_cpu_##Suffix##_inputs inputs,                             \
      Real const* cotangents,                                                  \
      int64_t cotangent_stride,                                                \
      Real* coordinate_gradient,                                               \
      int64_t gradient_stride,                                                 \
      void* workspace,                                                         \
      size_t workspace_bytes) {                                                \
    return static_cast<int32_t>(c_recompute_vjp<Real>(                         \
        executor,                                                              \
        mode,                                                                  \
        inputs,                                                                \
        cotangents,                                                            \
        cotangent_stride,                                                      \
        coordinate_gradient,                                                   \
        gradient_stride,                                                       \
        workspace,                                                             \
        workspace_bytes));                                                     \
  }

TMOL_DEFINE_DISULFIDE_C_ABI(f32, float)
TMOL_DEFINE_DISULFIDE_C_ABI(f64, double)
#undef TMOL_DEFINE_DISULFIDE_C_ABI

}  // namespace potentials
}  // namespace disulfide
}  // namespace score
}  // namespace tmol
