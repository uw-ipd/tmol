#pragma once

#include <cstddef>
#include <cstdint>

#include <Eigen/Core>
#include <tmol/utility/tensor/TensorAccessor.h>

#include "params_core.hh"
#include "shared_cpu.h"

namespace tmol {
namespace score {
namespace disulfide {
namespace potentials {

using CPUChunk = tmol_disulfide_cpu_chunk;
using CPUExecutor = tmol_disulfide_cpu_executor;

CPUExecutor serial_cpu_executor();

enum class DisulfideCPUStatus : int32_t {
  success = 0,
  invalid_executor,
  invalid_mode,
  invalid_shape,
  invalid_index,
  missing_derivative_output,
  insufficient_workspace,
};

enum class DisulfideForwardMode : int32_t { inference = 0, tracked = 1 };

enum class DisulfideVJPMode : int32_t {
  may_overlap = 0,
  // The caller certifies that atom indices do not repeat across pair rows.
  disjoint_pairs = 1,
};

using WorkspaceRequirements = tmol_disulfide_cpu_workspace;

template <typename Real>
using DisulfideVec3 = Eigen::Matrix<Real, 3, 1>;
using PreparedDisulfidePair = Eigen::Matrix<int32_t, 8, 1>;

// Pair entries are CA1, CB1, S1, S2, CB2, CA2, params1, params2.
template <typename Real>
struct DisulfideCPUInputs {
  TView<DisulfideVec3<Real>, 1, Device::CPU> coords;
  TView<PreparedDisulfidePair, 1, Device::CPU> pairs;
  TView<DisulfideGlobalParams<Real>, 1, Device::CPU> params;
};

template <typename Real>
struct DisulfideCPUForwardOutputs {
  TView<Real, 1, Device::CPU> energies;
  // Required only in tracked mode; shape [n_pairs, 6].
  TView<DisulfideVec3<Real>, 2, Device::CPU> coordinate_derivatives;
};

template <typename Real>
WorkspaceRequirements disulfide_cpu_forward_workspace(int64_t n_pairs);

template <typename Real>
WorkspaceRequirements disulfide_cpu_recompute_vjp_workspace(int64_t n_pairs);

template <typename Real>
DisulfideCPUStatus disulfide_cpu_forward(
    CPUExecutor executor,
    DisulfideCPUInputs<Real> inputs,
    DisulfideForwardMode mode,
    DisulfideCPUForwardOutputs<Real> outputs);

// Consumes derivatives retained by tracked forward and initializes gradient.
template <typename Real>
DisulfideCPUStatus disulfide_cpu_vjp(
    CPUExecutor executor,
    DisulfideVJPMode mode,
    TView<PreparedDisulfidePair, 1, Device::CPU> pairs,
    TView<DisulfideVec3<Real>, 2, Device::CPU> coordinate_derivatives,
    TView<Real, 1, Device::CPU> cotangents,
    TView<DisulfideVec3<Real>, 1, Device::CPU> coordinate_gradient);

// Explicit low-memory policy: recompute derivatives in caller-owned workspace.
template <typename Real>
DisulfideCPUStatus disulfide_cpu_recompute_vjp(
    CPUExecutor executor,
    DisulfideVJPMode mode,
    DisulfideCPUInputs<Real> inputs,
    TView<Real, 1, Device::CPU> cotangents,
    TView<DisulfideVec3<Real>, 1, Device::CPU> coordinate_gradient,
    void* workspace,
    size_t workspace_bytes);

#define TMOL_DECLARE_DISULFIDE_CPU(Real)                                       \
  extern template WorkspaceRequirements disulfide_cpu_forward_workspace<Real>( \
      int64_t);                                                                \
  extern template WorkspaceRequirements                                        \
  disulfide_cpu_recompute_vjp_workspace<Real>(int64_t);                        \
  extern template DisulfideCPUStatus disulfide_cpu_forward<Real>(              \
      CPUExecutor,                                                             \
      DisulfideCPUInputs<Real>,                                                \
      DisulfideForwardMode,                                                    \
      DisulfideCPUForwardOutputs<Real>);                                       \
  extern template DisulfideCPUStatus disulfide_cpu_vjp<Real>(                  \
      CPUExecutor,                                                             \
      DisulfideVJPMode,                                                        \
      TView<PreparedDisulfidePair, 1, Device::CPU>,                            \
      TView<DisulfideVec3<Real>, 2, Device::CPU>,                              \
      TView<Real, 1, Device::CPU>,                                             \
      TView<DisulfideVec3<Real>, 1, Device::CPU>);                             \
  extern template DisulfideCPUStatus disulfide_cpu_recompute_vjp<Real>(        \
      CPUExecutor,                                                             \
      DisulfideVJPMode,                                                        \
      DisulfideCPUInputs<Real>,                                                \
      TView<Real, 1, Device::CPU>,                                             \
      TView<DisulfideVec3<Real>, 1, Device::CPU>,                              \
      void*,                                                                   \
      size_t)

TMOL_DECLARE_DISULFIDE_CPU(float);
TMOL_DECLARE_DISULFIDE_CPU(double);
#undef TMOL_DECLARE_DISULFIDE_CPU

}  // namespace potentials
}  // namespace disulfide
}  // namespace score
}  // namespace tmol
