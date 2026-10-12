#include "shared_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <limits>
#include <type_traits>

#include "params_core.hh"
#include "potentials.hh"

namespace tmol {
namespace score {
namespace disulfide {
namespace potentials {
namespace {

constexpr int kThreads = 128;

template <typename Real>
using Vec3 = Eigen::Matrix<Real, 3, 1>;
using Pair = Eigen::Matrix<int32_t, 8, 1>;

int launch_blocks(int64_t count) {
  if (count <= 0) return 0;
  return static_cast<int>(
      std::min<int64_t>((count + kThreads - 1) / kThreads, 65535));
}

cudaStream_t decode_stream(uintptr_t stream) {
  return reinterpret_cast<cudaStream_t>(stream);
}

int32_t launch_status(cudaError_t status) {
  return status == cudaSuccess ? TMOL_DISULFIDE_CUDA_SUCCESS
                               : TMOL_DISULFIDE_CUDA_LAUNCH_ERROR;
}

int32_t reset_device_status(
    cudaStream_t stream, int32_t* device_status, bool required) {
  if (required && !device_status) return TMOL_DISULFIDE_CUDA_INVALID_SHAPE;
  if (!device_status) return TMOL_DISULFIDE_CUDA_SUCCESS;
  return launch_status(
      cudaMemsetAsync(device_status, 0, sizeof(*device_status), stream));
}

__device__ void record_invalid_index(int32_t* status) {
  atomicCAS(
      status,
      TMOL_DISULFIDE_CUDA_SUCCESS,
      TMOL_DISULFIDE_CUDA_INVALID_INDEX);
}

template <typename Real, typename Inputs>
__global__ void forward_kernel(
    Inputs inputs,
    int32_t mode,
    Real* energies,
    int64_t energy_stride,
    Real* derivatives,
    int64_t derivative_pair_stride,
    int64_t derivative_atom_stride,
    bool write_energies,
    int32_t* status) {
  int64_t const thread = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const step = blockDim.x * gridDim.x;
  for (int64_t i = thread; i < inputs.n_pairs; i += step) {
    auto const* pair_values = inputs.pairs + i * inputs.pair_stride * 8;
    bool valid = true;
    for (int j = 0; j < 6; ++j) {
      valid = valid && pair_values[j] >= 0 && pair_values[j] < inputs.n_atoms;
    }
    valid = valid && pair_values[6] >= 0 && pair_values[6] < inputs.n_params
            && pair_values[7] >= 0 && pair_values[7] < inputs.n_params;
    if (!valid) {
      record_invalid_index(status);
      continue;
    }

    Vec3<Real> local_coords_storage[6];
    auto const* coords = reinterpret_cast<Vec3<Real> const*>(inputs.coords);
#pragma unroll
    for (int j = 0; j < 6; ++j) {
      local_coords_storage[j] = coords[pair_values[j] * inputs.coord_stride];
    }
    int64_t coord_sizes[1] = {6};
    int64_t coord_strides[1] = {1};
    TView<Vec3<Real>, 1, Device::CUDA> local_coords(
        local_coords_storage, coord_sizes, coord_strides);
    auto const* params =
        reinterpret_cast<DisulfideGlobalParams<Real> const*>(inputs.params);
    auto const& params1 = params[pair_values[6] * inputs.param_stride];
    auto const& params2 = params[pair_values[7] * inputs.param_stride];

    if (mode == TMOL_DISULFIDE_CUDA_INFERENCE) {
      energies[i * energy_stride] = disulfide_energy<Real, Device::CUDA>(
          local_coords, 0, 1, 2, 3, 4, 5, params1, params2);
      continue;
    }

    auto* derivative_row =
        reinterpret_cast<Vec3<Real>*>(derivatives)
        + i * derivative_pair_stride;
#pragma unroll
    for (int j = 0; j < 6; ++j) {
      derivative_row[j * derivative_atom_stride].setZero();
    }
    int64_t derivative_sizes[2] = {1, 6};
    int64_t derivative_strides[2] = {
        derivative_pair_stride, derivative_atom_stride};
    TView<Vec3<Real>, 2, Device::CUDA> derivative_view(
        derivative_row, derivative_sizes, derivative_strides);
    Real* energy = write_energies ? energies + i * energy_stride : nullptr;
    if (energy) *energy = 0;
    accumulate_disulfide<Real, Device::CUDA>(
        local_coords,
        0,
        1,
        2,
        3,
        4,
        5,
        params1,
        params2,
        derivative_view,
        Real(1),
        energy);
  }
}

template <typename Real>
__global__ void zero_gradient_kernel(
    Real* gradient, int64_t n_atoms, int64_t gradient_stride) {
  auto* values = reinterpret_cast<Vec3<Real>*>(gradient);
  int64_t const thread = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const step = blockDim.x * gridDim.x;
  for (int64_t i = thread; i < n_atoms; i += step) {
    values[i * gradient_stride].setZero();
  }
}

template <typename Real>
__global__ void vjp_kernel(
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
    int64_t gradient_stride,
    int32_t* status) {
  auto const* derivative_values =
      reinterpret_cast<Vec3<Real> const*>(derivatives);
  auto* gradient_values = reinterpret_cast<Vec3<Real>*>(gradient);
  int64_t const thread = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const step = blockDim.x * gridDim.x;
  for (int64_t i = thread; i < n_pairs; i += step) {
    auto const* pair_values = pairs + i * pair_stride * 8;
    bool valid = true;
    for (int j = 0; j < 6; ++j) {
      valid = valid && pair_values[j] >= 0 && pair_values[j] < n_atoms;
    }
    if (!valid) {
      record_invalid_index(status);
      continue;
    }
    Real scale = cotangents[i * cotangent_stride];
#pragma unroll
    for (int j = 0; j < 6; ++j) {
      auto value =
          scale
          * derivative_values
                [i * derivative_pair_stride + j * derivative_atom_stride];
      auto& destination = gradient_values[pair_values[j] * gradient_stride];
#pragma unroll
      for (int axis = 0; axis < 3; ++axis) {
        if (mode == TMOL_DISULFIDE_CUDA_MAY_OVERLAP) {
          atomicAdd(&destination[axis], value[axis]);
        } else {
          destination[axis] += value[axis];
        }
      }
    }
  }
}

template <typename Real>
tmol_disulfide_cuda_workspace recompute_workspace(int64_t n_pairs) {
  if (n_pairs <= 0) return {0, alignof(Vec3<Real>)};
  constexpr size_t row_bytes = 6 * sizeof(Vec3<Real>);
  constexpr auto max = std::numeric_limits<size_t>::max();
  if (static_cast<uint64_t>(n_pairs) > max / row_bytes) {
    return {max, alignof(Vec3<Real>)};
  }
  return {static_cast<size_t>(n_pairs) * row_bytes, alignof(Vec3<Real>)};
}

template <typename Real, typename Inputs, typename Outputs>
int32_t launch_forward(
    cudaStream_t stream,
    Inputs inputs,
    int32_t mode,
    Outputs outputs,
    int32_t* device_status,
    bool write_energies,
    bool reset_status) {
  if (mode != TMOL_DISULFIDE_CUDA_INFERENCE
      && mode != TMOL_DISULFIDE_CUDA_TRACKED) {
    return TMOL_DISULFIDE_CUDA_INVALID_MODE;
  }
  if (!write_energies && mode != TMOL_DISULFIDE_CUDA_TRACKED) {
    return TMOL_DISULFIDE_CUDA_INVALID_MODE;
  }
  if (inputs.n_atoms < 0 || inputs.n_pairs < 0 || inputs.n_params < 0
      || (inputs.n_pairs
          && (inputs.coord_stride < 1 || inputs.pair_stride < 1
              || inputs.param_stride < 1
              || (write_energies && outputs.energy_stride < 1)
              || !inputs.coords || !inputs.pairs || !inputs.params
              || (write_energies && !outputs.energies)))) {
    return TMOL_DISULFIDE_CUDA_INVALID_SHAPE;
  }
  if (mode == TMOL_DISULFIDE_CUDA_TRACKED
      && inputs.n_pairs
      && (outputs.derivative_pair_stride < 1
          || outputs.derivative_atom_stride < 1
          || !outputs.coordinate_derivatives)) {
    return TMOL_DISULFIDE_CUDA_MISSING_DERIVATIVE_OUTPUT;
  }
  if (reset_status) {
    auto status =
        reset_device_status(stream, device_status, inputs.n_pairs != 0);
    if (status != TMOL_DISULFIDE_CUDA_SUCCESS) return status;
  }
  int blocks = launch_blocks(inputs.n_pairs);
  if (!blocks) return TMOL_DISULFIDE_CUDA_SUCCESS;
  forward_kernel<Real><<<blocks, kThreads, 0, stream>>>(
      inputs,
      mode,
      outputs.energies,
      outputs.energy_stride,
      outputs.coordinate_derivatives,
      outputs.derivative_pair_stride,
      outputs.derivative_atom_stride,
      write_energies,
      device_status);
  return launch_status(cudaPeekAtLastError());
}

template <typename Real>
int32_t launch_vjp(
    cudaStream_t stream,
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
    int64_t gradient_stride,
    int32_t* device_status,
    bool reset_status) {
  if (mode != TMOL_DISULFIDE_CUDA_MAY_OVERLAP
      && mode != TMOL_DISULFIDE_CUDA_DISJOINT_PAIRS) {
    return TMOL_DISULFIDE_CUDA_INVALID_MODE;
  }
  if (n_pairs < 0 || n_atoms < 0
      || (n_pairs
          && (pair_stride < 1 || derivative_pair_stride < 1
              || derivative_atom_stride < 1 || cotangent_stride < 1
              || !pairs || !derivatives || !cotangents || !device_status))
      || (n_atoms && (!gradient || gradient_stride < 1))) {
    return TMOL_DISULFIDE_CUDA_INVALID_SHAPE;
  }
  if (reset_status) {
    auto status = reset_device_status(stream, device_status, n_pairs != 0);
    if (status != TMOL_DISULFIDE_CUDA_SUCCESS) return status;
  }
  int gradient_blocks = launch_blocks(n_atoms);
  if (gradient_blocks) {
    zero_gradient_kernel<Real><<<gradient_blocks, kThreads, 0, stream>>>(
        gradient, n_atoms, gradient_stride);
    auto status = launch_status(cudaPeekAtLastError());
    if (status != TMOL_DISULFIDE_CUDA_SUCCESS) return status;
  }
  int blocks = launch_blocks(n_pairs);
  if (!blocks) return TMOL_DISULFIDE_CUDA_SUCCESS;
  vjp_kernel<Real><<<blocks, kThreads, 0, stream>>>(
      mode,
      pairs,
      n_pairs,
      pair_stride,
      derivatives,
      derivative_pair_stride,
      derivative_atom_stride,
      cotangents,
      cotangent_stride,
      gradient,
      n_atoms,
      gradient_stride,
      device_status);
  return launch_status(cudaPeekAtLastError());
}

template <typename Real, typename Inputs>
int32_t launch_recompute_vjp(
    cudaStream_t stream,
    int32_t mode,
    Inputs inputs,
    Real const* cotangents,
    int64_t cotangent_stride,
    Real* gradient,
    int64_t gradient_stride,
    void* workspace,
    size_t workspace_bytes,
    int32_t* device_status) {
  auto required = recompute_workspace<Real>(inputs.n_pairs);
  if (workspace_bytes < required.bytes || (required.bytes && !workspace)
      || (required.bytes
          && reinterpret_cast<uintptr_t>(workspace) % required.alignment)) {
    return TMOL_DISULFIDE_CUDA_INSUFFICIENT_WORKSPACE;
  }
  auto status =
      reset_device_status(stream, device_status, inputs.n_pairs != 0);
  if (status != TMOL_DISULFIDE_CUDA_SUCCESS) return status;
  using Outputs = std::conditional_t<
      std::is_same_v<Real, float>,
      tmol_disulfide_cuda_f32_forward_outputs,
      tmol_disulfide_cuda_f64_forward_outputs>;
  Outputs outputs{
      nullptr, 1, static_cast<Real*>(workspace), 6, 1};
  status = launch_forward<Real>(
      stream,
      inputs,
      TMOL_DISULFIDE_CUDA_TRACKED,
      outputs,
      device_status,
      false,
      false);
  if (status != TMOL_DISULFIDE_CUDA_SUCCESS) return status;
  return launch_vjp<Real>(
      stream,
      mode,
      inputs.pairs,
      inputs.n_pairs,
      inputs.pair_stride,
      static_cast<Real*>(workspace),
      6,
      1,
      cotangents,
      cotangent_stride,
      gradient,
      inputs.n_atoms,
      gradient_stride,
      device_status,
      false);
}

}  // namespace
}  // namespace potentials
}  // namespace disulfide
}  // namespace score
}  // namespace tmol

using namespace tmol::score::disulfide::potentials;

extern "C" TMOL_DISULFIDE_CUDA_API uint32_t
tmol_disulfide_cuda_abi_version() {
  return TMOL_DISULFIDE_CUDA_ABI_VERSION;
}

extern "C" TMOL_DISULFIDE_CUDA_API char const*
tmol_disulfide_cuda_status_message(int32_t status) {
  switch (status) {
    case TMOL_DISULFIDE_CUDA_SUCCESS:
      return "success";
    case TMOL_DISULFIDE_CUDA_INVALID_MODE:
      return "invalid execution mode";
    case TMOL_DISULFIDE_CUDA_INVALID_SHAPE:
      return "invalid borrowed-view shape or pointer";
    case TMOL_DISULFIDE_CUDA_INVALID_INDEX:
      return "prepared pair index out of bounds";
    case TMOL_DISULFIDE_CUDA_MISSING_DERIVATIVE_OUTPUT:
      return "tracked forward requires pair coordinate derivatives";
    case TMOL_DISULFIDE_CUDA_INSUFFICIENT_WORKSPACE:
      return "insufficient or misaligned caller workspace";
    case TMOL_DISULFIDE_CUDA_LAUNCH_ERROR:
      return "CUDA enqueue or launch error";
  }
  return "unknown disulfide CUDA status";
}

#define TMOL_DEFINE_DISULFIDE_CUDA_ABI(Suffix, Real)                           \
  extern "C" TMOL_DISULFIDE_CUDA_API tmol_disulfide_cuda_workspace             \
  tmol_disulfide_cuda_##Suffix##_forward_workspace(int64_t) {                  \
    return {0, alignof(Vec3<Real>)};                                            \
  }                                                                            \
  extern "C" TMOL_DISULFIDE_CUDA_API tmol_disulfide_cuda_workspace             \
  tmol_disulfide_cuda_##Suffix##_recompute_vjp_workspace(int64_t n_pairs) {    \
    return recompute_workspace<Real>(n_pairs);                                  \
  }                                                                            \
  extern "C" TMOL_DISULFIDE_CUDA_API int32_t                                   \
  tmol_disulfide_cuda_##Suffix##_forward(                                      \
      uintptr_t stream,                                                        \
      tmol_disulfide_cuda_##Suffix##_inputs inputs,                            \
      int32_t mode,                                                            \
      tmol_disulfide_cuda_##Suffix##_forward_outputs outputs,                  \
      int32_t* device_status) {                                                \
    return launch_forward<Real>(                                               \
        decode_stream(stream),                                                 \
        inputs,                                                                \
        mode,                                                                  \
        outputs,                                                               \
        device_status,                                                         \
        true,                                                                  \
        true);                                                                 \
  }                                                                            \
  extern "C" TMOL_DISULFIDE_CUDA_API int32_t                                   \
  tmol_disulfide_cuda_##Suffix##_vjp(                                          \
      uintptr_t stream,                                                        \
      int32_t mode,                                                            \
      int32_t const* pairs,                                                    \
      int64_t n_pairs,                                                         \
      int64_t pair_stride,                                                     \
      Real const* derivatives,                                                 \
      int64_t derivative_pair_stride,                                          \
      int64_t derivative_atom_stride,                                          \
      Real const* cotangents,                                                  \
      int64_t cotangent_stride,                                                \
      Real* gradient,                                                          \
      int64_t n_atoms,                                                         \
      int64_t gradient_stride,                                                 \
      int32_t* device_status) {                                                \
    return launch_vjp<Real>(                                                   \
        decode_stream(stream),                                                 \
        mode,                                                                  \
        pairs,                                                                 \
        n_pairs,                                                               \
        pair_stride,                                                           \
        derivatives,                                                           \
        derivative_pair_stride,                                                \
        derivative_atom_stride,                                                \
        cotangents,                                                            \
        cotangent_stride,                                                      \
        gradient,                                                              \
        n_atoms,                                                               \
        gradient_stride,                                                       \
        device_status,                                                         \
        true);                                                                 \
  }                                                                            \
  extern "C" TMOL_DISULFIDE_CUDA_API int32_t                                   \
  tmol_disulfide_cuda_##Suffix##_recompute_vjp(                                \
      uintptr_t stream,                                                        \
      int32_t mode,                                                            \
      tmol_disulfide_cuda_##Suffix##_inputs inputs,                            \
      Real const* cotangents,                                                  \
      int64_t cotangent_stride,                                                \
      Real* gradient,                                                          \
      int64_t gradient_stride,                                                 \
      void* workspace,                                                         \
      size_t workspace_bytes,                                                  \
      int32_t* device_status) {                                                \
    return launch_recompute_vjp<Real>(                                         \
        decode_stream(stream),                                                 \
        mode,                                                                  \
        inputs,                                                                \
        cotangents,                                                            \
        cotangent_stride,                                                      \
        gradient,                                                              \
        gradient_stride,                                                       \
        workspace,                                                             \
        workspace_bytes,                                                       \
        device_status);                                                        \
  }

TMOL_DEFINE_DISULFIDE_CUDA_ABI(f32, float);
TMOL_DEFINE_DISULFIDE_CUDA_ABI(f64, double);
#undef TMOL_DEFINE_DISULFIDE_CUDA_ABI
