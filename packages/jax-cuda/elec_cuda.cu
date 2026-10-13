#include <cuda_runtime_api.h>

#include <cstdint>

#include "xla/ffi/api/ffi.h"

#include <tmol/score/elec/potentials/potentials.hh>

namespace ffi = xla::ffi;
namespace pot = tmol::score::elec::potentials;

using F32R1 = ffi::BufferR1<ffi::DataType::F32>;
using F32R2 = ffi::BufferR2<ffi::DataType::F32>;
using F32R3 = ffi::BufferR3<ffi::DataType::F32>;
using U8R3 = ffi::BufferR3<ffi::DataType::U8>;
using PredR2 = ffi::BufferR2<ffi::DataType::PRED>;

constexpr int kTile = 16;
constexpr int kGlobalParams = 11;

__device__ void elec_tile_pair_from_linear(
    int index, int n_tiles, int& tile_i, int& tile_j) {
  int lo = 0;
  int hi = n_tiles;
  while (lo + 1 < hi) {
    int mid = (lo + hi) / 2;
    int start = mid * n_tiles - mid * (mid - 1) / 2;
    if (start <= index) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  tile_i = lo;
  int start = lo * n_tiles - lo * (lo - 1) / 2;
  tile_j = lo + index - start;
}

__device__ pot::ElecGlobalParams<float> elec_load_global(const float* values) {
  return {
      values[0],
      values[1],
      values[2],
      values[3],
      values[4],
      values[5],
      values[6],
      values[7],
      values[8],
      values[9],
      values[10]};
}

template <bool Gradient>
__global__ void elec_pairs(
    const float* coords,
    const float* charges,
    const uint8_t* bonded_path_length,
    const bool* atom_is_real,
    const float* global_values,
    const float* cotangent,
    int n_poses,
    int n_atoms,
    float* energies,
    float* gradients) {
  __shared__ int tile_i;
  __shared__ int tile_j;
  __shared__ float energy[kTile * kTile];

  int n_tiles = (n_atoms + kTile - 1) / kTile;
  int n_tile_pairs = n_tiles * (n_tiles + 1) / 2;
  int pose = blockIdx.x / n_tile_pairs;
  int tile_pair = blockIdx.x % n_tile_pairs;
  if (pose >= n_poses) return;
  if (threadIdx.x == 0) {
    elec_tile_pair_from_linear(tile_pair, n_tiles, tile_i, tile_j);
  }
  __syncthreads();

  int atom_i = tile_i * kTile + threadIdx.x / kTile;
  int atom_j = tile_j * kTile + threadIdx.x % kTile;
  int atom_base = pose * n_atoms;
  bool active = atom_i < n_atoms && atom_j < n_atoms && atom_i < atom_j;
  if (active) {
    active =
        atom_is_real[atom_base + atom_i] && atom_is_real[atom_base + atom_j];
  }

  float value = 0.0f;
  float delta[3] = {0.0f, 0.0f, 0.0f};
  if (active) {
    int cp = bonded_path_length[(atom_base + atom_i) * n_atoms + atom_j];
    active = cp >= 4;
    float charge_i = charges[atom_base + atom_i];
    float charge_j = charges[atom_base + atom_j];
    active = active && charge_i != 0.0f && charge_j != 0.0f;
    if (active) {
      int coord_i = (atom_base + atom_i) * 3;
      int coord_j = (atom_base + atom_j) * 3;
#pragma unroll
      for (int xyz = 0; xyz < 3; ++xyz) {
        delta[xyz] = coords[coord_i + xyz] - coords[coord_j + xyz];
      }
      float dist2 =
          delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
      pot::ElecGlobalParams<float> global = elec_load_global(global_values);
      active = dist2 < global.max_dis * global.max_dis;
      if (active) {
        float dist = sqrtf(dist2);
        auto result = pot::elec_delec_ddist<float>(
            dist, charge_i, charge_j, float(cp), global);
        if constexpr (Gradient) {
          float radial = cotangent[pose] * tmol::score::common::get<1>(result);
          if (dist != 0.0f && radial != 0.0f) {
#pragma unroll
            for (int xyz = 0; xyz < 3; ++xyz) {
              float deriv = radial * delta[xyz] / dist;
              atomicAdd(gradients + coord_i + xyz, deriv);
              atomicAdd(gradients + coord_j + xyz, -deriv);
            }
          }
        } else {
          value = tmol::score::common::get<0>(result);
        }
      }
    }
  }

  if constexpr (!Gradient) {
    energy[threadIdx.x] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
      if (threadIdx.x < stride) {
        energy[threadIdx.x] += energy[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(energies + pose, energy[0]);
  }
}

ffi::Error validate_shapes(
    F32R3 coords, F32R2 charges, U8R3 bpl, PredR2 real, F32R1 global) {
  auto c = coords.dimensions();
  auto q = charges.dimensions();
  auto b = bpl.dimensions();
  auto r = real.dimensions();
  if (c[2] != 3 || q[0] != c[0] || q[1] != c[1] || b[0] != c[0] || b[1] != c[1]
      || b[2] != c[1] || r[0] != c[0] || r[1] != c[1]
      || global.element_count() != kGlobalParams) {
    return ffi::Error::InvalidArgument(
        "incompatible electrostatics FFI shapes");
  }
  return ffi::Error::Success();
}

ffi::Error forward(
    cudaStream_t stream,
    F32R3 coords,
    F32R2 charges,
    U8R3 bpl,
    PredR2 real,
    F32R1 global,
    ffi::Result<F32R1> output) {
  if (auto error = validate_shapes(coords, charges, bpl, real, global);
      error.failure()) {
    return error;
  }
  int n_poses = coords.dimensions()[0];
  int n_atoms = coords.dimensions()[1];
  int n_tiles = (n_atoms + kTile - 1) / kTile;
  int blocks = n_poses * n_tiles * (n_tiles + 1) / 2;
  cudaMemsetAsync(output->typed_data(), 0, output->size_bytes(), stream);
  if (blocks == 0) return ffi::Error::Success();
  elec_pairs<false><<<blocks, kTile * kTile, 0, stream>>>(
      coords.typed_data(),
      charges.typed_data(),
      bpl.typed_data(),
      real.typed_data(),
      global.typed_data(),
      nullptr,
      n_poses,
      n_atoms,
      output->typed_data(),
      nullptr);
  if (auto status = cudaPeekAtLastError(); status != cudaSuccess) {
    return ffi::Error::Internal(cudaGetErrorString(status));
  }
  return ffi::Error::Success();
}

ffi::Error backward(
    cudaStream_t stream,
    F32R3 coords,
    F32R2 charges,
    U8R3 bpl,
    PredR2 real,
    F32R1 global,
    F32R1 cotangent,
    ffi::Result<F32R3> gradient) {
  if (auto error = validate_shapes(coords, charges, bpl, real, global);
      error.failure()) {
    return error;
  }
  int n_poses = coords.dimensions()[0];
  int n_atoms = coords.dimensions()[1];
  int n_tiles = (n_atoms + kTile - 1) / kTile;
  int blocks = n_poses * n_tiles * (n_tiles + 1) / 2;
  cudaMemsetAsync(gradient->typed_data(), 0, gradient->size_bytes(), stream);
  if (blocks == 0) return ffi::Error::Success();
  elec_pairs<true><<<blocks, kTile * kTile, 0, stream>>>(
      coords.typed_data(),
      charges.typed_data(),
      bpl.typed_data(),
      real.typed_data(),
      global.typed_data(),
      cotangent.typed_data(),
      n_poses,
      n_atoms,
      nullptr,
      gradient->typed_data());
  if (auto status = cudaPeekAtLastError(); status != cudaSuccess) {
    return ffi::Error::Internal(cudaGetErrorString(status));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_jax_elec_forward,
    forward,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<F32R3>()
        .Arg<F32R2>()
        .Arg<U8R3>()
        .Arg<PredR2>()
        .Arg<F32R1>()
        .Ret<F32R1>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_jax_elec_backward,
    backward,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<F32R3>()
        .Arg<F32R2>()
        .Arg<U8R3>()
        .Arg<PredR2>()
        .Arg<F32R1>()
        .Arg<F32R1>()
        .Ret<F32R3>());
