#include <cuda_runtime_api.h>

#include <cstdint>

#include "xla/ffi/api/ffi.h"

#include <tmol/score/ljlk/potentials/lj.hh>
#include <tmol/score/ljlk/potentials/lk_isotropic.hh>

// The shared scalar headers use constexpr std::min/max from device functions.
namespace ffi = xla::ffi;
namespace pot = tmol::score::ljlk::potentials;

using F32R1 = ffi::BufferR1<ffi::DataType::F32>;
using F32R2 = ffi::BufferR2<ffi::DataType::F32>;
using F32R3 = ffi::BufferR3<ffi::DataType::F32>;
using S32R3 = ffi::BufferR3<ffi::DataType::S32>;
using PredR2 = ffi::BufferR2<ffi::DataType::PRED>;

constexpr int kTile = 16;
constexpr int kParams = 13;

__device__ void tile_pair_from_linear(
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

__device__ pot::LJLKTypeParams<float> load_params(const float* values) {
  pot::LJLKTypeParams<float> out;
  out.lj_radius = values[0];
  out.lj_sqrt_wdepth = values[1];
  out.lk_dgfree = values[2];
  out.lk_lambda = values[3];
  out.lk_volume = values[4];
  out.is_donor = values[5];
  out.is_hydroxyl = values[6];
  out.is_polarh = values[7];
  out.is_acceptor = values[8];
  out.is_carbon_lk = values[9];
  out.is_hydrogen = values[10];
  out.lk_coeff = values[11];
  out.lk_inv_lambda2 = values[12];
  return out;
}

__device__ pot::LJGlobalParams<float> load_global(const float* values) {
  return {values[0], values[1], values[2], values[3], values[4]};
}

template <bool Gradient>
__global__ void ljlk_pairs(
    const float* coords,
    const float* params,
    const int32_t* bonded_path_length,
    const bool* atom_is_real,
    const float* global_values,
    const float* cotangent,
    int n_poses,
    int n_atoms,
    float* energies,
    float* gradients) {
  __shared__ int tile_i;
  __shared__ int tile_j;
  __shared__ float energy[3][kTile * kTile];

  int n_tiles = (n_atoms + kTile - 1) / kTile;
  int n_tile_pairs = n_tiles * (n_tiles + 1) / 2;
  int pose = blockIdx.x / n_tile_pairs;
  int tile_pair = blockIdx.x % n_tile_pairs;
  if (pose >= n_poses) return;
  if (threadIdx.x == 0) {
    tile_pair_from_linear(tile_pair, n_tiles, tile_i, tile_j);
  }
  __syncthreads();

  int atom_i = tile_i * kTile + threadIdx.x / kTile;
  int atom_j = tile_j * kTile + threadIdx.x % kTile;
  float values[3] = {0.0f, 0.0f, 0.0f};
  bool active = atom_i < n_atoms && atom_j < n_atoms && atom_i < atom_j;
  int atom_base = pose * n_atoms;
  if (active) {
    active = atom_is_real[atom_base + atom_i]
             && atom_is_real[atom_base + atom_j];
  }

  float delta[3] = {0.0f, 0.0f, 0.0f};
  float dist = 0.0f;
  int cp = 0;
  pot::LJLKTypeParams<float> p_i;
  pot::LJLKTypeParams<float> p_j;
  pot::LJGlobalParams<float> global;
  if (active) {
    int coord_i = (atom_base + atom_i) * 3;
    int coord_j = (atom_base + atom_j) * 3;
#pragma unroll
    for (int xyz = 0; xyz < 3; ++xyz) {
      delta[xyz] = coords[coord_i + xyz] - coords[coord_j + xyz];
    }
    float dist2 = delta[0] * delta[0] + delta[1] * delta[1]
                  + delta[2] * delta[2];
    global = load_global(global_values);
    active = dist2 < global.max_dis * global.max_dis;
    if (active) {
      cp = bonded_path_length[(atom_base + atom_i) * n_atoms + atom_j];
      active = cp >= 4;
    }
    if (active) {
      dist = sqrtf(dist2);
      p_i = load_params(params + (atom_base + atom_i) * kParams);
      p_j = load_params(params + (atom_base + atom_j) * kParams);
      auto lj = pot::lj_score<float>::V_dV(
          dist, cp, p_i.lj_params(), p_j.lj_params(), global);
      auto lk = pot::lk_isotropic_score<float>::V_dV(
          dist, cp, p_i.lk_params(), p_j.lk_params(), global);
      if constexpr (Gradient) {
        float radial = cotangent[pose * 3] * lj.dVatr_ddist
                       + cotangent[pose * 3 + 1] * lj.dVrep_ddist
                       + cotangent[pose * 3 + 2] * lk.dV_ddist;
        if (dist != 0.0f && radial != 0.0f) {
          int coord_i = (atom_base + atom_i) * 3;
          int coord_j = (atom_base + atom_j) * 3;
#pragma unroll
          for (int xyz = 0; xyz < 3; ++xyz) {
            float deriv = radial * delta[xyz] / dist;
            atomicAdd(gradients + coord_i + xyz, deriv);
            atomicAdd(gradients + coord_j + xyz, -deriv);
          }
        }
      } else {
        values[0] = lj.Vatr;
        values[1] = lj.Vrep;
        values[2] = lk.V;
      }
    }
  }

  if constexpr (!Gradient) {
#pragma unroll
    for (int score_type = 0; score_type < 3; ++score_type) {
      energy[score_type][threadIdx.x] = values[score_type];
    }
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
      if (threadIdx.x < stride) {
#pragma unroll
        for (int score_type = 0; score_type < 3; ++score_type) {
          energy[score_type][threadIdx.x] +=
              energy[score_type][threadIdx.x + stride];
        }
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
#pragma unroll
      for (int score_type = 0; score_type < 3; ++score_type) {
        atomicAdd(energies + pose * 3 + score_type, energy[score_type][0]);
      }
    }
  }
}

ffi::Error validate_shapes(
    F32R3 coords,
    F32R3 params,
    S32R3 bpl,
    PredR2 real,
    F32R1 global) {
  auto c = coords.dimensions();
  auto p = params.dimensions();
  auto b = bpl.dimensions();
  auto r = real.dimensions();
  if (c[2] != 3 || p[0] != c[0] || p[1] != c[1]
      || p[2] != kParams || b[0] != c[0] || b[1] != c[1]
      || b[2] != c[1] || r[0] != c[0] || r[1] != c[1]
      || global.element_count() != 5) {
    return ffi::Error::InvalidArgument("incompatible LJ/LK FFI shapes");
  }
  return ffi::Error::Success();
}

ffi::Error forward(
    cudaStream_t stream,
    F32R3 coords,
    F32R3 params,
    S32R3 bpl,
    PredR2 real,
    F32R1 global,
    ffi::Result<F32R2> output) {
  if (auto error = validate_shapes(coords, params, bpl, real, global);
      error.failure()) {
    return error;
  }
  int n_poses = coords.dimensions()[0];
  int n_atoms = coords.dimensions()[1];
  int n_tiles = (n_atoms + kTile - 1) / kTile;
  int blocks = n_poses * n_tiles * (n_tiles + 1) / 2;
  cudaMemsetAsync(output->typed_data(), 0, output->size_bytes(), stream);
  if (blocks == 0) return ffi::Error::Success();
  ljlk_pairs<false><<<blocks, kTile * kTile, 0, stream>>>(
      coords.typed_data(),
      params.typed_data(),
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
    F32R3 params,
    S32R3 bpl,
    PredR2 real,
    F32R1 global,
    F32R2 cotangent,
    ffi::Result<F32R3> gradient) {
  if (auto error = validate_shapes(coords, params, bpl, real, global);
      error.failure()) {
    return error;
  }
  int n_poses = coords.dimensions()[0];
  int n_atoms = coords.dimensions()[1];
  int n_tiles = (n_atoms + kTile - 1) / kTile;
  int blocks = n_poses * n_tiles * (n_tiles + 1) / 2;
  cudaMemsetAsync(gradient->typed_data(), 0, gradient->size_bytes(), stream);
  if (blocks == 0) return ffi::Error::Success();
  ljlk_pairs<true><<<blocks, kTile * kTile, 0, stream>>>(
      coords.typed_data(),
      params.typed_data(),
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
    tmol_jax_ljlk_forward,
    forward,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<F32R3>()
        .Arg<F32R3>()
        .Arg<S32R3>()
        .Arg<PredR2>()
        .Arg<F32R1>()
        .Ret<F32R2>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_jax_ljlk_backward,
    backward,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<F32R3>()
        .Arg<F32R3>()
        .Arg<S32R3>()
        .Arg<PredR2>()
        .Arg<F32R1>()
        .Arg<F32R2>()
        .Ret<F32R3>());
