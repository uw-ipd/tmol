#include <cuda_runtime_api.h>

#include <cstdint>

#include "xla/ffi/api/ffi.h"
#include <tmol/score/lk_ball/potentials/scalar.hh>

namespace ffi = xla::ffi;
namespace pot = tmol::score::lk_ball::potentials;

using F32R1 = ffi::BufferR1<ffi::DataType::F32>;
using F32R2 = ffi::BufferR2<ffi::DataType::F32>;
using F32R3 = ffi::BufferR3<ffi::DataType::F32>;
using F32R4 = ffi::BufferR4<ffi::DataType::F32>;
using S32R2 = ffi::BufferR2<ffi::DataType::S32>;
using U8R3 = ffi::BufferR3<ffi::DataType::U8>;

constexpr int kTile = 16;
constexpr int kWaters = 4;
constexpr int kParams = 9;

ffi::Error validate_shapes(
    F32R3 coords,
    F32R4 waters,
    F32R3 params,
    U8R3 bpl,
    S32R2 polars,
    S32R2 occluders,
    F32R1 global) {
  auto c = coords.dimensions();
  auto w = waters.dimensions();
  auto p = params.dimensions();
  auto b = bpl.dimensions();
  auto pi = polars.dimensions();
  auto oi = occluders.dimensions();
  if (c[2] != 3 || w[0] != c[0] || w[1] != c[1] || w[2] != kWaters || w[3] != 3
      || p[0] != c[0] || p[1] != c[1] || p[2] != kParams || b[0] != c[0]
      || b[1] != c[1] || b[2] != c[1] || pi[0] != c[0] || oi[0] != c[0]
      || global.element_count() != 5) {
    return ffi::Error::InvalidArgument("incompatible LK-ball FFI shapes");
  }
  return ffi::Error::Success();
}

__device__ pot::LKBallTypeParams<float> load_lkball_params(
    const float* values) {
  return {
      values[0],
      values[1],
      values[2],
      values[3],
      values[4],
      values[5],
      values[6],
      values[7],
      values[8]};
}

__device__ pot::LKBallGlobalParams<float> load_lkball_global(
    const float* values) {
  return {values[0], values[1], values[2], values[3], values[4]};
}

template <bool Gradient>
__global__ void lkball_pairs(
    const float* coords,
    const float* waters,
    const float* params,
    const uint8_t* bonded_path_length,
    const int32_t* polar_indices,
    const int32_t* occluder_indices,
    const float* global_values,
    const float* cotangent,
    int n_poses,
    int n_atoms,
    int n_polar,
    int n_occluder,
    float* energies,
    float* coord_gradients,
    float* water_gradients) {
  using Real3 = Eigen::Matrix<float, 3, 1>;
  using Waters = Eigen::Matrix<float, kWaters, 3>;
  __shared__ float energy[4][kTile * kTile];

  int n_polar_tiles = (n_polar + kTile - 1) / kTile;
  int n_occluder_tiles = (n_occluder + kTile - 1) / kTile;
  int tiles_per_pose = n_polar_tiles * n_occluder_tiles;
  int pose = blockIdx.x / tiles_per_pose;
  int tile = blockIdx.x % tiles_per_pose;
  if (pose >= n_poses) return;
  int polar_slot = (tile / n_occluder_tiles) * kTile + threadIdx.x / kTile;
  int occluder_slot = (tile % n_occluder_tiles) * kTile + threadIdx.x % kTile;
  int polar =
      polar_slot < n_polar ? polar_indices[pose * n_polar + polar_slot] : -1;
  int occluder = occluder_slot < n_occluder
                     ? occluder_indices[pose * n_occluder + occluder_slot]
                     : -1;
  float values[4] = {0, 0, 0, 0};
  bool active = polar >= 0 && occluder >= 0 && polar != occluder;
  int atom_base = pose * n_atoms;
  int cp = 0;
  Real3 I;
  Real3 J;
  Waters WI;
  Waters WJ;
  pot::LKBallTypeParams<float> pi;
  pot::LKBallTypeParams<float> pj;
  pot::LKBallGlobalParams<float> global;
  if (active) {
    cp = bonded_path_length[(atom_base + polar) * n_atoms + occluder];
    active = cp > 3;
  }
  if (active) {
    int ibase = (atom_base + polar) * 3;
    int jbase = (atom_base + occluder) * 3;
#pragma unroll
    for (int d = 0; d < 3; ++d) {
      I[d] = coords[ibase + d];
      J[d] = coords[jbase + d];
    }
#pragma unroll
    for (int w = 0; w < kWaters; ++w) {
#pragma unroll
      for (int d = 0; d < 3; ++d) {
        WI(w, d) = waters[((atom_base + polar) * kWaters + w) * 3 + d];
        WJ(w, d) = waters[((atom_base + occluder) * kWaters + w) * 3 + d];
      }
    }
    pi = load_lkball_params(params + (atom_base + polar) * kParams);
    pj = load_lkball_params(params + (atom_base + occluder) * kParams);
    global = load_lkball_global(global_values);
    float dist2 = (I - J).squaredNorm();
    active = dist2 < global.distance_threshold * global.distance_threshold;
  }
  if (active) {
    if constexpr (Gradient) {
      Eigen::Matrix<float, 4, 1> weights;
#pragma unroll
      for (int score = 0; score < 4; ++score) {
        weights[score] = cotangent[pose * 4 + score];
      }
      auto deriv = pot::lk_ball_score<float, kWaters>::weighted_dV(
          I, J, WI, WJ, cp, pi, pj, global, weights);
      int ibase = (atom_base + polar) * 3;
      int jbase = (atom_base + occluder) * 3;
#pragma unroll
      for (int d = 0; d < 3; ++d) {
        atomicAdd(coord_gradients + ibase + d, deriv.dI[d]);
        atomicAdd(coord_gradients + jbase + d, deriv.dJ[d]);
      }
#pragma unroll
      for (int w = 0; w < kWaters; ++w) {
#pragma unroll
        for (int d = 0; d < 3; ++d) {
          atomicAdd(
              water_gradients + ((atom_base + polar) * kWaters + w) * 3 + d,
              deriv.dWI(w, d));
          atomicAdd(
              water_gradients + ((atom_base + occluder) * kWaters + w) * 3 + d,
              deriv.dWJ(w, d));
        }
      }
    } else {
      auto value = pot::lk_ball_score<float, kWaters>::V(
          I, J, WI, WJ, cp, pi, pj, global);
      values[0] = value.lkball_iso;
      values[1] = value.lkball;
      values[2] = value.lkbridge;
      values[3] = value.lkbridge_uncpl;
    }
  }

  if constexpr (!Gradient) {
#pragma unroll
    for (int score = 0; score < 4; ++score) {
      energy[score][threadIdx.x] = values[score];
    }
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
      if (threadIdx.x < stride) {
#pragma unroll
        for (int score = 0; score < 4; ++score) {
          energy[score][threadIdx.x] += energy[score][threadIdx.x + stride];
        }
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
#pragma unroll
      for (int score = 0; score < 4; ++score) {
        atomicAdd(energies + pose * 4 + score, energy[score][0]);
      }
    }
  }
}

ffi::Error forward(
    cudaStream_t stream,
    F32R3 coords,
    F32R4 waters,
    F32R3 params,
    U8R3 bpl,
    S32R2 polars,
    S32R2 occluders,
    F32R1 global,
    ffi::Result<F32R2> output) {
  if (auto error = validate_shapes(
          coords, waters, params, bpl, polars, occluders, global);
      error.failure()) {
    return error;
  }
  int poses = coords.dimensions()[0];
  int atoms = coords.dimensions()[1];
  int polar_count = polars.dimensions()[1];
  int occluder_count = occluders.dimensions()[1];
  int blocks = poses * ((polar_count + kTile - 1) / kTile)
               * ((occluder_count + kTile - 1) / kTile);
  cudaMemsetAsync(output->typed_data(), 0, output->size_bytes(), stream);
  if (blocks) {
    lkball_pairs<false><<<blocks, kTile * kTile, 0, stream>>>(
        coords.typed_data(),
        waters.typed_data(),
        params.typed_data(),
        bpl.typed_data(),
        polars.typed_data(),
        occluders.typed_data(),
        global.typed_data(),
        nullptr,
        poses,
        atoms,
        polar_count,
        occluder_count,
        output->typed_data(),
        nullptr,
        nullptr);
  }
  auto status = cudaPeekAtLastError();
  return status == cudaSuccess
             ? ffi::Error::Success()
             : ffi::Error::Internal(cudaGetErrorString(status));
}

ffi::Error backward(
    cudaStream_t stream,
    F32R3 coords,
    F32R4 waters,
    F32R3 params,
    U8R3 bpl,
    S32R2 polars,
    S32R2 occluders,
    F32R1 global,
    F32R2 cotangent,
    ffi::Result<F32R3> coord_gradient,
    ffi::Result<F32R4> water_gradient) {
  if (auto error = validate_shapes(
          coords, waters, params, bpl, polars, occluders, global);
      error.failure()) {
    return error;
  }
  if (cotangent.dimensions()[0] != coords.dimensions()[0]
      || cotangent.dimensions()[1] != 4) {
    return ffi::Error::InvalidArgument("incompatible LK-ball cotangent shape");
  }
  int poses = coords.dimensions()[0];
  int atoms = coords.dimensions()[1];
  int polar_count = polars.dimensions()[1];
  int occluder_count = occluders.dimensions()[1];
  int blocks = poses * ((polar_count + kTile - 1) / kTile)
               * ((occluder_count + kTile - 1) / kTile);
  cudaMemsetAsync(
      coord_gradient->typed_data(), 0, coord_gradient->size_bytes(), stream);
  cudaMemsetAsync(
      water_gradient->typed_data(), 0, water_gradient->size_bytes(), stream);
  if (blocks) {
    lkball_pairs<true><<<blocks, kTile * kTile, 0, stream>>>(
        coords.typed_data(),
        waters.typed_data(),
        params.typed_data(),
        bpl.typed_data(),
        polars.typed_data(),
        occluders.typed_data(),
        global.typed_data(),
        cotangent.typed_data(),
        poses,
        atoms,
        polar_count,
        occluder_count,
        nullptr,
        coord_gradient->typed_data(),
        water_gradient->typed_data());
  }
  auto status = cudaPeekAtLastError();
  return status == cudaSuccess
             ? ffi::Error::Success()
             : ffi::Error::Internal(cudaGetErrorString(status));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_jax_lkball_forward,
    forward,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<F32R3>()
        .Arg<F32R4>()
        .Arg<F32R3>()
        .Arg<U8R3>()
        .Arg<S32R2>()
        .Arg<S32R2>()
        .Arg<F32R1>()
        .Ret<F32R2>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_jax_lkball_backward,
    backward,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<F32R3>()
        .Arg<F32R4>()
        .Arg<F32R3>()
        .Arg<U8R3>()
        .Arg<S32R2>()
        .Arg<S32R2>()
        .Arg<F32R1>()
        .Arg<F32R2>()
        .Ret<F32R3>()
        .Ret<F32R4>());
