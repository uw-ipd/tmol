#include <cuda_runtime_api.h>

#include <cstdint>

#include "xla/ffi/api/ffi.h"

#include <tmol/score/hbond/potentials/potentials.hh>

namespace ffi = xla::ffi;
namespace pot = tmol::score::hbond::potentials;

using F32R1 = ffi::BufferR1<ffi::DataType::F32>;
using F32R3 = ffi::BufferR3<ffi::DataType::F32>;
using S32R2 = ffi::BufferR2<ffi::DataType::S32>;
using S32R3 = ffi::BufferR3<ffi::DataType::S32>;
using U32R4 = ffi::BufferR4<ffi::DataType::U32>;

constexpr int kTile = 16;
constexpr int kPolynomialWords = 30;

__device__ double hbond_load_double(const uint32_t* words) {
  union {
    uint64_t bits;
    double value;
  } decoded;
  decoded.bits = uint64_t(words[0]) | (uint64_t(words[1]) << 32);
  return decoded.value;
}

__device__ pot::HBondPoly<double> hbond_load_poly(const uint32_t* words) {
  pot::HBondPoly<double> result;
#pragma unroll
  for (int i = 0; i < 11; ++i) {
    result.coeffs[i] = hbond_load_double(words + 2 * i);
  }
  result.zero_pad = 0;
  result.range[0] = hbond_load_double(words + 22);
  result.range[1] = hbond_load_double(words + 24);
  result.bound[0] = hbond_load_double(words + 26);
  result.bound[1] = hbond_load_double(words + 28);
  return result;
}

__device__ pot::HBondGlobalParams<float> hbond_load_global(
    const float* values) {
  return {values[0], values[1], values[2], values[3], values[4], values[5]};
}

template <bool Gradient>
__global__ void hbond_pairs(
    const float* coords,
    const int32_t* donors,
    const int32_t* acceptors,
    const int32_t* separations,
    const float* pair_params,
    const uint32_t* polynomial_words,
    const float* global_values,
    const float* cotangent,
    int n_poses,
    int n_atoms,
    int n_donors,
    int n_acceptors,
    int n_donor_types,
    int n_acceptor_types,
    float* energies,
    float* gradients) {
  using Real3 = Eigen::Matrix<float, 3, 1>;
  __shared__ float energy[kTile * kTile];

  int donor_tiles = (n_donors + kTile - 1) / kTile;
  int acceptor_tiles = (n_acceptors + kTile - 1) / kTile;
  int tiles_per_pose = donor_tiles * acceptor_tiles;
  int pose = blockIdx.x / tiles_per_pose;
  int tile = blockIdx.x % tiles_per_pose;
  if (pose >= n_poses) return;

  int donor_slot = (tile / acceptor_tiles) * kTile + threadIdx.x / kTile;
  int acceptor_slot = (tile % acceptor_tiles) * kTile + threadIdx.x % kTile;
  bool active = donor_slot < n_donors && acceptor_slot < n_acceptors;
  int h = -1;
  int d = -1;
  int a = -1;
  int b = -1;
  int b0 = -1;
  int donor_type = 0;
  int acceptor_type = 0;
  if (active) {
    int donor_base = (pose * n_donors + donor_slot) * 3;
    int acceptor_base = (pose * n_acceptors + acceptor_slot) * 4;
    h = donors[donor_base];
    d = donors[donor_base + 1];
    donor_type = donors[donor_base + 2];
    a = acceptors[acceptor_base];
    b = acceptors[acceptor_base + 1];
    b0 = acceptors[acceptor_base + 2];
    acceptor_type = acceptors[acceptor_base + 3];
    int separation = separations
        [(pose * n_donors + donor_slot) * n_acceptors + acceptor_slot];
    active = h >= 0 && d >= 0 && a >= 0 && b >= 0 && b0 >= 0 && separation >= 5
             && donor_type >= 0 && donor_type < n_donor_types
             && acceptor_type >= 0 && acceptor_type < n_acceptor_types;
  }

  Real3 D;
  Real3 H;
  Real3 A;
  Real3 B;
  Real3 B0;
  if (active) {
    int atom_base = pose * n_atoms;
#pragma unroll
    for (int xyz = 0; xyz < 3; ++xyz) {
      D[xyz] = coords[(atom_base + d) * 3 + xyz];
      H[xyz] = coords[(atom_base + h) * 3 + xyz];
      A[xyz] = coords[(atom_base + a) * 3 + xyz];
      B[xyz] = coords[(atom_base + b) * 3 + xyz];
      B0[xyz] = coords[(atom_base + b0) * 3 + xyz];
    }
    float cutoff = global_values[5];
    active = (A - H).squaredNorm() < cutoff * cutoff;
  }

  float value = 0;
  if (active) {
    int pair = donor_type * n_acceptor_types + acceptor_type;
    pot::HBondPairParams<float> parameters = {
        pair_params[pair * 2 + 1], pair_params[pair * 2], 1.0f};
    const uint32_t* words = polynomial_words + pair * 3 * kPolynomialWords;
    pot::HBondPolynomials<double> polynomials = {
        hbond_load_poly(words),
        hbond_load_poly(words + kPolynomialWords),
        hbond_load_poly(words + 2 * kPolynomialWords)};
    auto globals = hbond_load_global(global_values);
    if constexpr (Gradient) {
      auto result = pot::hbond_score<float, int32_t>::V_dV(
          D, H, A, B, B0, parameters, polynomials, globals);
      float scale = cotangent[pose];
      int atom_base = pose * n_atoms;
#pragma unroll
      for (int xyz = 0; xyz < 3; ++xyz) {
        atomicAdd(
            gradients + (atom_base + d) * 3 + xyz, scale * result.dV_dD[xyz]);
        atomicAdd(
            gradients + (atom_base + h) * 3 + xyz, scale * result.dV_dH[xyz]);
        atomicAdd(
            gradients + (atom_base + a) * 3 + xyz, scale * result.dV_dA[xyz]);
        atomicAdd(
            gradients + (atom_base + b) * 3 + xyz, scale * result.dV_dB[xyz]);
        atomicAdd(
            gradients + (atom_base + b0) * 3 + xyz, scale * result.dV_dB0[xyz]);
      }
    } else {
      value = pot::hbond_score<float, int32_t>::V(
          D, H, A, B, B0, parameters, polynomials, globals);
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

ffi::Error hbond_validate_shapes(
    F32R3 coords,
    S32R3 donors,
    S32R3 acceptors,
    S32R3 separations,
    F32R3 pair_params,
    U32R4 polynomial_words,
    F32R1 globals) {
  auto c = coords.dimensions();
  auto d = donors.dimensions();
  auto a = acceptors.dimensions();
  auto s = separations.dimensions();
  auto p = pair_params.dimensions();
  auto w = polynomial_words.dimensions();
  if (c[2] != 3 || d[0] != c[0] || d[2] != 3 || a[0] != c[0] || a[2] != 4
      || s[0] != c[0] || s[1] != d[1] || s[2] != a[1] || p[2] != 2
      || w[0] != p[0] || w[1] != p[1] || w[2] != 3 || w[3] != kPolynomialWords
      || globals.element_count() != 6) {
    return ffi::Error::InvalidArgument("incompatible hydrogen-bond FFI shapes");
  }
  return ffi::Error::Success();
}

ffi::Error hbond_forward(
    cudaStream_t stream,
    F32R3 coords,
    S32R3 donors,
    S32R3 acceptors,
    S32R3 separations,
    F32R3 pair_params,
    U32R4 polynomial_words,
    F32R1 globals,
    ffi::Result<F32R1> output) {
  if (auto error = hbond_validate_shapes(
          coords,
          donors,
          acceptors,
          separations,
          pair_params,
          polynomial_words,
          globals);
      error.failure()) {
    return error;
  }
  int donor_tiles = (donors.dimensions()[1] + kTile - 1) / kTile;
  int acceptor_tiles = (acceptors.dimensions()[1] + kTile - 1) / kTile;
  int blocks = coords.dimensions()[0] * donor_tiles * acceptor_tiles;
  cudaMemsetAsync(output->typed_data(), 0, output->size_bytes(), stream);
  if (blocks) {
    hbond_pairs<false><<<blocks, kTile * kTile, 0, stream>>>(
        coords.typed_data(),
        donors.typed_data(),
        acceptors.typed_data(),
        separations.typed_data(),
        pair_params.typed_data(),
        polynomial_words.typed_data(),
        globals.typed_data(),
        nullptr,
        coords.dimensions()[0],
        coords.dimensions()[1],
        donors.dimensions()[1],
        acceptors.dimensions()[1],
        pair_params.dimensions()[0],
        pair_params.dimensions()[1],
        output->typed_data(),
        nullptr);
  }
  auto status = cudaPeekAtLastError();
  return status == cudaSuccess
             ? ffi::Error::Success()
             : ffi::Error::Internal(cudaGetErrorString(status));
}

ffi::Error hbond_backward(
    cudaStream_t stream,
    F32R3 coords,
    S32R3 donors,
    S32R3 acceptors,
    S32R3 separations,
    F32R3 pair_params,
    U32R4 polynomial_words,
    F32R1 globals,
    F32R1 cotangent,
    ffi::Result<F32R3> gradient) {
  if (auto error = hbond_validate_shapes(
          coords,
          donors,
          acceptors,
          separations,
          pair_params,
          polynomial_words,
          globals);
      error.failure()) {
    return error;
  }
  int donor_tiles = (donors.dimensions()[1] + kTile - 1) / kTile;
  int acceptor_tiles = (acceptors.dimensions()[1] + kTile - 1) / kTile;
  int blocks = coords.dimensions()[0] * donor_tiles * acceptor_tiles;
  cudaMemsetAsync(gradient->typed_data(), 0, gradient->size_bytes(), stream);
  if (blocks) {
    hbond_pairs<true><<<blocks, kTile * kTile, 0, stream>>>(
        coords.typed_data(),
        donors.typed_data(),
        acceptors.typed_data(),
        separations.typed_data(),
        pair_params.typed_data(),
        polynomial_words.typed_data(),
        globals.typed_data(),
        cotangent.typed_data(),
        coords.dimensions()[0],
        coords.dimensions()[1],
        donors.dimensions()[1],
        acceptors.dimensions()[1],
        pair_params.dimensions()[0],
        pair_params.dimensions()[1],
        nullptr,
        gradient->typed_data());
  }
  auto status = cudaPeekAtLastError();
  return status == cudaSuccess
             ? ffi::Error::Success()
             : ffi::Error::Internal(cudaGetErrorString(status));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_jax_hbond_forward,
    hbond_forward,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<F32R3>()
        .Arg<S32R3>()
        .Arg<S32R3>()
        .Arg<S32R3>()
        .Arg<F32R3>()
        .Arg<U32R4>()
        .Arg<F32R1>()
        .Ret<F32R1>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_jax_hbond_backward,
    hbond_backward,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<F32R3>()
        .Arg<S32R3>()
        .Arg<S32R3>()
        .Arg<S32R3>()
        .Arg<F32R3>()
        .Arg<U32R4>()
        .Arg<F32R1>()
        .Arg<F32R1>()
        .Ret<F32R3>());
