#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <Eigen/Core>

#include <tmol/score/disulfide/potentials/params_core.hh>
#include <tmol/score/disulfide/potentials/shared_cpu.h>
#include <tmol/score/disulfide/potentials/shared_cuda.h>

using Params =
    tmol::score::disulfide::potentials::DisulfideGlobalParams<double>;
using V3 = Eigen::Matrix<double, 3, 1>;
using Pair = Eigen::Matrix<int32_t, 8, 1>;

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::fprintf(stderr, "CHECK failed at line %d: %s\n", __LINE__, #condition); \
      return 1;                                                                \
    }                                                                          \
  } while (false)

#define CUDA_CHECK(expression) CHECK((expression) == cudaSuccess)

namespace {

Params parameters() {
  Params p{};
  double values[33] = {2.03, 0.15, -0.2,  0.1, 1.2,  1.8, 0.2,  1.1, -1.5,
                       -0.1, 0.8,  1.4,   0.1, -1.2, 0.7, -0.2, 0.1, 0.9,
                       0.05, 1.3,  1.0,   0.6, 0.4,  0.8, 1.1,  0.3, 0.15,
                       0.9,  -1.0, -0.05, 1.0, 1.0,  1.0};
  std::memcpy(&p, values, sizeof(p));
  return p;
}

template <typename T>
T* device_copy(T const* source, size_t count) {
  T* result = nullptr;
  if (cudaMalloc(&result, count * sizeof(T)) != cudaSuccess) return nullptr;
  if (cudaMemcpy(
          result, source, count * sizeof(T), cudaMemcpyHostToDevice)
      != cudaSuccess) {
    cudaFree(result);
    return nullptr;
  }
  return result;
}

}  // namespace

int main() {
  V3 base[] = {
      {-1.0, 1.0, 0.2},
      {0.0, 1.0, 0.0},
      {0.0, 0.0, 0.0},
      {2.05, 0.1, 0.1},
      {2.2, 1.0, 0.3},
      {3.0, 1.3, 0.9}};
  V3 coords[12];
  for (int atom = 0; atom < 6; ++atom) {
    coords[atom] = base[atom];
    coords[atom + 6] = base[atom] + V3(0.4, -0.2, 0.3);
  }
  Pair pairs[] = {
      {0, 1, 2, 3, 4, 5, 0, 1}, {6, 7, 8, 9, 10, 11, 1, 0}};
  Params params[] = {parameters(), parameters()};
  params[1].chirality = -1;
  params[1].dcs_mu1 *= -1;
  params[1].dcs_mu2 *= -1;
  params[1].dcs_mu3 *= -1;
  double cotangents[] = {0.7, -1.3};

  double cpu_energies[2] = {};
  V3 cpu_derivatives[2][6] = {};
  V3 cpu_gradient[12] = {};
  tmol_disulfide_cpu_f64_inputs cpu_inputs{
      coords[0].data(),
      12,
      1,
      pairs[0].data(),
      2,
      1,
      reinterpret_cast<double*>(params),
      2,
      1};
  tmol_disulfide_cpu_f64_forward_outputs cpu_outputs{
      cpu_energies, 1, cpu_derivatives[0][0].data(), 6, 1};
  CHECK(
      tmol_disulfide_cpu_f64_forward(
          tmol_disulfide_cpu_serial_executor(),
          cpu_inputs,
          TMOL_DISULFIDE_CPU_TRACKED,
          cpu_outputs)
      == TMOL_DISULFIDE_CPU_SUCCESS);
  CHECK(
      tmol_disulfide_cpu_f64_vjp(
          tmol_disulfide_cpu_serial_executor(),
          TMOL_DISULFIDE_CPU_DISJOINT_PAIRS,
          pairs[0].data(),
          2,
          1,
          cpu_derivatives[0][0].data(),
          6,
          1,
          cotangents,
          1,
          cpu_gradient[0].data(),
          12,
          1)
      == TMOL_DISULFIDE_CPU_SUCCESS);

  auto* d_coords = device_copy(coords, 12);
  auto* d_pairs = device_copy(pairs, 2);
  auto* d_params = device_copy(params, 2);
  auto* d_cotangents = device_copy(cotangents, 2);
  double *d_energies = nullptr, *d_derivatives = nullptr, *d_gradient = nullptr;
  int32_t* d_status = nullptr;
  CUDA_CHECK(cudaMalloc(&d_energies, 2 * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_derivatives, 2 * 6 * sizeof(V3)));
  CUDA_CHECK(cudaMalloc(&d_gradient, 12 * sizeof(V3)));
  CUDA_CHECK(cudaMalloc(&d_status, sizeof(int32_t)));
  CHECK(d_coords && d_pairs && d_params && d_cotangents);

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  uintptr_t stream_value = reinterpret_cast<uintptr_t>(stream);
  tmol_disulfide_cuda_f64_inputs inputs{
      reinterpret_cast<double*>(d_coords),
      12,
      1,
      reinterpret_cast<int32_t*>(d_pairs),
      2,
      1,
      reinterpret_cast<double*>(d_params),
      2,
      1};
  tmol_disulfide_cuda_f64_forward_outputs outputs{
      d_energies, 1, d_derivatives, 6, 1};
  CHECK(tmol_disulfide_cuda_abi_version() == TMOL_DISULFIDE_CUDA_ABI_VERSION);
  CHECK(
      tmol_disulfide_cuda_f64_forward(
          stream_value,
          inputs,
          TMOL_DISULFIDE_CUDA_TRACKED,
          outputs,
          d_status)
      == TMOL_DISULFIDE_CUDA_SUCCESS);
  CHECK(
      tmol_disulfide_cuda_f64_vjp(
          stream_value,
          TMOL_DISULFIDE_CUDA_MAY_OVERLAP,
          reinterpret_cast<int32_t*>(d_pairs),
          2,
          1,
          d_derivatives,
          6,
          1,
          d_cotangents,
          1,
          d_gradient,
          12,
          1,
          d_status)
      == TMOL_DISULFIDE_CUDA_SUCCESS);

  double gpu_energies[2] = {};
  V3 gpu_derivatives[2][6] = {}, gpu_gradient[12] = {};
  int32_t status = -1;
  CUDA_CHECK(cudaMemcpyAsync(
      gpu_energies,
      d_energies,
      sizeof(gpu_energies),
      cudaMemcpyDeviceToHost,
      stream));
  CUDA_CHECK(cudaMemcpyAsync(
      gpu_derivatives,
      d_derivatives,
      sizeof(gpu_derivatives),
      cudaMemcpyDeviceToHost,
      stream));
  CUDA_CHECK(cudaMemcpyAsync(
      gpu_gradient,
      d_gradient,
      sizeof(gpu_gradient),
      cudaMemcpyDeviceToHost,
      stream));
  CUDA_CHECK(cudaMemcpyAsync(
      &status, d_status, sizeof(status), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(status == TMOL_DISULFIDE_CUDA_SUCCESS);
  for (int pair = 0; pair < 2; ++pair) {
    CHECK(std::abs(gpu_energies[pair] - cpu_energies[pair]) < 1e-10);
    for (int atom = 0; atom < 6; ++atom)
      CHECK(
          (gpu_derivatives[pair][atom] - cpu_derivatives[pair][atom]).norm()
          < 1e-9);
  }
  for (int atom = 0; atom < 12; ++atom)
    CHECK((gpu_gradient[atom] - cpu_gradient[atom]).norm() < 1e-9);

  auto required = tmol_disulfide_cuda_f64_recompute_vjp_workspace(2);
  CHECK(required.bytes == 2 * 6 * sizeof(V3));
  void* d_workspace = nullptr;
  CUDA_CHECK(cudaMalloc(&d_workspace, required.bytes));
  CHECK(
      tmol_disulfide_cuda_f64_recompute_vjp(
          stream_value,
          TMOL_DISULFIDE_CUDA_DISJOINT_PAIRS,
          inputs,
          d_cotangents,
          1,
          d_gradient,
          1,
          d_workspace,
          required.bytes,
          d_status)
      == TMOL_DISULFIDE_CUDA_SUCCESS);
  CUDA_CHECK(cudaMemcpyAsync(
      gpu_gradient,
      d_gradient,
      sizeof(gpu_gradient),
      cudaMemcpyDeviceToHost,
      stream));
  CUDA_CHECK(cudaMemcpyAsync(
      &status, d_status, sizeof(status), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(status == TMOL_DISULFIDE_CUDA_SUCCESS);
  for (int atom = 0; atom < 12; ++atom)
    CHECK((gpu_gradient[atom] - cpu_gradient[atom]).norm() < 1e-9);

  Pair invalid_pairs[] = {pairs[0], pairs[1]};
  invalid_pairs[1][0] = 12;
  CUDA_CHECK(cudaMemcpyAsync(
      d_pairs, invalid_pairs, sizeof(invalid_pairs), cudaMemcpyHostToDevice, stream));
  CHECK(
      tmol_disulfide_cuda_f64_forward(
          stream_value,
          inputs,
          TMOL_DISULFIDE_CUDA_TRACKED,
          outputs,
          d_status)
      == TMOL_DISULFIDE_CUDA_SUCCESS);
  CUDA_CHECK(cudaMemcpyAsync(
      &status, d_status, sizeof(status), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(status == TMOL_DISULFIDE_CUDA_INVALID_INDEX);

  tmol_disulfide_cuda_f32_inputs empty_inputs{};
  tmol_disulfide_cuda_f32_forward_outputs empty_outputs{};
  CHECK(
      tmol_disulfide_cuda_f32_forward(
          0,
          empty_inputs,
          TMOL_DISULFIDE_CUDA_INFERENCE,
          empty_outputs,
          nullptr)
      == TMOL_DISULFIDE_CUDA_SUCCESS);

  CUDA_CHECK(cudaFree(d_workspace));
  CUDA_CHECK(cudaFree(d_status));
  CUDA_CHECK(cudaFree(d_gradient));
  CUDA_CHECK(cudaFree(d_derivatives));
  CUDA_CHECK(cudaFree(d_energies));
  CUDA_CHECK(cudaFree(d_cotangents));
  CUDA_CHECK(cudaFree(d_params));
  CUDA_CHECK(cudaFree(d_pairs));
  CUDA_CHECK(cudaFree(d_coords));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}
