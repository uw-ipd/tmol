#include <cmath>
#include <cstring>
#include <cstdio>
#include <vector>

#include <Eigen/Core>
#include <tmol/score/disulfide/potentials/params_core.hh>
#include <tmol/score/disulfide/potentials/shared_cpu.h>

using Params =
    tmol::score::disulfide::potentials::DisulfideGlobalParams<double>;
using V3 = Eigen::Matrix<double, 3, 1>;
using Pair = Eigen::Matrix<int32_t, 8, 1>;

#define CHECK(condition)                                                  \
  do {                                                                    \
    if (!(condition)) {                                                   \
      std::fprintf(                                                       \
          stderr, "CHECK failed at line %d: %s\n", __LINE__, #condition); \
      return 1;                                                           \
    }                                                                     \
  } while (false)

namespace {

Params parameters() {
  Params p{};
  double values[33] = {2.03, 0.15, -0.2,  0.1, 1.2,  1.8, 0.2,  1.1, -1.5,
                       -0.1, 0.8,  1.4,   0.1, -1.2, 0.7, -0.2, 0.1, 0.9,
                       0.05, 1.3,  1.0,   0.6, 0.4,  0.8, 1.1,  0.3, 0.15,
                       0.9,  -1.0, -0.05, 1.0, 1.0,  1.0};
  static_assert(sizeof(p) == sizeof(values));
  std::memcpy(&p, values, sizeof(p));
  return p;
}

struct CountingExecutor {
  int calls = 0;
  int64_t last_count = -1;
};

void counted_parallel_for(
    void* opaque, int64_t count, tmol_disulfide_cpu_chunk chunk, void* task) {
  auto& executor = *static_cast<CountingExecutor*>(opaque);
  ++executor.calls;
  executor.last_count = count;
  auto middle = count / 2;
  chunk(task, 0, middle);
  chunk(task, middle, count);
}
void reverse_counted_parallel_for(
    void* opaque, int64_t count, tmol_disulfide_cpu_chunk chunk, void* task) {
  auto& executor = *static_cast<CountingExecutor*>(opaque);
  ++executor.calls;
  executor.last_count = count;
  auto middle = count / 2;
  chunk(task, middle, count);
  chunk(task, 0, middle);
}

}  // namespace

int main() {
  static_assert(sizeof(V3) == 3 * sizeof(double));
  static_assert(sizeof(Pair) == 8 * sizeof(int32_t));
  V3 coords[] = {
      {-1.0, 1.0, 0.2},
      {0.0, 1.0, 0.0},
      {0.0, 0.0, 0.0},
      {2.05, 0.1, 0.1},
      {2.2, 1.0, 0.3},
      {3.0, 1.3, 0.9}};
  Pair pairs[] = {{0, 1, 2, 3, 4, 5, 0, 1}};
  Params params[] = {parameters(), parameters()};
  params[1].chirality = -1;
  params[1].dcs_mu1 *= -1;
  params[1].dcs_mu2 *= -1;
  params[1].dcs_mu3 *= -1;
  double energies[1] = {};
  V3 derivatives[1][6] = {};
  double cotangents[1] = {0.7};
  V3 gradient[6] = {}, recomputed_gradient[6] = {};
  tmol_disulfide_cpu_f64_inputs inputs{
      coords[0].data(),
      6,
      1,
      pairs[0].data(),
      1,
      1,
      reinterpret_cast<double*>(params),
      2,
      1};
  tmol_disulfide_cpu_f64_forward_outputs outputs{
      energies, 1, derivatives[0][0].data(), 6, 1};
  CountingExecutor count;
  tmol_disulfide_cpu_executor executor{&count, counted_parallel_for};

  CHECK(tmol_disulfide_cpu_abi_version() == TMOL_DISULFIDE_CPU_ABI_VERSION);
  CHECK(tmol_disulfide_cpu_status_message(TMOL_DISULFIDE_CPU_SUCCESS)[0]);
  CHECK(tmol_disulfide_cpu_f64_forward_workspace(1).bytes == 0);
  CHECK(
      tmol_disulfide_cpu_f64_forward(
          executor, inputs, TMOL_DISULFIDE_CPU_TRACKED, outputs)
      == TMOL_DISULFIDE_CPU_SUCCESS);
  CHECK(count.calls == 1 && count.last_count == 1);
  CHECK(std::isfinite(energies[0]));
  double inference_energy = 0;
  tmol_disulfide_cpu_f64_forward_outputs inference_outputs{
      &inference_energy, 1, nullptr, 0, 0};
  CHECK(
      tmol_disulfide_cpu_f64_forward(
          executor, inputs, TMOL_DISULFIDE_CPU_INFERENCE, inference_outputs)
      == TMOL_DISULFIDE_CPU_SUCCESS);
  CHECK(std::abs(inference_energy - energies[0]) < 1e-12);

  auto invalid_stride_inputs = inputs;
  invalid_stride_inputs.coord_stride = 0;
  CHECK(
      tmol_disulfide_cpu_f64_forward(
          executor,
          invalid_stride_inputs,
          TMOL_DISULFIDE_CPU_INFERENCE,
          inference_outputs)
      == TMOL_DISULFIDE_CPU_INVALID_SHAPE);
  auto invalid_stride_outputs = outputs;
  invalid_stride_outputs.derivative_atom_stride = 0;
  CHECK(
      tmol_disulfide_cpu_f64_forward(
          executor,
          inputs,
          TMOL_DISULFIDE_CPU_TRACKED,
          invalid_stride_outputs)
      == TMOL_DISULFIDE_CPU_MISSING_DERIVATIVE_OUTPUT);

  Pair chirality_pairs[] = {
      {0, 1, 2, 3, 4, 5, 0, 0},
      {0, 1, 2, 3, 4, 5, 0, 1},
      {0, 1, 2, 3, 4, 5, 1, 0},
      {0, 1, 2, 3, 4, 5, 1, 1}};
  double chirality_energies[4] = {}, chirality_tracked_energies[4] = {};
  V3 chirality_derivatives[4][6] = {};
  auto chirality_inputs = inputs;
  chirality_inputs.pairs = chirality_pairs[0].data();
  chirality_inputs.n_pairs = 4;
  tmol_disulfide_cpu_f64_forward_outputs chirality_outputs{
      chirality_energies, 1, nullptr, 0, 0};
  CHECK(
      tmol_disulfide_cpu_f64_forward(
          executor,
          chirality_inputs,
          TMOL_DISULFIDE_CPU_INFERENCE,
          chirality_outputs)
      == TMOL_DISULFIDE_CPU_SUCCESS);
  chirality_outputs = {
      chirality_tracked_energies, 1, chirality_derivatives[0][0].data(), 6, 1};
  CHECK(
      tmol_disulfide_cpu_f64_forward(
          executor,
          chirality_inputs,
          TMOL_DISULFIDE_CPU_TRACKED,
          chirality_outputs)
      == TMOL_DISULFIDE_CPU_SUCCESS);
  for (int pair = 0; pair < 4; ++pair) {
    CHECK(std::isfinite(chirality_energies[pair]));
    CHECK(
        std::abs(chirality_energies[pair] - chirality_tracked_energies[pair])
        < 1e-12);
  }

  constexpr double epsilon = 1e-6;
  for (int atom = 0; atom < 6; ++atom) {
    for (int axis = 0; axis < 3; ++axis) {
      double original = coords[atom][axis], plus = 0, minus = 0;
      coords[atom][axis] = original + epsilon;
      inference_outputs.energies = &plus;
      CHECK(
          tmol_disulfide_cpu_f64_forward(
              tmol_disulfide_cpu_serial_executor(),
              inputs,
              TMOL_DISULFIDE_CPU_INFERENCE,
              inference_outputs)
          == TMOL_DISULFIDE_CPU_SUCCESS);
      coords[atom][axis] = original - epsilon;
      inference_outputs.energies = &minus;
      CHECK(
          tmol_disulfide_cpu_f64_forward(
              tmol_disulfide_cpu_serial_executor(),
              inputs,
              TMOL_DISULFIDE_CPU_INFERENCE,
              inference_outputs)
          == TMOL_DISULFIDE_CPU_SUCCESS);
      coords[atom][axis] = original;
      CHECK(
          std::abs((plus - minus) / (2 * epsilon) - derivatives[0][atom][axis])
          < 2e-5);
    }
  }

  CHECK(
      tmol_disulfide_cpu_f64_vjp(
          executor,
          TMOL_DISULFIDE_CPU_DISJOINT_PAIRS,
          pairs[0].data(),
          1,
          1,
          derivatives[0][0].data(),
          6,
          1,
          cotangents,
          1,
          gradient[0].data(),
          6,
          1)
      == TMOL_DISULFIDE_CPU_SUCCESS);
  for (int atom = 0; atom < 6; ++atom) {
    CHECK(
        (gradient[atom] - cotangents[0] * derivatives[0][atom]).norm() < 1e-12);
  }
  CHECK(
      tmol_disulfide_cpu_f64_vjp(
          executor,
          TMOL_DISULFIDE_CPU_DISJOINT_PAIRS,
          pairs[0].data(),
          1,
          1,
          derivatives[0][0].data(),
          6,
          1,
          cotangents,
          0,
          gradient[0].data(),
          6,
          1)
      == TMOL_DISULFIDE_CPU_INVALID_SHAPE);

  Pair overlapping_pairs[] = {pairs[0], pairs[0]};
  V3 overlapping_derivatives[2][6];
  for (int pair = 0; pair < 2; ++pair)
    for (int atom = 0; atom < 6; ++atom)
      overlapping_derivatives[pair][atom] = derivatives[0][atom];
  double overlapping_cotangents[] = {0.2, 0.3};
  V3 overlapping_gradient[6];
  int calls_before_overlap = count.calls;
  CHECK(
      tmol_disulfide_cpu_f64_vjp(
          executor,
          TMOL_DISULFIDE_CPU_MAY_OVERLAP,
          overlapping_pairs[0].data(),
          2,
          1,
          overlapping_derivatives[0][0].data(),
          6,
          1,
          overlapping_cotangents,
          1,
          overlapping_gradient[0].data(),
          6,
          1)
      == TMOL_DISULFIDE_CPU_SUCCESS);
  CHECK(count.calls == calls_before_overlap + 1 && count.last_count == 1);
  for (int atom = 0; atom < 6; ++atom)
    CHECK(
        (overlapping_gradient[atom] - 0.5 * derivatives[0][atom]).norm()
        < 1e-12);

  auto required = tmol_disulfide_cpu_f64_recompute_vjp_workspace(1);
  std::vector<V3> workspace(6);
  CHECK(required.bytes == workspace.size() * sizeof(V3));
  CHECK(
      tmol_disulfide_cpu_f64_recompute_vjp(
          executor,
          TMOL_DISULFIDE_CPU_DISJOINT_PAIRS,
          inputs,
          cotangents,
          1,
          recomputed_gradient[0].data(),
          1,
          workspace.data(),
          required.bytes)
      == TMOL_DISULFIDE_CPU_SUCCESS);
  for (int atom = 0; atom < 6; ++atom)
    CHECK((recomputed_gradient[atom] - gradient[atom]).norm() < 1e-12);
  CHECK(
      tmol_disulfide_cpu_f64_recompute_vjp(
          executor,
          TMOL_DISULFIDE_CPU_DISJOINT_PAIRS,
          inputs,
          cotangents,
          1,
          recomputed_gradient[0].data(),
          1,
          workspace.data(),
          required.bytes - 1)
      == TMOL_DISULFIDE_CPU_INSUFFICIENT_WORKSPACE);
  CHECK(
      tmol_disulfide_cpu_f64_recompute_vjp(
          executor,
          TMOL_DISULFIDE_CPU_DISJOINT_PAIRS,
          inputs,
          cotangents,
          0,
          recomputed_gradient[0].data(),
          1,
          workspace.data(),
          required.bytes)
      == TMOL_DISULFIDE_CPU_INVALID_SHAPE);
  V3 multipair_coords[12];
  for (int atom = 0; atom < 6; ++atom) {
    multipair_coords[atom] = coords[atom];
    multipair_coords[atom + 6] = coords[atom] + V3(0.4, -0.2, 0.3);
  }
  Pair multipairs[] = {{0, 1, 2, 3, 4, 5, 0, 1}, {6, 7, 8, 9, 10, 11, 1, 0}};
  double multipair_energies[2] = {};
  V3 multipair_derivatives[2][6] = {};
  double multipair_cotangents[2] = {0.7, -1.3};
  V3 multipair_expected_gradient[12] = {}, multipair_gradient[12] = {};
  auto multipair_inputs = inputs;
  multipair_inputs.coords = multipair_coords[0].data();
  multipair_inputs.n_atoms = 12;
  multipair_inputs.pairs = multipairs[0].data();
  multipair_inputs.n_pairs = 2;
  tmol_disulfide_cpu_f64_forward_outputs multipair_outputs{
      multipair_energies, 1, multipair_derivatives[0][0].data(), 6, 1};
  CountingExecutor reverse_count;
  tmol_disulfide_cpu_executor reverse_executor{
      &reverse_count, reverse_counted_parallel_for};
  CHECK(
      tmol_disulfide_cpu_f64_forward(
          reverse_executor,
          multipair_inputs,
          TMOL_DISULFIDE_CPU_TRACKED,
          multipair_outputs)
      == TMOL_DISULFIDE_CPU_SUCCESS);
  CHECK(
      tmol_disulfide_cpu_f64_vjp(
          reverse_executor,
          TMOL_DISULFIDE_CPU_DISJOINT_PAIRS,
          multipairs[0].data(),
          2,
          1,
          multipair_derivatives[0][0].data(),
          6,
          1,
          multipair_cotangents,
          1,
          multipair_expected_gradient[0].data(),
          12,
          1)
      == TMOL_DISULFIDE_CPU_SUCCESS);
  auto multipair_required = tmol_disulfide_cpu_f64_recompute_vjp_workspace(2);
  CHECK(multipair_required.bytes == 2 * 6 * sizeof(V3));
  std::vector<double> multipair_workspace(
      multipair_required.bytes / sizeof(double));
  CHECK(
      tmol_disulfide_cpu_f64_recompute_vjp(
          reverse_executor,
          TMOL_DISULFIDE_CPU_DISJOINT_PAIRS,
          multipair_inputs,
          multipair_cotangents,
          1,
          multipair_gradient[0].data(),
          1,
          multipair_workspace.data(),
          multipair_required.bytes)
      == TMOL_DISULFIDE_CPU_SUCCESS);
  for (int atom = 0; atom < 12; ++atom)
    CHECK(
        (multipair_gradient[atom] - multipair_expected_gradient[atom]).norm()
        < 1e-12);

  float f32_coords[18], f32_params[66], f32_energy = 0;
  for (int atom = 0; atom < 6; ++atom)
    for (int axis = 0; axis < 3; ++axis)
      f32_coords[3 * atom + axis] = static_cast<float>(coords[atom][axis]);
  auto const* double_params = reinterpret_cast<double const*>(params);
  for (int i = 0; i < 66; ++i)
    f32_params[i] = static_cast<float>(double_params[i]);
  CHECK(tmol_disulfide_cpu_f32_forward_workspace(0).bytes == 0);
  tmol_disulfide_cpu_f32_inputs f32_inputs{
      f32_coords, 6, 1, pairs[0].data(), 1, 1, f32_params, 2, 1};
  tmol_disulfide_cpu_f32_forward_outputs f32_outputs{
      &f32_energy, 1, nullptr, 0, 0};
  CHECK(
      tmol_disulfide_cpu_f32_forward(
          tmol_disulfide_cpu_serial_executor(),
          f32_inputs,
          TMOL_DISULFIDE_CPU_INFERENCE,
          f32_outputs)
      == TMOL_DISULFIDE_CPU_SUCCESS);
  CHECK(std::isfinite(f32_energy));
  tmol_disulfide_cpu_f32_inputs empty_f32{};
  CHECK(tmol_disulfide_cpu_f32_recompute_vjp_workspace(0).bytes == 0);
  tmol_disulfide_cpu_f32_forward_outputs empty_f32_outputs{
      &f32_energy, 1, nullptr, 0, 0};
  CHECK(
      tmol_disulfide_cpu_f32_forward(
          tmol_disulfide_cpu_serial_executor(),
          empty_f32,
          TMOL_DISULFIDE_CPU_INFERENCE,
          empty_f32_outputs)
      == TMOL_DISULFIDE_CPU_SUCCESS);
  CHECK(
      tmol_disulfide_cpu_f32_vjp(
          tmol_disulfide_cpu_serial_executor(),
          TMOL_DISULFIDE_CPU_MAY_OVERLAP,
          nullptr,
          0,
          1,
          nullptr,
          6,
          1,
          nullptr,
          1,
          nullptr,
          0,
          1)
      == TMOL_DISULFIDE_CPU_SUCCESS);
  CHECK(
      tmol_disulfide_cpu_f32_recompute_vjp(
          tmol_disulfide_cpu_serial_executor(),
          TMOL_DISULFIDE_CPU_MAY_OVERLAP,
          empty_f32,
          nullptr,
          1,
          nullptr,
          1,
          nullptr,
          0)
      == TMOL_DISULFIDE_CPU_SUCCESS);

  pairs[0][0] = 6;
  CHECK(
      tmol_disulfide_cpu_f64_forward(
          executor, inputs, TMOL_DISULFIDE_CPU_INFERENCE, inference_outputs)
      == TMOL_DISULFIDE_CPU_INVALID_INDEX);
  return 0;
}
