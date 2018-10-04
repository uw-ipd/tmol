#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <torch/torch.h>
#include "lj.hh"

namespace tmol {
namespace score {
namespace ljlk {
at::Tensor lj_intra(
    at::Tensor coords_t,
    at::Tensor types_t,
    at::Tensor bonded_path_length_t,
    at::Tensor lj_sigma_t,
    at::Tensor lj_switch_slope_t,
    at::Tensor lj_switch_intercept_t,
    at::Tensor lj_coeff_sigma12_t,
    at::Tensor lj_coeff_sigma6_t,
    at::Tensor lj_spline_y0_t,
    at::Tensor lj_spline_dy0_t,
    at::Tensor lj_switch_dis2sigma_t,
    at::Tensor spline_start_t,
    at::Tensor max_dis_t) {
  auto out_t = coords_t.type().zeros({coords_t.size(0), coords_t.size(0)});
  auto out = out_t.accessor<float, 2>();

  auto coords = tmol::reinterpret_tensor<Eigen::Vector3f, float, 2>(coords_t);
  auto types = types_t.accessor<int64_t, 1>();
  auto bonded_path_length = bonded_path_length_t.accessor<uint8_t, 2>();

  auto lj_sigma = lj_sigma_t.accessor<float, 2>();
  auto lj_switch_slope = lj_switch_slope_t.accessor<float, 2>();
  auto lj_switch_intercept = lj_switch_intercept_t.accessor<float, 2>();
  auto lj_coeff_sigma12 = lj_coeff_sigma12_t.accessor<float, 2>();
  auto lj_coeff_sigma6 = lj_coeff_sigma6_t.accessor<float, 2>();
  auto lj_spline_y0 = lj_spline_y0_t.accessor<float, 2>();
  auto lj_spline_dy0 = lj_spline_dy0_t.accessor<float, 2>();

  // Reshape globals into 1d, accessor cast of 0-d tensors segfaults.
  auto lj_switch_dis2sigma =
      lj_switch_dis2sigma_t.reshape({1}).accessor<float, 1>();
  auto spline_start = spline_start_t.reshape({1}).accessor<float, 1>();
  auto max_dis = max_dis_t.reshape({1}).accessor<float, 1>();

  for (int i = 0; i < coords.size(0); ++i) {
    auto a = coords[i][0];
    auto at = types[i];

    if (at == -1) {
      continue;
    }

    for (int j = i; j < coords.size(0); ++j) {
      auto b = coords[j][0];
      auto bt = types[j];
      if (bt == -1) {
        continue;
      }

      Eigen::Vector3f delta = a - b;
      auto dist = std::sqrt(delta.dot(delta));

      out[i][j] = lj_potential(
          dist,
          bonded_path_length[i][j],
          lj_sigma[at][bt],
          lj_switch_slope[at][bt],
          lj_switch_intercept[at][bt],
          lj_coeff_sigma12[at][bt],
          lj_coeff_sigma6[at][bt],
          lj_spline_y0[at][bt],
          lj_spline_dy0[at][bt],
          lj_switch_dis2sigma[0],
          spline_start[0],
          max_dis[0]);
    }
  }

  return out_t;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace pybind11::literals;
  m.def(
      "lj_potential",
      &lj_potential<float, int64_t>,
      "LJ potential.",
      "dist"_a,
      "bonded_path_length"_a,
      "lj_sigma"_a,
      "lj_switch_slope"_a,
      "lj_switch_intercept"_a,
      "lj_coeff_sigma12"_a,
      "lj_coeff_sigma6"_a,
      "lj_spline_y0"_a,
      "lj_spline_dy0"_a,
      "lj_switch_dis2sigma"_a,
      "spline_start"_a,
      "max_dis"_a);

  m.def(
      "lj_intra",
      &lj_intra,
      "LJ intra-coordinate score.",
      "coords"_a,
      "types"_a,
      "bonded_path_length"_a,
      "lj_sigma"_a,
      "lj_switch_slope"_a,
      "lj_switch_intercept"_a,
      "lj_coeff_sigma12"_a,
      "lj_coeff_sigma6"_a,
      "lj_spline_y0"_a,
      "lj_spline_dy0"_a,
      "lj_switch_dis2sigma"_a,
      "spline_start"_a,
      "max_dis"_a);
}

}  // namespace ljlk
}  // namespace score
}  // namespace tmol
