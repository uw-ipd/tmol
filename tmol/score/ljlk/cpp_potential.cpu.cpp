#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <torch/torch.h>
#include "lj.hh"

namespace tmol {
namespace score {
namespace lj {

template <typename Real, typename AtomType, typename Int>
at::Tensor lj_intra(at::Tensor coords_t, at::Tensor types_t, LJ_PARAM_ARGS) {
  auto out_t = coords_t.type().zeros({coords_t.size(0), coords_t.size(0)});

  auto coords = tmol::reinterpret_tensor<Eigen::Vector3f, Real, 2>(coords_t);
  auto types = types_t.accessor<AtomType, 1>();

  auto out = out_t.accessor<Real, 2>();

  LJ_PARAM_UNPACK

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

      out[i][j] =
          lj(dist,
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
  m.def("lj", &lj<float, uint8_t>, "LJ potential.", "dist"_a, LJ_PARAM_PYARGS);

  m.def(
      "d_lj_d_dist",
      &d_lj_d_dist<float, uint8_t>,
      "LJ potential derivative.",
      "dist"_a,
      LJ_PARAM_PYARGS);

  m.def(
      "lj_intra",
      &lj_intra<float, int64_t, uint8_t>,
      "LJ intra-coordinate score.",
      "coords"_a,
      "types"_a,
      LJ_PARAM_PYARGS);
}

}  // namespace lj
}  // namespace score
}  // namespace tmol
