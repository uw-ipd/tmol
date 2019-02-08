#include <pybind11/eigen.h>
#include <tmol/utility/tensor/pybind.h>
#include <torch/torch.h>

#include <tmol/score/ljlk/potentials/lk_ball.hh>

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

template <typename Real>
void bind_build_waters(pybind11::module& m) {
  using namespace pybind11::literals;
  using namespace pybind11;
  typedef Eigen::Matrix<Real, 3, 1> Real3;

  m.def("build_don_water_V", &build_don_water<Real>::V, "D"_a, "H"_a, "dist"_a);
  m.def(
      "build_don_water_dV",
      [](Real3 D, Real3 H, Real dist) {
        return build_don_water<Real>::dV(D, H, dist).astuple();
      },
      "D"_a,
      "H"_a,
      "dist"_a);

  m.def(
      "build_acc_water_V",
      &build_acc_water<Real>::V,
      "A"_a,
      "B"_a,
      "B0"_a,
      "dist"_a,
      "angle"_a,
      "torsion"_a);

  m.def(
      "build_acc_water_dV",
      [](Real3 A, Real3 B, Real3 B0, Real dist, Real angle, Real torsion) {
        return build_acc_water<Real>::dV(A, B, B0, dist, angle, torsion)
            .astuple();
      },
      "A"_a,
      "B"_a,
      "B0"_a,
      "dist"_a,
      "angle"_a,
      "torsion"_a);
}

template <typename Real, int MAX_WATER>
void bind_potentials(pybind11::module& m) {
  using namespace pybind11::literals;
  using namespace pybind11;

  typedef Eigen::Matrix<Real, 3, 1> Real3;
  typedef Eigen::Matrix<Real, MAX_WATER, 3> WatersMat;

  m.def(
      "lk_fraction_V",
      &lk_fraction<Real, MAX_WATER>::V,
      "WI"_a,
      "J"_a,
      "lk_radius_j"_a);

  m.def(
      "lk_fraction_dV",
      [](WatersMat WI, Real3 J, Real lj_radius_j) {
        return lk_fraction<Real, MAX_WATER>::dV(WI, J, lj_radius_j).astuple();
      },
      "WI"_a,
      "J"_a,
      "lk_radius_j"_a);

  m.def(
      "lk_bridge_fraction_V",
      &lk_bridge_fraction<Real, MAX_WATER>::V,
      "coord_i"_a,
      "coord_j"_a,
      "waters_i"_a,
      "waters_j"_a,
      "lkb_water_dist"_a);

  m.def(
      "lk_bridge_fraction_dV",
      [](Real3 coord_i,
         Real3 coord_j,
         WatersMat waters_i,
         WatersMat waters_j,
         Real lkb_water_dist) {
        return lk_bridge_fraction<Real, MAX_WATER>::dV(
                   coord_i, coord_j, waters_i, waters_j, lkb_water_dist)
            .astuple();
      },
      "coord_i"_a,
      "coord_j"_a,
      "waters_i"_a,
      "waters_j"_a,
      "lkb_water_dist"_a);

  m.def(
      "lk_ball_score_V",
      [](Eigen::Matrix<Real, 3, 1> coord_i,
         Eigen::Matrix<Real, 3, 1> coord_j,
         Eigen::Matrix<Real, MAX_WATER, 3> waters_i,
         Eigen::Matrix<Real, MAX_WATER, 3> waters_j,
         Real bonded_path_length,
         Real lkb_water_dist,
         LKTypeParams_args(i_),
         LKTypeParams_args(j_),
         LJGlobalParams_args()) {
        return lk_ball_score<Real, MAX_WATER>::V(
                   coord_i,
                   coord_j,
                   waters_i,
                   waters_j,
                   bonded_path_length,
                   lkb_water_dist,
                   LKTypeParams_struct(i_),
                   LKTypeParams_struct(j_),
                   LJGlobalParams_struct())
            .astuple();
      },
      "coord_i"_a,
      "coord_j"_a,
      "waters_i"_a,
      "waters_j"_a,
      "bonded_path_length"_a,
      "lkb_water_dist"_a,
      LKTypeParams_pyargs(i_),
      LKTypeParams_pyargs(j_),
      LJGlobalParams_pyargs());

  m.def(
      "lk_ball_score_dV",
      [](Eigen::Matrix<Real, 3, 1> coord_i,
         Eigen::Matrix<Real, 3, 1> coord_j,
         Eigen::Matrix<Real, MAX_WATER, 3> waters_i,
         Eigen::Matrix<Real, MAX_WATER, 3> waters_j,
         Real bonded_path_length,
         Real lkb_water_dist,
         LKTypeParams_args(i_),
         LKTypeParams_args(j_),
         LJGlobalParams_args()) {
        return lk_ball_score<Real, MAX_WATER>::dV(
                   coord_i,
                   coord_j,
                   waters_i,
                   waters_j,
                   bonded_path_length,
                   lkb_water_dist,
                   LKTypeParams_struct(i_),
                   LKTypeParams_struct(j_),
                   LJGlobalParams_struct())
            .astuple();
      },
      "coord_i"_a,
      "coord_j"_a,
      "waters_i"_a,
      "waters_j"_a,
      "bonded_path_length"_a,
      "lkb_water_dist"_a,
      LKTypeParams_pyargs(i_),
      LKTypeParams_pyargs(j_),
      LJGlobalParams_pyargs());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  bind_build_waters<double>(m);
  bind_potentials<double, 2>(m);
  bind_potentials<double, 3>(m);
}

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
