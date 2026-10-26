#pragma once

#include <cmath>
#include <limits>

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/gen_coord.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/gridscore/potentials/grid_lk_ball.hh>
#include <tmol/score/lk_ball/potentials/water.hh>

namespace tmol {
namespace score {
namespace gridscore {
namespace potentials {

template <typename Real>
EIGEN_DEVICE_FUNC bool grid_lkb_finite(GridLKBVec<Real, 3> const& xyz) {
  return std::isfinite(xyz[0]) && std::isfinite(xyz[1])
         && std::isfinite(xyz[2]);
}

template <
    template <tmol::Device> class DeviceOps,
    tmol::Device D,
    typename Real,
    typename Int>
auto GridLKBallDispatch<DeviceOps, D, Real, Int>::build_waters(
    ContextManager& mgr,
    TView<GridLKBVec<Real, 3>, 2, D> ligand_coords,
    TView<Int, 1, D> ligand_polar_atoms,
    TView<Int, 2, D> ligand_water_kinds,
    TView<Int, 3, D> ligand_water_atoms,
    TView<GridLKBVec<Real, 3>, 2, D> ligand_water_params,
    TView<Int, 2, D> ligand_water_is_ring,
    Real water_distance) -> TPack<GridLKBVec<Real, 3>, 3, D> {
  using lk_ball::potentials::build_don_water;
  using Real3 = GridLKBVec<Real, 3>;
  int const n_poses = ligand_coords.size(0);
  int const n_polar = ligand_polar_atoms.size(0);
  auto waters_t = TPack<Real3, 3, D>::empty({n_poses, n_polar, 4});
  auto waters = waters_t.view;

  // One work item per (pose, polar atom, water slot).
  LAUNCH_BOX_32;
  auto build_water = [=] TMOL_DEVICE_FUNC(int work) {
    int const pose = work / (n_polar * 4);
    int const polar = (work / 4) % n_polar;
    int const water = work % 4;
    Int const kind = ligand_water_kinds[polar][water];
    if (kind == 0) {
      waters[pose][polar][water] =
          Real3::Constant(std::numeric_limits<Real>::quiet_NaN());
      return;
    }
    Real3 const xyz0 = ligand_coords[pose][ligand_water_atoms[polar][water][0]];
    Real3 xyz1 = ligand_coords[pose][ligand_water_atoms[polar][water][1]];
    Real3 result;
    if (kind == 1) {
      // A hydrogen coincident with its donor has no direction; fall back to it.
      result = (xyz1 - xyz0).squaredNorm() > Real(0)
                   ? build_don_water<Real>::V(xyz0, xyz1, water_distance)
                   : xyz0;
    } else {
      Real3 const xyz2 =
          ligand_coords[pose][ligand_water_atoms[polar][water][2]];
      Real3 const parameters = ligand_water_params[polar][water];
      if (ligand_water_is_ring[polar][water]) xyz1 = (xyz1 + xyz2) / Real(2);
      result = common::build_coordinate<Real>::V(
          xyz0, xyz1, xyz2, water_distance, parameters[1], parameters[2]);
      if (!grid_lkb_finite(result)) result = xyz0;
    }
    waters[pose][polar][water] = result;
  };
  DeviceOps<D>::template forall<launch_t>(
      mgr, n_poses * n_polar * 4, build_water);
  return waters_t;
}

template <
    template <tmol::Device> class DeviceOps,
    tmol::Device D,
    typename Real,
    typename Int>
auto GridLKBallDispatch<DeviceOps, D, Real, Int>::propagate_water_gradients(
    ContextManager& mgr,
    TView<GridLKBVec<Real, 3>, 2, D> ligand_coords,
    TView<Int, 1, D> ligand_polar_atoms,
    TView<Int, 2, D> ligand_water_kinds,
    TView<Int, 3, D> ligand_water_atoms,
    TView<GridLKBVec<Real, 3>, 2, D> ligand_water_params,
    TView<Int, 2, D> ligand_water_is_ring,
    TView<GridLKBVec<Real, 3>, 3, D> water_gradients,
    Real water_distance) -> TPack<GridLKBVec<Real, 3>, 2, D> {
  using lk_ball::potentials::build_don_water;
  using Real3 = GridLKBVec<Real, 3>;
  int const n_poses = ligand_coords.size(0);
  int const n_atoms = ligand_coords.size(1);
  int const n_polar = ligand_polar_atoms.size(0);
  auto gradients_t = TPack<Real3, 2, D>::zeros({n_poses, n_atoms});
  auto gradients = gradients_t.view;

  LAUNCH_BOX_32;
  auto propagate_water = [=] TMOL_DEVICE_FUNC(int work) {
    int const pose = work / (n_polar * 4);
    int const polar = (work / 4) % n_polar;
    int const water = work % 4;
    Int const kind = ligand_water_kinds[polar][water];
    if (kind == 0) return;
    auto add = [&] TMOL_DEVICE_FUNC(Int atom, Real3 const& gradient) {
      common::accumulate<D, Real3>::add(gradients[pose][atom], gradient);
    };
    Int const atom0 = ligand_water_atoms[polar][water][0];
    Int const atom1 = ligand_water_atoms[polar][water][1];
    Real3 const xyz0 = ligand_coords[pose][atom0];
    Real3 xyz1 = ligand_coords[pose][atom1];
    Real3 const dE_dW = water_gradients[pose][polar][water];
    if (kind == 1) {
      auto const deriv = build_don_water<Real>::dV(xyz0, xyz1, water_distance);
      add(atom0, deriv.dD * dE_dW);
      add(atom1, deriv.dH * dE_dW);
      return;
    }
    Int const atom2 = ligand_water_atoms[polar][water][2];
    Real3 const xyz2 = ligand_coords[pose][atom2];
    Real3 const parameters = ligand_water_params[polar][water];
    bool const ring = ligand_water_is_ring[polar][water] != 0;
    if (ring) xyz1 = (xyz1 + xyz2) / Real(2);
    auto const deriv = common::build_coordinate<Real>::dV(
        xyz0, xyz1, xyz2, water_distance, parameters[1], parameters[2]);
    add(atom0, deriv.dp * dE_dW);
    if (ring) {
      add(atom1, Real(0.5) * deriv.dgp * dE_dW);
      add(atom2, Real(0.5) * deriv.dgp * dE_dW);
    } else {
      add(atom1, deriv.dgp * dE_dW);
    }
    add(atom2, deriv.dggp * dE_dW);
  };
  DeviceOps<D>::template forall<launch_t>(
      mgr, n_poses * n_polar * 4, propagate_water);
  return gradients_t;
}

}  // namespace potentials
}  // namespace gridscore
}  // namespace score
}  // namespace tmol
