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
  int const n_atoms = ligand_coords.size(1);
  int const n_polar = ligand_polar_atoms.size(0);
  auto waters_t = TPack<Real3, 3, D>::empty({n_poses, n_polar, 4});
  auto waters = waters_t.view;

  LAUNCH_BOX_32;
  auto build_pose = [=] TMOL_DEVICE_FUNC(int pose) {
    Real const nan = std::numeric_limits<Real>::quiet_NaN();
    for (int polar = 0; polar < n_polar; ++polar) {
      for (int water = 0; water < 4; ++water) {
        waters[pose][polar][water] = Real3::Constant(nan);
        Int const kind = ligand_water_kinds[polar][water];
        if (kind == 0) continue;
        Int const atom0 = ligand_water_atoms[polar][water][0];
        Int const atom1 = ligand_water_atoms[polar][water][1];
        if ((kind != 1 && kind != 2) || atom0 < 0 || atom0 >= n_atoms
            || atom1 < 0 || atom1 >= n_atoms) {
          continue;
        }
        Real3 const xyz0 = ligand_coords[pose][atom0];
        Real3 xyz1 = ligand_coords[pose][atom1];
        Real3 const parameters = ligand_water_params[polar][water];
        if (!grid_lkb_finite(xyz0) || !grid_lkb_finite(xyz1)
            || !grid_lkb_finite(parameters)) {
          continue;
        }
        Real3 result;
        if (kind == 1) {
          Real3 const direction = xyz1 - xyz0;
          if (direction.squaredNorm() > Real(0)) {
            result = build_don_water<Real>::V(xyz0, xyz1, water_distance);
          } else {
            result = xyz0;
          }
        } else {
          Int const atom2 = ligand_water_atoms[polar][water][2];
          if (atom2 < 0 || atom2 >= n_atoms) continue;
          Real3 const xyz2 = ligand_coords[pose][atom2];
          if (!grid_lkb_finite(xyz2)) continue;
          if (ligand_water_is_ring[polar][water]) {
            xyz1 = (xyz1 + xyz2) / Real(2);
          }
          result = common::build_coordinate<Real>::V(
              xyz0,
              xyz1,
              xyz2,
              water_distance,
              parameters[1],
              parameters[2]);
          if (!grid_lkb_finite(result)) result = xyz0;
        }
        waters[pose][polar][water] = result;
      }
    }
  };
  DeviceOps<D>::template forall<launch_t>(mgr, n_poses, build_pose);
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
  auto propagate_pose = [=] TMOL_DEVICE_FUNC(int pose) {
    for (int polar = 0; polar < n_polar; ++polar) {
      for (int water = 0; water < 4; ++water) {
        Int const kind = ligand_water_kinds[polar][water];
        if (kind == 0) continue;
        Int const atom0 = ligand_water_atoms[polar][water][0];
        Int const atom1 = ligand_water_atoms[polar][water][1];
        if ((kind != 1 && kind != 2) || atom0 < 0 || atom0 >= n_atoms
            || atom1 < 0 || atom1 >= n_atoms) {
          continue;
        }
        Real3 const xyz0 = ligand_coords[pose][atom0];
        Real3 xyz1 = ligand_coords[pose][atom1];
        Real3 const parameters = ligand_water_params[polar][water];
        Real3 const dE_dW = water_gradients[pose][polar][water];
        if (kind == 1) {
          auto const deriv =
              build_don_water<Real>::dV(xyz0, xyz1, water_distance);
          gradients[pose][atom0] += deriv.dD * dE_dW;
          gradients[pose][atom1] += deriv.dH * dE_dW;
          continue;
        }
        Int const atom2 = ligand_water_atoms[polar][water][2];
        if (atom2 < 0 || atom2 >= n_atoms) continue;
        Real3 const xyz2 = ligand_coords[pose][atom2];
        bool const ring = ligand_water_is_ring[polar][water] != 0;
        if (ring) xyz1 = (xyz1 + xyz2) / Real(2);
        auto const deriv = common::build_coordinate<Real>::dV(
            xyz0,
            xyz1,
            xyz2,
            water_distance,
            parameters[1],
            parameters[2]);
        gradients[pose][atom0] += deriv.dp * dE_dW;
        if (ring) {
          gradients[pose][atom1] += Real(0.5) * deriv.dgp * dE_dW;
          gradients[pose][atom2] += Real(0.5) * deriv.dgp * dE_dW;
        } else {
          gradients[pose][atom1] += deriv.dgp * dE_dW;
        }
        gradients[pose][atom2] += deriv.dggp * dE_dW;
      }
    }
  };
  DeviceOps<D>::template forall<launch_t>(mgr, n_poses, propagate_pose);
  return gradients_t;
}

}  // namespace potentials
}  // namespace gridscore
}  // namespace score
}  // namespace tmol
