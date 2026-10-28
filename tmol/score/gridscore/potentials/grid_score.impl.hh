#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/gen_coord.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/gridscore/potentials/grid_interp_point.hh>
#include <tmol/score/gridscore/potentials/grid_score.hh>
#include <tmol/score/lk_ball/potentials/water.hh>

namespace tmol {
namespace score {
namespace gridscore {
namespace potentials {

namespace grid_score_detail {

// Position of one LK-ball water from the current ligand coordinates.
template <tmol::Device D, typename Real, typename Int>
EIGEN_DEVICE_FUNC GridVec<Real, 3> water_position(
    TView<GridVec<Real, 3>, 2, D> const& coords,
    int pose,
    Int kind,
    Int const atoms[3],
    GridVec<Real, 3> const& params,
    bool ring,
    Real water_distance) {
  using Real3 = GridVec<Real, 3>;
  Real3 const xyz0 = coords[pose][atoms[0]];
  Real3 xyz1 = coords[pose][atoms[1]];
  if (kind == 1) {
    // A hydrogen coincident with its donor has no direction: use the donor.
    return (xyz1 - xyz0).squaredNorm() > Real(0)
               ? lk_ball::potentials::build_don_water<Real>::V(
                     xyz0, xyz1, water_distance)
               : xyz0;
  }
  Real3 const xyz2 = coords[pose][atoms[2]];
  if (ring) xyz1 = (xyz1 + xyz2) / Real(2);
  Real3 const result = common::build_coordinate<Real>::V(
      xyz0, xyz1, xyz2, water_distance, params[1], params[2]);
  bool const finite = std::isfinite(result[0]) && std::isfinite(result[1])
                      && std::isfinite(result[2]);
  return finite ? result : xyz0;
}

// Adds dE/d(water position) to the gradient of the atoms that built the water.
template <tmol::Device D, typename Real, typename Int>
EIGEN_DEVICE_FUNC void propagate_water_gradient(
    TView<GridVec<Real, 3>, 2, D> const& coords,
    TView<GridVec<Real, 3>, 2, D> gradient,
    int pose,
    Int kind,
    Int const atoms[3],
    GridVec<Real, 3> const& params,
    bool ring,
    Real water_distance,
    GridVec<Real, 3> const& dE_dW) {
  using Real3 = GridVec<Real, 3>;
  Real3 const xyz0 = coords[pose][atoms[0]];
  Real3 xyz1 = coords[pose][atoms[1]];
  if (kind == 1) {
    auto const deriv = lk_ball::potentials::build_don_water<Real>::dV(
        xyz0, xyz1, water_distance);
    common::accumulate<D, Real3>::add(
        gradient[pose][atoms[0]], deriv.dD * dE_dW);
    common::accumulate<D, Real3>::add(
        gradient[pose][atoms[1]], deriv.dH * dE_dW);
    return;
  }
  Real3 const xyz2 = coords[pose][atoms[2]];
  if (ring) xyz1 = (xyz1 + xyz2) / Real(2);
  auto const deriv = common::build_coordinate<Real>::dV(
      xyz0, xyz1, xyz2, water_distance, params[1], params[2]);
  common::accumulate<D, Real3>::add(gradient[pose][atoms[0]], deriv.dp * dE_dW);
  if (ring) {
    common::accumulate<D, Real3>::add(
        gradient[pose][atoms[1]], Real(0.5) * deriv.dgp * dE_dW);
    common::accumulate<D, Real3>::add(
        gradient[pose][atoms[2]], Real(0.5) * deriv.dgp * dE_dW);
  } else {
    common::accumulate<D, Real3>::add(
        gradient[pose][atoms[1]], deriv.dgp * dE_dW);
  }
  common::accumulate<D, Real3>::add(
      gradient[pose][atoms[2]], deriv.dggp * dE_dW);
}

}  // namespace grid_score_detail

template <
    template <tmol::Device> class DeviceOps,
    tmol::Device D,
    typename Real,
    typename Int>
auto GridScoreDispatch<DeviceOps, D, Real, Int>::atoms(
    ContextManager& mgr,
    TView<GridVec<Real, 3>, 2, D> coords,
    TView<Int, 1, D> grid_atoms,
    TView<Int, 1, D> atom_types,
    TView<Real, 1, D> charges,
    TView<Real, 4, D> atr,
    TView<Real, 4, D> rep,
    TView<Real, 4, D> sol,
    TView<Real, 4, D> elec,
    TView<Real, 4, D> lk_ball,
    TView<Real, 1, D> origin,
    Real voxel_spacing,
    TView<Real, 1, D> weights,
    Real out_of_bounds_penalty)
    -> std::tuple<TPack<Real, 2, D>, TPack<GridVec<Real, 3>, 2, D>> {
  int const n_poses = coords.size(0);
  int const n_atoms = coords.size(1);
  int const n_grid_atoms = grid_atoms.size(0);
  int const n_types = atr.size(0);
  bool const with_lk_ball = lk_ball.size(0) > 0;
  int const dims[3] = {int(atr.size(1)), int(atr.size(2)), int(atr.size(3))};
  auto energy_t = TPack<Real, 2, D>::zeros({n_poses, n_grid_atoms});
  auto gradient_t = TPack<GridVec<Real, 3>, 2, D>::zeros({n_poses, n_atoms});
  auto energy = energy_t.view;
  auto gradient = gradient_t.view;

  LAUNCH_BOX_32;
  DeviceOps<D>::template forall_grouped<launch_t>(
      mgr, n_poses, n_grid_atoms, [=] TMOL_DEVICE_FUNC(int work) {
        int const pose = work / n_grid_atoms;
        int const g = work % n_grid_atoms;
        Real const scale = Real(1) / voxel_spacing;
        Int const atom = grid_atoms[g];
        Int const type = atom_types[g];
        Real position[3];
        for (int axis = 0; axis < 3; ++axis) {
          position[axis] =
              (coords[pose][atom][axis] - origin[axis]) / voxel_spacing;
        }
        auto const stencil =
            make_grid_stencil<Real>(position, dims, out_of_bounds_penalty);
        if (!stencil.valid || type < 0 || type >= n_types) {
          energy[pose][g] = std::numeric_limits<Real>::quiet_NaN();
          for (int axis = 0; axis < 3; ++axis) {
            gradient[pose][atom][axis] = std::numeric_limits<Real>::quiet_NaN();
          }
          return;
        }
        Real v_atr, v_rep, v_sol, v_elec, v_lk = 0;
        Real d_atr[3], d_rep[3], d_sol[3], d_elec[3], d_lk[3] = {0, 0, 0};
        contract_grid<D, Real>(atr, type, stencil, v_atr, d_atr);
        contract_grid<D, Real>(rep, type, stencil, v_rep, d_rep);
        contract_grid<D, Real>(sol, type, stencil, v_sol, d_sol);
        contract_grid<D, Real>(elec, 0, stencil, v_elec, d_elec);
        if (with_lk_ball) {
          contract_grid<D, Real>(lk_ball, type, stencil, v_lk, d_lk);
        }
        Real const charge = charges[g];
        // fa_rep is stored transformed (rep^-4 - 1): chain rule through it.
        Real const rep_slope = Real(-4) * std::pow(v_rep, Real(-5));
        energy[pose][g] = weights[0] * v_atr
                          + weights[1] * (std::pow(v_rep, Real(-4)) - Real(1))
                          + weights[2] * v_sol + weights[3] * charge * v_elec
                          + v_lk + stencil.penalty;
        for (int axis = 0; axis < 3; ++axis) {
          gradient[pose][atom][axis] =
              scale
              * (weights[0] * d_atr[axis] + weights[1] * rep_slope * d_rep[axis]
                 + weights[2] * d_sol[axis] + weights[3] * charge * d_elec[axis]
                 + d_lk[axis] + stencil.penalty_gradient[axis]);
        }
      });
  return {energy_t, gradient_t};
}

template <
    template <tmol::Device> class DeviceOps,
    tmol::Device D,
    typename Real,
    typename Int>
void GridScoreDispatch<DeviceOps, D, Real, Int>::waters(
    ContextManager& mgr,
    TView<GridVec<Real, 3>, 2, D> coords,
    TView<Int, 1, D> polar_atoms,
    TView<Int, 2, D> water_kinds,
    TView<Int, 3, D> water_atoms,
    TView<GridVec<Real, 3>, 2, D> water_params,
    TView<Int, 2, D> water_is_ring,
    TView<Int, 1, D> polar_types,
    TView<Real, 4, D> water_grid,
    TView<Real, 4, D> bridge_grid,
    TView<Real, 1, D> origin,
    Real voxel_spacing,
    Real water_distance,
    Real fade,
    bool bridge_enabled,
    TView<Real, 1, D> energy,
    TView<GridVec<Real, 3>, 2, D> gradient) {
  using Real3 = GridVec<Real, 3>;
  using namespace grid_score_detail;
  int const n_poses = coords.size(0);
  int const n_polar = polar_atoms.size(0);
  int const dims[3] = {
      int(water_grid.size(1)),
      int(water_grid.size(2)),
      int(water_grid.size(3))};

  LAUNCH_BOX_32;
  // Groups are poses: items of a pose run serially on CPU (they share atoms in
  // the gradient), poses run in parallel.
  DeviceOps<D>::template forall_grouped<launch_t>(
      mgr, n_poses, n_polar, [=] TMOL_DEVICE_FUNC(int work) {
        int const pose = work / n_polar;
        int const polar = work % n_polar;
        Real const scale = Real(1) / voxel_spacing;
        {
          bool active[4];
          bool any_water = false;
          Real value_water[4], value_bridge[4];
          Real slope_water[4][3], slope_bridge[4][3];
          for (int slot = 0; slot < 4; ++slot) {
            active[slot] = water_kinds[polar][slot] != 0;
            any_water = any_water || active[slot];
            if (!active[slot]) continue;
            Int atoms[3] = {
                water_atoms[polar][slot][0],
                water_atoms[polar][slot][1],
                water_atoms[polar][slot][2]};
            Real3 const water = water_position<D, Real, Int>(
                coords,
                pose,
                water_kinds[polar][slot],
                atoms,
                water_params[polar][slot],
                water_is_ring[polar][slot] != 0,
                water_distance);
            Real position[3];
            for (int axis = 0; axis < 3; ++axis) {
              position[axis] = (water[axis] - origin[axis]) / voxel_spacing;
            }
            auto const stencil =
                make_grid_stencil<Real>(position, dims, Real(0));
            if (!stencil.valid) {
              Real const nan = std::numeric_limits<Real>::quiet_NaN();
              value_water[slot] = value_bridge[slot] = nan;
              for (int axis = 0; axis < 3; ++axis) {
                slope_water[slot][axis] = slope_bridge[slot][axis] = nan;
              }
              continue;
            }
            contract_grid<D, Real>(
                water_grid,
                polar_types[polar],
                stencil,
                value_water[slot],
                slope_water[slot]);
            contract_grid<D, Real>(
                bridge_grid,
                polar_types[polar],
                stencil,
                value_bridge[slot],
                slope_bridge[slot]);
          }
          if (!any_water) return;

          // Softmax over the waters of the polar atom (softmin for bridges).
          Real max_water = -std::numeric_limits<Real>::infinity();
          Real max_bridge = -std::numeric_limits<Real>::infinity();
          for (int slot = 0; slot < 4; ++slot) {
            if (!active[slot]) continue;
            max_water = std::max(max_water, value_water[slot] / fade);
            max_bridge = std::max(max_bridge, -value_bridge[slot] / fade);
          }
          Real sum_water = 0, sum_bridge = 0;
          for (int slot = 0; slot < 4; ++slot) {
            if (!active[slot]) continue;
            sum_water += std::exp(value_water[slot] / fade - max_water);
            sum_bridge += std::exp(-value_bridge[slot] / fade - max_bridge);
          }
          Real polar_energy = fade * (max_water + std::log(sum_water));
          if (bridge_enabled) {
            polar_energy -= fade * (max_bridge + std::log(sum_bridge));
          }
          common::accumulate<D, Real>::add(energy[pose], polar_energy);

          for (int slot = 0; slot < 4; ++slot) {
            if (!active[slot]) continue;
            Real const weight_water =
                std::exp(value_water[slot] / fade - max_water) / sum_water;
            Real const weight_bridge =
                bridge_enabled
                    ? std::exp(-value_bridge[slot] / fade - max_bridge)
                          / sum_bridge
                    : Real(0);
            Real3 dE_dW;
            for (int axis = 0; axis < 3; ++axis) {
              dE_dW[axis] = (weight_water * slope_water[slot][axis]
                             + weight_bridge * slope_bridge[slot][axis])
                            * scale;
            }
            Int atoms[3] = {
                water_atoms[polar][slot][0],
                water_atoms[polar][slot][1],
                water_atoms[polar][slot][2]};
            propagate_water_gradient<D, Real, Int>(
                coords,
                gradient,
                pose,
                water_kinds[polar][slot],
                atoms,
                water_params[polar][slot],
                water_is_ring[polar][slot] != 0,
                water_distance,
                dE_dW);
          }
        }
      });
}

}  // namespace potentials
}  // namespace gridscore
}  // namespace score
}  // namespace tmol
