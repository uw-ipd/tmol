#pragma once

#include <cmath>
#include <limits>

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/gen_coord.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/gridscore/potentials/live_lk_ball.hh>
#include <tmol/score/lk_ball/potentials/lk_ball.hh>
#include <tmol/score/lk_ball/potentials/water.hh>

namespace tmol {
namespace score {
namespace gridscore {
namespace potentials {

template <
    template <tmol::Device> class DeviceOps,
    tmol::Device D,
    typename Real,
    typename Int>
auto LiveLKBallDispatch<DeviceOps, D, Real, Int>::forward(
    ContextManager& mgr,
    TView<LKBVec<Real, 3>, 2, D> ligand_coords,
    TView<lk_ball::potentials::LKBallTypeParams<Real>, 1, D>
        ligand_atom_params,
    TView<Int, 1, D> ligand_heavy_atoms,
    TView<Int, 1, D> ligand_polar_atoms,
    TView<Int, 1, D> ligand_atom_to_polar,
    TView<Int, 2, D> ligand_water_kinds,
    TView<Int, 3, D> ligand_water_atoms,
    TView<LKBVec<Real, 3>, 2, D> ligand_water_params,
    TView<Int, 2, D> ligand_water_is_ring,
    TView<LKBVec<Real, 3>, 1, D> receptor_coords,
    TView<lk_ball::potentials::LKBallTypeParams<Real>, 1, D> receptor_params,
    TView<LKBVec<Real, 3>, 2, D> receptor_waters,
    TView<Int, 1, D> receptor_polar_atoms,
    TView<Real, 1, D> cell_origin,
    TView<Int, 1, D> cell_dims,
    TView<Int, 1, D> receptor_cell_offsets,
    TView<Int, 1, D> receptor_cell_indices,
    TView<Int, 1, D> receptor_polar_cell_offsets,
    TView<Int, 1, D> receptor_polar_cell_indices,
    TView<lk_ball::potentials::LKBallGlobalParams<Real>, 1, D> global_params,
    bool use_cell_list)
    -> std::tuple<TPack<Real, 2, D>, TPack<LKBVec<Real, 3>, 3, D>> {
  using lk_ball::potentials::LKBallTypeParams;
  using lk_ball::potentials::build_don_water;
  using lk_ball::potentials::lk_ball_score;
  using Waters = Eigen::Matrix<Real, 4, 3>;
  using Real3 = LKBVec<Real, 3>;
  using common::accumulate;

  int const n_poses = ligand_coords.size(0);
  int const n_atoms = ligand_coords.size(1);
  int const n_ligand_heavy = ligand_heavy_atoms.size(0);
  int const n_ligand_polar = ligand_polar_atoms.size(0);
  auto energies_t = TPack<Real, 2, D>::zeros({n_poses, 4});
  auto gradients_t =
      TPack<Real3, 3, D>::zeros({n_poses, 4, n_atoms});
  auto energies = energies_t.view;
  auto gradients = gradients_t.view;

  // One work item per (pose, ligand site). A site is either a polar ligand atom
  // (desolvated side of the pair, scored against all nearby receptor atoms) or
  // a heavy ligand atom (occluding side, scored against nearby receptor polar
  // atoms). Items of one pose update shared outputs through accumulate<>.
  // Each site is further split in three x-slabs of the 3x3x3 cell neighborhood.
  int const n_sites = n_ligand_polar + n_ligand_heavy;
  LAUNCH_BOX_32;
  auto evaluate_site = [=] TMOL_DEVICE_FUNC(int work) {
    int const pose = work / (n_sites * 3);
    int const site = (work / 3) % n_sites;
    int const x_slab = work % 3 - 1;  // this item scans receptor cells at cx + x_slab
    bool const is_polar_site = site < n_ligand_polar;
    Int const atom = is_polar_site
                         ? ligand_polar_atoms[site]
                         : ligand_heavy_atoms[site - n_ligand_polar];
    Int const polar_site = is_polar_site ? Int(site) : ligand_atom_to_polar[atom];
    Real const cutoff = global_params[0].distance_threshold;
    auto const globals = global_params[0];
    Real3 const L = ligand_coords[pose][atom];

    auto add_gradient = [&] TMOL_DEVICE_FUNC(
                            int channel, Int target, Real3 const& gradient) {
      accumulate<D, Real3>::add(gradients[pose][channel][target], gradient);
    };
    // Ligand waters (kind 1: donor, kind 2: acceptor) of this atom's polar site.
    Waters WL = Waters::Constant(std::numeric_limits<Real>::quiet_NaN());
    if (polar_site >= 0) {
      for (int water = 0; water < 4; ++water) {
        Int const kind = ligand_water_kinds[polar_site][water];
        if (kind == 0) continue;
        Real3 const xyz0 = ligand_coords[pose][ligand_water_atoms[polar_site][water][0]];
        Real3 xyz1 = ligand_coords[pose][ligand_water_atoms[polar_site][water][1]];
        Real3 const parameters = ligand_water_params[polar_site][water];
        if (kind == 1) {
          WL.row(water) = build_don_water<Real>::V(xyz0, xyz1, parameters[0]);
          continue;
        }
        Real3 const xyz2 = ligand_coords[pose][ligand_water_atoms[polar_site][water][2]];
        if (ligand_water_is_ring[polar_site][water]) xyz1 = (xyz1 + xyz2) / Real(2);
        WL.row(water) = common::build_coordinate<Real>::V(
            xyz0, xyz1, xyz2, parameters[0], parameters[1], parameters[2]);
      }
    }
    auto add_water_gradient = [&] TMOL_DEVICE_FUNC(
                                  int water, int channel, Real3 const& dE_dW) {
      Int const kind = ligand_water_kinds[polar_site][water];
      if (kind == 0) return;
      Int const atom0 = ligand_water_atoms[polar_site][water][0];
      Int const atom1 = ligand_water_atoms[polar_site][water][1];
      Real3 const xyz0 = ligand_coords[pose][atom0];
      Real3 xyz1 = ligand_coords[pose][atom1];
      Real3 const parameters = ligand_water_params[polar_site][water];
      if (kind == 1) {
        auto const dwater = build_don_water<Real>::dV(xyz0, xyz1, parameters[0]);
        add_gradient(channel, atom0, dwater.dD * dE_dW);
        add_gradient(channel, atom1, dwater.dH * dE_dW);
        return;
      }
      Int const atom2 = ligand_water_atoms[polar_site][water][2];
      Real3 const xyz2 = ligand_coords[pose][atom2];
      bool const ring = ligand_water_is_ring[polar_site][water] != 0;
      if (ring) xyz1 = (xyz1 + xyz2) / Real(2);
      auto const dwater = common::build_coordinate<Real>::dV(
          xyz0, xyz1, xyz2, parameters[0], parameters[1], parameters[2]);
      add_gradient(channel, atom0, dwater.dp * dE_dW);
      if (ring) {
        add_gradient(channel, atom1, Real(0.5) * dwater.dgp * dE_dW);
        add_gradient(channel, atom2, Real(0.5) * dwater.dgp * dE_dW);
      } else {
        add_gradient(channel, atom1, dwater.dgp * dE_dW);
      }
      add_gradient(channel, atom2, dwater.dggp * dE_dW);
    };

    // Pair contributions are summed locally; the chain rule through the waters
    // is linear, so one propagation and one atomic flush per work item suffice.
    Eigen::Matrix<Real, 4, 1> energy = Eigen::Matrix<Real, 4, 1>::Zero();
    lk_ball::potentials::lk_ball_dV_dReal3<Real> d_direct =
        lk_ball::potentials::lk_ball_dV_dReal3<Real>::Zero();
    lk_ball::potentials::lk_ball_dV_dWater<Real, 4> d_water =
        lk_ball::potentials::lk_ball_dV_dWater<Real, 4>::Zero();

    auto score_pair = [&] TMOL_DEVICE_FUNC(Int receptor_atom) {
      Real3 const R = receptor_coords[receptor_atom];
      if ((L - R).norm() >= cutoff) return;
      Waters WR;
      for (int water = 0; water < 4; ++water) {
        WR.row(water) = receptor_waters[receptor_atom][water];
      }
      // lk_ball_score is asymmetric: I is the desolvated atom, J the occluder.
      auto const& lig = ligand_atom_params[atom];
      auto const& rec = receptor_params[receptor_atom];
      auto const value =
          is_polar_site
              ? lk_ball_score<Real, 4>::V(L, R, WL, WR, Real(5), lig, rec, globals)
              : lk_ball_score<Real, 4>::V(R, L, WR, WL, Real(5), rec, lig, globals);
      auto const deriv =
          is_polar_site
              ? lk_ball_score<Real, 4>::dV(L, R, WL, WR, Real(5), lig, rec, globals)
              : lk_ball_score<Real, 4>::dV(R, L, WR, WL, Real(5), rec, lig, globals);
      energy[0] += value.lkball_iso;
      energy[1] += value.lkball;
      energy[2] += value.lkbridge;
      energy[3] += value.lkbridge_uncpl;
      auto const& direct = is_polar_site ? deriv.dI : deriv.dJ;
      d_direct.d_lkball_iso += direct.d_lkball_iso;
      d_direct.d_lkball += direct.d_lkball;
      d_direct.d_lkbridge += direct.d_lkbridge;
      d_direct.d_lkbridge_uncpl += direct.d_lkbridge_uncpl;
      if (polar_site >= 0) {
        auto const& dW = is_polar_site ? deriv.dWI : deriv.dWJ;
        d_water.d_lkball_iso += dW.d_lkball_iso;
        d_water.d_lkball += dW.d_lkball;
        d_water.d_lkbridge += dW.d_lkbridge;
        d_water.d_lkbridge_uncpl += dW.d_lkbridge_uncpl;
      }
    };

    TView<Int, 1, D> const offsets =
        is_polar_site ? receptor_cell_offsets : receptor_polar_cell_offsets;
    TView<Int, 1, D> const indices =
        is_polar_site ? receptor_cell_indices : receptor_polar_cell_indices;
    auto flush = [&] TMOL_DEVICE_FUNC() {
      for (int c = 0; c < 4; ++c) {
        accumulate<D, Real>::add(energies[pose][c], energy[c]);
      }
      add_gradient(0, atom, d_direct.d_lkball_iso);
      add_gradient(1, atom, d_direct.d_lkball);
      add_gradient(2, atom, d_direct.d_lkbridge);
      add_gradient(3, atom, d_direct.d_lkbridge_uncpl);
      if (polar_site < 0) return;
      for (int water = 0; water < 4; ++water) {
        add_water_gradient(water, 0, d_water.d_lkball_iso.row(water).transpose());
        add_water_gradient(water, 1, d_water.d_lkball.row(water).transpose());
        add_water_gradient(water, 2, d_water.d_lkbridge.row(water).transpose());
        add_water_gradient(
            water, 3, d_water.d_lkbridge_uncpl.row(water).transpose());
      }
    };
    if (!use_cell_list) {
      if (x_slab == -1) {
        for (Int entry = 0; entry < indices.size(0); ++entry) {
          score_pair(indices[entry]);
        }
        flush();
      }
      return;
    }
    Int cell_xyz[3];
    for (int axis = 0; axis < 3; ++axis) {
      cell_xyz[axis] =
          static_cast<Int>(std::floor((L[axis] - cell_origin[axis]) / cutoff));
    }
    {
      Int const x = cell_xyz[0] + x_slab;
      if (x < 0 || x >= cell_dims[0]) return;
      for (Int dy = -1; dy <= 1; ++dy) {
        Int const y = cell_xyz[1] + dy;
        if (y < 0 || y >= cell_dims[1]) continue;
        for (Int dz = -1; dz <= 1; ++dz) {
          Int const z = cell_xyz[2] + dz;
          if (z < 0 || z >= cell_dims[2]) continue;
          Int const cell = (x * cell_dims[1] + y) * cell_dims[2] + z;
          for (Int entry = offsets[cell]; entry < offsets[cell + 1]; ++entry) {
            score_pair(indices[entry]);
          }
        }
      }
    }
    flush();
  };
  DeviceOps<D>::template forall<launch_t>(mgr, n_poses * n_sites * 3, evaluate_site);
  return std::make_tuple(energies_t, gradients_t);
}

}  // namespace potentials
}  // namespace gridscore
}  // namespace score
}  // namespace tmol
