#pragma once

#include <cmath>
#include <limits>

#include <tmol/score/common/gen_coord.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/gridscore/potentials/live_lk_ball.hh>
#include <tmol/score/lk_ball/potentials/lk_ball.hh>
#include <tmol/score/lk_ball/potentials/water.hh>

namespace tmol {
namespace score {
namespace gridscore {
namespace potentials {

template <typename Real>
EIGEN_DEVICE_FUNC bool live_lk_ball_finite(LKBVec<Real, 3> const& xyz) {
  return std::isfinite(xyz[0]) && std::isfinite(xyz[1])
         && std::isfinite(xyz[2]);
}

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
  using lk_ball::potentials::lk_ball_dV_dReal3;
  using lk_ball::potentials::lk_ball_dV_dWater;
  using lk_ball::potentials::lk_ball_score;
  using lk_ball::potentials::lk_ball_Vt;
  using Waters = Eigen::Matrix<Real, 4, 3>;
  using Real3 = LKBVec<Real, 3>;

  int const n_poses = ligand_coords.size(0);
  int const n_atoms = ligand_coords.size(1);
  int const n_ligand_heavy = ligand_heavy_atoms.size(0);
  int const n_ligand_polar = ligand_polar_atoms.size(0);
  Int const n_receptor = receptor_coords.size(0);
  Int const n_receptor_polar = receptor_polar_atoms.size(0);
  Int const n_cells = receptor_cell_offsets.size(0) - 1;
  Int const n_receptor_entries = receptor_cell_indices.size(0);
  Int const n_receptor_polar_entries = receptor_polar_cell_indices.size(0);
  auto energies_t = TPack<Real, 2, D>::zeros({n_poses, 4});
  auto gradients_t =
      TPack<Real3, 3, D>::zeros({n_poses, 4, n_atoms});
  auto energies = energies_t.view;
  auto gradients = gradients_t.view;

  LAUNCH_BOX_32;
  auto evaluate_pose = [=] TMOL_DEVICE_FUNC(int pose) {
    auto const globals = global_params[0];
    Real const cutoff = globals.distance_threshold;
    Int const dim_y = cell_dims[1];
    Int const dim_z = cell_dims[2];
    bool valid = true;
    Real const nan = std::numeric_limits<Real>::quiet_NaN();

    auto set_nan = [&] TMOL_DEVICE_FUNC() {
      for (int channel = 0; channel < 4; ++channel) {
        energies[pose][channel] = nan;
        for (int atom = 0; atom < n_atoms; ++atom) {
          gradients[pose][channel][atom] = Real3::Constant(nan);
        }
      }
    };
    if (!std::isfinite(cutoff) || cutoff <= Real(0)
        || !std::isfinite(cell_origin[0])
        || !std::isfinite(cell_origin[1])
        || !std::isfinite(cell_origin[2]) || cell_dims[0] <= 0
        || cell_dims[1] <= 0 || cell_dims[2] <= 0 || n_cells <= 0) {
      set_nan();
      return;
    }

    auto add_gradient = [&] TMOL_DEVICE_FUNC(
                            int channel, Int atom, Real3 const& gradient) {
      if (atom < 0 || atom >= n_atoms) {
        valid = false;
        return;
      }
      gradients[pose][channel][atom] += gradient;
    };
    auto cell_coordinate = [=] TMOL_DEVICE_FUNC(Real value, int axis) -> Int {
      return static_cast<Int>(std::floor((value - cell_origin[axis]) / cutoff));
    };
    auto cell_index = [=] TMOL_DEVICE_FUNC(Int x, Int y, Int z) -> Int {
      Int const max_int = std::numeric_limits<Int>::max();
      if (dim_y > max_int / dim_z) return -1;
      Int const plane = dim_y * dim_z;
      if (y > (max_int - z) / dim_z) return -1;
      Int const row = y * dim_z + z;
      if (x > (max_int - row) / plane) return -1;
      Int const result = x * plane + row;
      return result >= 0 && result < n_cells ? result : -1;
    };
    auto safe_cell_anchor = [=] TMOL_DEVICE_FUNC(Real3 const& xyz) -> bool {
      Real const max_cell_coordinate = Real(1 << 30);
      return live_lk_ball_finite(xyz)
             && std::abs((xyz[0] - cell_origin[0]) / cutoff)
                    <= max_cell_coordinate
             && std::abs((xyz[1] - cell_origin[1]) / cutoff)
                    <= max_cell_coordinate
             && std::abs((xyz[2] - cell_origin[2]) / cutoff)
                    <= max_cell_coordinate;
    };

    auto build_ligand_waters = [&] TMOL_DEVICE_FUNC(
                                    Int polar_site, Waters& waters) -> bool {
      waters = Waters::Constant(nan);
      if (polar_site < 0 || polar_site >= n_ligand_polar) return false;
      for (int water = 0; water < 4; ++water) {
        Int const kind = ligand_water_kinds[polar_site][water];
        if (kind == 0) continue;
        if (kind != 1 && kind != 2) return false;
        Int const atom0 = ligand_water_atoms[polar_site][water][0];
        Int const atom1 = ligand_water_atoms[polar_site][water][1];
        if (atom0 < 0 || atom0 >= n_atoms || atom1 < 0 || atom1 >= n_atoms) {
          return false;
        }
        Real3 const xyz0 = ligand_coords[pose][atom0];
        Real3 xyz1 = ligand_coords[pose][atom1];
        Real3 const parameters = ligand_water_params[polar_site][water];
        if (!live_lk_ball_finite(xyz0) || !live_lk_ball_finite(xyz1)
            || !live_lk_ball_finite(parameters)) {
          return false;
        }
        if (kind == 1) {
          waters.row(water) =
              build_don_water<Real>::V(xyz0, xyz1, parameters[0]);
          continue;
        }
        Int const atom2 = ligand_water_atoms[polar_site][water][2];
        if (atom2 < 0 || atom2 >= n_atoms) return false;
        Real3 const xyz2 = ligand_coords[pose][atom2];
        if (!live_lk_ball_finite(xyz2)) return false;
        if (ligand_water_is_ring[polar_site][water]) {
          xyz1 = (xyz1 + xyz2) / Real(2);
        }
        waters.row(water) = common::build_coordinate<Real>::V(
            xyz0,
            xyz1,
            xyz2,
            parameters[0],
            parameters[1],
            parameters[2]);
      }
      return true;
    };
    auto load_receptor_waters =
        [&] TMOL_DEVICE_FUNC(Int atom, Waters& waters) -> bool {
      if (atom < 0 || atom >= n_receptor) return false;
      for (int water = 0; water < 4; ++water) {
        waters.row(water) = receptor_waters[atom][water];
      }
      return true;
    };
    auto propagate_water_gradient = [&] TMOL_DEVICE_FUNC(
                                        Int polar_site,
                                        int water,
                                        int channel,
                                        Real3 const& dE_dW) {
      Int const kind = ligand_water_kinds[polar_site][water];
      if (kind == 0) return;
      Int const atom0 = ligand_water_atoms[polar_site][water][0];
      Int const atom1 = ligand_water_atoms[polar_site][water][1];
      Real3 const xyz0 = ligand_coords[pose][atom0];
      Real3 xyz1 = ligand_coords[pose][atom1];
      Real3 const parameters = ligand_water_params[polar_site][water];
      if (kind == 1) {
        auto const dwater =
            build_don_water<Real>::dV(xyz0, xyz1, parameters[0]);
        add_gradient(channel, atom0, dwater.dD * dE_dW);
        add_gradient(channel, atom1, dwater.dH * dE_dW);
        return;
      }
      Int const atom2 = ligand_water_atoms[polar_site][water][2];
      Real3 const xyz2 = ligand_coords[pose][atom2];
      bool const ring = ligand_water_is_ring[polar_site][water] != 0;
      if (ring) xyz1 = (xyz1 + xyz2) / Real(2);
      auto const dwater = common::build_coordinate<Real>::dV(
          xyz0,
          xyz1,
          xyz2,
          parameters[0],
          parameters[1],
          parameters[2]);
      add_gradient(channel, atom0, dwater.dp * dE_dW);
      if (ring) {
        add_gradient(channel, atom1, Real(0.5) * dwater.dgp * dE_dW);
        add_gradient(channel, atom2, Real(0.5) * dwater.dgp * dE_dW);
      } else {
        add_gradient(channel, atom1, dwater.dgp * dE_dW);
      }
      add_gradient(channel, atom2, dwater.dggp * dE_dW);
    };
    auto add_direct_derivatives = [&] TMOL_DEVICE_FUNC(
                                      Int atom,
                                      lk_ball_dV_dReal3<Real> const& deriv) {
      add_gradient(0, atom, deriv.d_lkball_iso);
      add_gradient(1, atom, deriv.d_lkball);
      add_gradient(2, atom, deriv.d_lkbridge);
      add_gradient(3, atom, deriv.d_lkbridge_uncpl);
    };
    auto add_water_derivatives = [&] TMOL_DEVICE_FUNC(
                                     Int polar_site,
                                     lk_ball_dV_dWater<Real, 4> const& deriv) {
      for (int water = 0; water < 4; ++water) {
        propagate_water_gradient(
            polar_site,
            water,
            0,
            deriv.d_lkball_iso.row(water).transpose());
        propagate_water_gradient(
            polar_site,
            water,
            1,
            deriv.d_lkball.row(water).transpose());
        propagate_water_gradient(
            polar_site,
            water,
            2,
            deriv.d_lkbridge.row(water).transpose());
        propagate_water_gradient(
            polar_site,
            water,
            3,
            deriv.d_lkbridge_uncpl.row(water).transpose());
      }
    };
    auto add_energy = [&] TMOL_DEVICE_FUNC(lk_ball_Vt<Real> const& value) {
      energies[pose][0] += value.lkball_iso;
      energies[pose][1] += value.lkball;
      energies[pose][2] += value.lkbridge;
      energies[pose][3] += value.lkbridge_uncpl;
    };

    for (int ligand_site = 0; ligand_site < n_ligand_polar; ++ligand_site) {
      Int const ligand_atom = ligand_polar_atoms[ligand_site];
      if (ligand_atom < 0 || ligand_atom >= n_atoms) {
        valid = false;
        continue;
      }
      Real3 const I = ligand_coords[pose][ligand_atom];
      Waters WI;
      if (!safe_cell_anchor(I)
          || !build_ligand_waters(ligand_site, WI)) {
        valid = false;
        continue;
      }
      auto score_pair = [&] TMOL_DEVICE_FUNC(Int receptor_atom) {
        if (receptor_atom < 0 || receptor_atom >= n_receptor) {
          valid = false;
          return;
        }
        Real3 const J = receptor_coords[receptor_atom];
        Real const distance = (I - J).norm();
        if (!std::isfinite(distance)) {
          valid = false;
          return;
        }
        if (distance >= cutoff) return;
        Waters WJ;
        if (!load_receptor_waters(receptor_atom, WJ)) {
          valid = false;
          return;
        }
        auto const value = lk_ball_score<Real, 4>::V(
            I,
            J,
            WI,
            WJ,
            Real(5),
            ligand_atom_params[ligand_atom],
            receptor_params[receptor_atom],
            globals);
        auto const deriv = lk_ball_score<Real, 4>::dV(
            I,
            J,
            WI,
            WJ,
            Real(5),
            ligand_atom_params[ligand_atom],
            receptor_params[receptor_atom],
            globals);
        add_energy(value);
        add_direct_derivatives(ligand_atom, deriv.dI);
        add_water_derivatives(ligand_site, deriv.dWI);
      };
      if (!use_cell_list) {
        for (Int entry = 0; entry < n_receptor_entries; ++entry) {
          score_pair(receptor_cell_indices[entry]);
        }
      } else {
        Int const cx = cell_coordinate(I[0], 0);
        Int const cy = cell_coordinate(I[1], 1);
        Int const cz = cell_coordinate(I[2], 2);
        for (Int dx = -1; dx <= 1; ++dx) {
          Int const x = cx + dx;
          if (x < 0 || x >= cell_dims[0]) continue;
          for (Int dy = -1; dy <= 1; ++dy) {
            Int const y = cy + dy;
            if (y < 0 || y >= cell_dims[1]) continue;
            for (Int dz = -1; dz <= 1; ++dz) {
              Int const z = cz + dz;
              if (z < 0 || z >= cell_dims[2]) continue;
              Int const cell = cell_index(x, y, z);
              if (cell < 0) {
                valid = false;
                continue;
              }
              Int const begin = receptor_cell_offsets[cell];
              Int const end = receptor_cell_offsets[cell + 1];
              if (begin < 0 || end < begin || end > n_receptor_entries) {
                valid = false;
                continue;
              }
              for (Int entry = begin; entry < end; ++entry) {
                score_pair(receptor_cell_indices[entry]);
              }
            }
          }
        }
      }
    }

    for (int ligand_site = 0; ligand_site < n_ligand_heavy; ++ligand_site) {
      Int const ligand_atom = ligand_heavy_atoms[ligand_site];
      if (ligand_atom < 0 || ligand_atom >= n_atoms) {
        valid = false;
        continue;
      }
      Real3 const J = ligand_coords[pose][ligand_atom];
      if (!safe_cell_anchor(J)) {
        valid = false;
        continue;
      }
      Int const ligand_polar_site = ligand_atom_to_polar[ligand_atom];
      Waters WJ = Waters::Constant(nan);
      if (ligand_polar_site >= 0
          && !build_ligand_waters(ligand_polar_site, WJ)) {
        valid = false;
        continue;
      }
      if (ligand_polar_site >= n_ligand_polar) {
        valid = false;
        continue;
      }
      auto score_pair = [&] TMOL_DEVICE_FUNC(Int receptor_atom) {
        if (receptor_atom < 0 || receptor_atom >= n_receptor) {
          valid = false;
          return;
        }
        Real3 const I = receptor_coords[receptor_atom];
        Real const distance = (I - J).norm();
        if (!std::isfinite(distance)) {
          valid = false;
          return;
        }
        if (distance >= cutoff) return;
        Waters WI;
        if (!load_receptor_waters(receptor_atom, WI)) {
          valid = false;
          return;
        }
        auto const value = lk_ball_score<Real, 4>::V(
            I,
            J,
            WI,
            WJ,
            Real(5),
            receptor_params[receptor_atom],
            ligand_atom_params[ligand_atom],
            globals);
        auto const deriv = lk_ball_score<Real, 4>::dV(
            I,
            J,
            WI,
            WJ,
            Real(5),
            receptor_params[receptor_atom],
            ligand_atom_params[ligand_atom],
            globals);
        add_energy(value);
        add_direct_derivatives(ligand_atom, deriv.dJ);
        if (ligand_polar_site >= 0) {
          add_water_derivatives(ligand_polar_site, deriv.dWJ);
        }
      };
      if (!use_cell_list) {
        for (Int entry = 0; entry < n_receptor_polar_entries; ++entry) {
          score_pair(receptor_polar_cell_indices[entry]);
        }
      } else {
        Int const cx = cell_coordinate(J[0], 0);
        Int const cy = cell_coordinate(J[1], 1);
        Int const cz = cell_coordinate(J[2], 2);
        for (Int dx = -1; dx <= 1; ++dx) {
          Int const x = cx + dx;
          if (x < 0 || x >= cell_dims[0]) continue;
          for (Int dy = -1; dy <= 1; ++dy) {
            Int const y = cy + dy;
            if (y < 0 || y >= cell_dims[1]) continue;
            for (Int dz = -1; dz <= 1; ++dz) {
              Int const z = cz + dz;
              if (z < 0 || z >= cell_dims[2]) continue;
              Int const cell = cell_index(x, y, z);
              if (cell < 0) {
                valid = false;
                continue;
              }
              Int const begin = receptor_polar_cell_offsets[cell];
              Int const end = receptor_polar_cell_offsets[cell + 1];
              if (begin < 0 || end < begin
                  || end > n_receptor_polar_entries) {
                valid = false;
                continue;
              }
              for (Int entry = begin; entry < end; ++entry) {
                score_pair(receptor_polar_cell_indices[entry]);
              }
            }
          }
        }
      }
    }

    if (!valid) set_nan();
  };
  DeviceOps<D>::template forall<launch_t>(mgr, n_poses, evaluate_pose);
  return std::make_tuple(energies_t, gradients_t);
}

}  // namespace potentials
}  // namespace gridscore
}  // namespace score
}  // namespace tmol
