#pragma once

#include <cmath>
#include <limits>

#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/gridscore/potentials/live_hbond.hh>
#include <tmol/score/hbond/potentials/potentials.hh>

namespace tmol {
namespace score {
namespace gridscore {
namespace potentials {

template <typename Real>
EIGEN_DEVICE_FUNC bool live_hbond_finite(HBVec<Real, 3> const& xyz) {
  return std::isfinite(xyz[0]) && std::isfinite(xyz[1])
         && std::isfinite(xyz[2]);
}

template <typename Real>
EIGEN_DEVICE_FUNC Real live_hbond_distance(
    HBVec<Real, 3> const& first, HBVec<Real, 3> const& second) {
  return (first - second).norm();
}

template <
    template <tmol::Device> class DeviceOps,
    tmol::Device D,
    typename Real,
    typename Int>
auto LiveHBondDispatch<DeviceOps, D, Real, Int>::forward(
    ContextManager& mgr,
    TView<HBVec<Real, 3>, 2, D> ligand_coords,
    TView<Int, 2, D> ligand_donor_atoms,
    TView<Int, 1, D> ligand_donor_types,
    TView<Int, 2, D> ligand_acceptor_atoms,
    TView<Int, 1, D> ligand_acceptor_types,
    TView<HBVec<Real, 3>, 2, D> receptor_donors,
    TView<Int, 1, D> receptor_donor_types,
    TView<HBVec<Real, 3>, 2, D> receptor_acceptors,
    TView<Int, 1, D> receptor_acceptor_types,
    TView<Real, 1, D> cell_origin,
    TView<Int, 1, D> cell_dims,
    TView<Int, 1, D> receptor_donor_cell_offsets,
    TView<Int, 1, D> receptor_donor_cell_indices,
    TView<Int, 1, D> receptor_acceptor_cell_offsets,
    TView<Int, 1, D> receptor_acceptor_cell_indices,
    TView<hbond::potentials::HBondPairParams<Real>, 2, D> pair_params,
    TView<hbond::potentials::HBondPolynomials<double>, 2, D>
        pair_polynomials,
    TView<hbond::potentials::HBondGlobalParams<Real>, 1, D> global_params)
    -> std::tuple<TPack<Real, 1, D>, TPack<HBVec<Real, 3>, 2, D>> {
  using hbond::potentials::hbond_score;

  int const n_poses = ligand_coords.size(0);
  int const n_atoms = ligand_coords.size(1);
  int const n_ligand_donors = ligand_donor_atoms.size(0);
  int const n_ligand_acceptors = ligand_acceptor_atoms.size(0);
  Int const n_receptor_donors = receptor_donors.size(0);
  Int const n_receptor_acceptors = receptor_acceptors.size(0);
  Int const n_donor_types = pair_params.size(0);
  Int const n_acceptor_types = pair_params.size(1);
  Int const n_cells = receptor_donor_cell_offsets.size(0) - 1;
  Int const n_receptor_donor_entries = receptor_donor_cell_indices.size(0);
  Int const n_receptor_acceptor_entries = receptor_acceptor_cell_indices.size(0);
  auto energies_t = TPack<Real, 1, D>::zeros({n_poses});
  auto gradients_t = TPack<HBVec<Real, 3>, 2, D>::zeros({n_poses, n_atoms});
  auto energies = energies_t.view;
  auto gradients = gradients_t.view;
  LAUNCH_BOX_32;
  auto evaluate_pose = [=] TMOL_DEVICE_FUNC(int pose) {
    // Read the global-parameter and cell-dimension scalars on-device: these
    // views may reference device memory, so dereferencing them on the host
    // (outside this lambda) segfaults for the CUDA dispatch.
    auto const globals = global_params[0];
    Real const cutoff = globals.max_ha_dis;
    Int const dim_y = cell_dims[1];
    Int const dim_z = cell_dims[2];
    Real total = 0;
    bool valid = true;

    auto set_nan = [&] TMOL_DEVICE_FUNC() {
      Real const nan = std::numeric_limits<Real>::quiet_NaN();
      energies[pose] = nan;
      for (int atom = 0; atom < n_atoms; ++atom) {
        for (int axis = 0; axis < 3; ++axis) {
          gradients[pose][atom][axis] = nan;
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
      return result < n_cells ? result : -1;
    };
    auto add_gradient = [&] TMOL_DEVICE_FUNC(
                            Int atom, HBVec<Real, 3> const& gradient) {
      for (int axis = 0; axis < 3; ++axis) {
        gradients[pose][atom][axis] += gradient[axis];
      }
    };

    for (int ligand_site = 0; ligand_site < n_ligand_donors; ++ligand_site) {
      Int const donor_atom = ligand_donor_atoms[ligand_site][0];
      Int const hydrogen_atom = ligand_donor_atoms[ligand_site][1];
      Int const donor_type = ligand_donor_types[ligand_site];
      if (donor_atom < 0 || donor_atom >= n_atoms || hydrogen_atom < 0
          || hydrogen_atom >= n_atoms || donor_type < 0
          || donor_type >= n_donor_types) {
        valid = false;
        continue;
      }
      HBVec<Real, 3> const donor = ligand_coords[pose][donor_atom];
      HBVec<Real, 3> const hydrogen = ligand_coords[pose][hydrogen_atom];
      if (!live_hbond_finite(donor) || !live_hbond_finite(hydrogen)) {
        valid = false;
        continue;
      }
      Real const max_cell_coordinate = Real(1 << 30);
      if (std::abs((hydrogen[0] - cell_origin[0]) / cutoff)
              > max_cell_coordinate
          || std::abs((hydrogen[1] - cell_origin[1]) / cutoff)
                 > max_cell_coordinate
          || std::abs((hydrogen[2] - cell_origin[2]) / cutoff)
                 > max_cell_coordinate) {
        valid = false;
        continue;
      }
      Int const center_x = cell_coordinate(hydrogen[0], 0);
      Int const center_y = cell_coordinate(hydrogen[1], 1);
      Int const center_z = cell_coordinate(hydrogen[2], 2);
      for (Int dx = -1; dx <= 1; ++dx) {
        Int const x = center_x + dx;
        if (x < 0 || x >= cell_dims[0]) continue;
        for (Int dy = -1; dy <= 1; ++dy) {
          Int const y = center_y + dy;
          if (y < 0 || y >= cell_dims[1]) continue;
          for (Int dz = -1; dz <= 1; ++dz) {
            Int const z = center_z + dz;
            if (z < 0 || z >= cell_dims[2]) continue;
            Int const cell = cell_index(x, y, z);
            if (cell < 0) {
              valid = false;
              continue;
            }
            Int const cell_begin = receptor_acceptor_cell_offsets[cell];
            Int const cell_end = receptor_acceptor_cell_offsets[cell + 1];
            if (cell_begin < 0 || cell_end < cell_begin
                || cell_end > n_receptor_acceptor_entries) {
              valid = false;
              continue;
            }
            for (Int entry = cell_begin; entry < cell_end; ++entry) {
              Int const receptor_site = receptor_acceptor_cell_indices[entry];
              if (receptor_site < 0 || receptor_site >= n_receptor_acceptors) {
                valid = false;
                continue;
              }
              HBVec<Real, 3> const acceptor = receptor_acceptors[receptor_site][0];
              HBVec<Real, 3> const base = receptor_acceptors[receptor_site][1];
              HBVec<Real, 3> const base0 = receptor_acceptors[receptor_site][2];
              Int const acceptor_type = receptor_acceptor_types[receptor_site];
              if (!live_hbond_finite(acceptor) || !live_hbond_finite(base)
                  || !live_hbond_finite(base0) || acceptor_type < 0
                  || acceptor_type >= n_acceptor_types) {
                valid = false;
                continue;
              }
              if (live_hbond_distance(hydrogen, acceptor) >= cutoff) {
                continue;
              }
              auto const result = hbond_score<Real, Int>::V_dV(
                  donor,
                  hydrogen,
                  acceptor,
                  base,
                  base0,
                  pair_params[donor_type][acceptor_type],
                  pair_polynomials[donor_type][acceptor_type],
                  globals);
              total += result.V;
              add_gradient(donor_atom, result.dV_dD);
              add_gradient(hydrogen_atom, result.dV_dH);
            }
          }
        }
      }
    }

    for (int ligand_site = 0; ligand_site < n_ligand_acceptors; ++ligand_site) {
      Int const acceptor_atom = ligand_acceptor_atoms[ligand_site][0];
      Int const base_atom = ligand_acceptor_atoms[ligand_site][1];
      Int const base0_atom = ligand_acceptor_atoms[ligand_site][2];
      Int const acceptor_type = ligand_acceptor_types[ligand_site];
      if (acceptor_atom < 0 || acceptor_atom >= n_atoms || base_atom < 0
          || base_atom >= n_atoms || base0_atom < 0 || base0_atom >= n_atoms
          || acceptor_type < 0 || acceptor_type >= n_acceptor_types) {
        valid = false;
        continue;
      }
      HBVec<Real, 3> const acceptor = ligand_coords[pose][acceptor_atom];
      HBVec<Real, 3> const base = ligand_coords[pose][base_atom];
      HBVec<Real, 3> const base0 = ligand_coords[pose][base0_atom];
      if (!live_hbond_finite(acceptor) || !live_hbond_finite(base)
          || !live_hbond_finite(base0)) {
        valid = false;
        continue;
      }
      Real const max_cell_coordinate = Real(1 << 30);
      if (std::abs((acceptor[0] - cell_origin[0]) / cutoff)
              > max_cell_coordinate
          || std::abs((acceptor[1] - cell_origin[1]) / cutoff)
                 > max_cell_coordinate
          || std::abs((acceptor[2] - cell_origin[2]) / cutoff)
                 > max_cell_coordinate) {
        valid = false;
        continue;
      }
      Int const center_x = cell_coordinate(acceptor[0], 0);
      Int const center_y = cell_coordinate(acceptor[1], 1);
      Int const center_z = cell_coordinate(acceptor[2], 2);
      for (Int dx = -1; dx <= 1; ++dx) {
        Int const x = center_x + dx;
        if (x < 0 || x >= cell_dims[0]) continue;
        for (Int dy = -1; dy <= 1; ++dy) {
          Int const y = center_y + dy;
          if (y < 0 || y >= cell_dims[1]) continue;
          for (Int dz = -1; dz <= 1; ++dz) {
            Int const z = center_z + dz;
            if (z < 0 || z >= cell_dims[2]) continue;
            Int const cell = cell_index(x, y, z);
            if (cell < 0) {
              valid = false;
              continue;
            }
            Int const cell_begin = receptor_donor_cell_offsets[cell];
            Int const cell_end = receptor_donor_cell_offsets[cell + 1];
            if (cell_begin < 0 || cell_end < cell_begin
                || cell_end > n_receptor_donor_entries) {
              valid = false;
              continue;
            }
            for (Int entry = cell_begin; entry < cell_end; ++entry) {
              Int const receptor_site = receptor_donor_cell_indices[entry];
              if (receptor_site < 0 || receptor_site >= n_receptor_donors) {
                valid = false;
                continue;
              }
              HBVec<Real, 3> const receptor_donor =
                  receptor_donors[receptor_site][0];
              HBVec<Real, 3> const hydrogen = receptor_donors[receptor_site][1];
              Int const donor_type = receptor_donor_types[receptor_site];
              if (!live_hbond_finite(receptor_donor)
                  || !live_hbond_finite(hydrogen) || donor_type < 0
                  || donor_type >= n_donor_types) {
                valid = false;
                continue;
              }
              if (live_hbond_distance(hydrogen, acceptor) >= cutoff) {
                continue;
              }
              auto const result = hbond_score<Real, Int>::V_dV(
                  receptor_donor,
                  hydrogen,
                  acceptor,
                  base,
                  base0,
                  pair_params[donor_type][acceptor_type],
                  pair_polynomials[donor_type][acceptor_type],
                  globals);
              total += result.V;
              add_gradient(acceptor_atom, result.dV_dA);
              add_gradient(base_atom, result.dV_dB);
              add_gradient(base0_atom, result.dV_dB0);
            }
          }
        }
      }
    }

    if (valid) {
      energies[pose] = total;
    } else {
      set_nan();
    }
  };
  DeviceOps<D>::template forall<launch_t>(mgr, n_poses, evaluate_pose);
  return std::make_tuple(energies_t, gradients_t);
}

}  // namespace potentials
}  // namespace gridscore
}  // namespace score
}  // namespace tmol
