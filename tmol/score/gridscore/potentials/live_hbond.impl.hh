#pragma once

#include <cmath>
#include <limits>

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/gridscore/potentials/live_hbond.hh>
#include <tmol/score/hbond/potentials/potentials.hh>

namespace tmol {
namespace score {
namespace gridscore {
namespace potentials {

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
    TView<hbond::potentials::HBondPolynomials<double>, 2, D> pair_polynomials,
    TView<hbond::potentials::HBondGlobalParams<Real>, 1, D> global_params)
    -> std::tuple<TPack<Real, 1, D>, TPack<HBVec<Real, 3>, 2, D>> {
  using hbond::potentials::hbond_score;

  int const n_poses = ligand_coords.size(0);
  int const n_atoms = ligand_coords.size(1);
  int const n_ligand_donors = ligand_donor_atoms.size(0);
  int const n_ligand_acceptors = ligand_acceptor_atoms.size(0);
  auto energies_t = TPack<Real, 1, D>::zeros({n_poses});
  auto gradients_t = TPack<HBVec<Real, 3>, 2, D>::zeros({n_poses, n_atoms});
  auto energies = energies_t.view;
  auto gradients = gradients_t.view;
  // One work item per (pose, ligand site): a donor hydrogen is scored against
  // the receptor acceptors around it, a ligand acceptor against the receptor
  // donors.
  int const n_sites = n_ligand_donors + n_ligand_acceptors;
  LAUNCH_BOX_32;
  auto evaluate_site = [=] TMOL_DEVICE_FUNC(int work) {
    using Vec3 = HBVec<Real, 3>;
    int const pose = work / n_sites;
    int const site = work % n_sites;
    bool const is_donor = site < n_ligand_donors;
    auto const globals = global_params[0];
    Real const cutoff = globals.max_ha_dis;

    // Ligand site: three atoms (donor, hydrogen, unused | acceptor, base,
    // base0).
    Int atoms[3] = {-1, -1, -1};
    Int site_type;
    if (is_donor) {
      atoms[0] = ligand_donor_atoms[site][0];
      atoms[1] = ligand_donor_atoms[site][1];
      site_type = ligand_donor_types[site];
    } else {
      for (int k = 0; k < 3; ++k) {
        atoms[k] = ligand_acceptor_atoms[site - n_ligand_donors][k];
      }
      site_type = ligand_acceptor_types[site - n_ligand_donors];
    }
    Vec3 const p0 = ligand_coords[pose][atoms[0]];
    Vec3 const p1 = ligand_coords[pose][atoms[1]];
    Vec3 p2 = Vec3::Zero();
    if (!is_donor) p2 = ligand_coords[pose][atoms[2]];
    Vec3 const anchor = is_donor ? p1 : p0;  // hydrogen | acceptor

    TView<Int, 1, D> const offsets =
        is_donor ? receptor_acceptor_cell_offsets : receptor_donor_cell_offsets;
    TView<Int, 1, D> const indices =
        is_donor ? receptor_acceptor_cell_indices : receptor_donor_cell_indices;
    Real total = 0;
    Vec3 grad[3] = {Vec3::Zero(), Vec3::Zero(), Vec3::Zero()};

    Int cell_xyz[3];
    for (int axis = 0; axis < 3; ++axis) {
      cell_xyz[axis] = static_cast<Int>(
          std::floor((anchor[axis] - cell_origin[axis]) / cutoff));
    }
    for (Int dx = -1; dx <= 1; ++dx) {
      Int const x = cell_xyz[0] + dx;
      if (x < 0 || x >= cell_dims[0]) continue;
      for (Int dy = -1; dy <= 1; ++dy) {
        Int const y = cell_xyz[1] + dy;
        if (y < 0 || y >= cell_dims[1]) continue;
        for (Int dz = -1; dz <= 1; ++dz) {
          Int const z = cell_xyz[2] + dz;
          if (z < 0 || z >= cell_dims[2]) continue;
          Int const cell = (x * cell_dims[1] + y) * cell_dims[2] + z;
          for (Int entry = offsets[cell]; entry < offsets[cell + 1]; ++entry) {
            Int const receptor_site = indices[entry];
            if (is_donor) {
              Vec3 const acceptor = receptor_acceptors[receptor_site][0];
              if ((p1 - acceptor).norm() >= cutoff) continue;
              auto const r = hbond_score<Real, Int>::V_dV(
                  p0,
                  p1,
                  acceptor,
                  receptor_acceptors[receptor_site][1],
                  receptor_acceptors[receptor_site][2],
                  pair_params[site_type]
                             [receptor_acceptor_types[receptor_site]],
                  pair_polynomials[site_type]
                                  [receptor_acceptor_types[receptor_site]],
                  globals);
              total += r.V;
              grad[0] += r.dV_dD;
              grad[1] += r.dV_dH;
            } else {
              Vec3 const hydrogen = receptor_donors[receptor_site][1];
              if ((hydrogen - p0).norm() >= cutoff) continue;
              auto const r = hbond_score<Real, Int>::V_dV(
                  receptor_donors[receptor_site][0],
                  hydrogen,
                  p0,
                  p1,
                  p2,
                  pair_params[receptor_donor_types[receptor_site]][site_type],
                  pair_polynomials[receptor_donor_types[receptor_site]]
                                  [site_type],
                  globals);
              total += r.V;
              grad[0] += r.dV_dA;
              grad[1] += r.dV_dB;
              grad[2] += r.dV_dB0;
            }
          }
        }
      }
    }
    common::accumulate<D, Real>::add(energies[pose], total);
    for (int k = 0; k < (is_donor ? 2 : 3); ++k) {
      common::accumulate<D, Vec3>::add(gradients[pose][atoms[k]], grad[k]);
    }
  };
  DeviceOps<D>::template forall<launch_t>(
      mgr, n_poses * n_sites, evaluate_site);
  return std::make_tuple(energies_t, gradients_t);
}

}  // namespace potentials
}  // namespace gridscore
}  // namespace score
}  // namespace tmol
