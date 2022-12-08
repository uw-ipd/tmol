#pragma once

#include <tmol/utility/tensor/TensorAccessor.h>

#include <tmol/score/common/data_loading.hh>
#include <tmol/score/elec/potentials/params.hh>
#include <tmol/score/elec/potentials/potentials.hh>

// #include <tmol/score/ljlk/potentials/common.hh>
// #include <tmol/score/ljlk/potentials/lj.hh>
// #include <tmol/score/ljlk/potentials/lk_isotropic.hh>

namespace tmol {
namespace score {
namespace elec {
namespace potentials {

template <typename Real>
class ElecSingleResData {
 public:
  int block_type;
  int block_coord_offset;
  int n_atoms;
  int n_conn;
  Real *coords;
  Real *charges;
  unsigned char *path_dist;
};

template <typename Real>
class ElecScoringData {
 public:
  int pose_ind;
  ElecSingleResData<Real> r1;
  ElecSingleResData<Real> r2;
  int max_important_bond_separation;
  int min_separation;
  bool in_count_pair_striking_dist;
  unsigned char *conn_seps;
  ElecGlobalParams<Real> global_params;
  Real total_elec;
};

template <typename Real, int TILE_SIZE, int MAX_N_CONN>
struct ElecBlockPairSharedData {
  Real coords1[TILE_SIZE * 3];  // 786 bytes for coords
  Real coords2[TILE_SIZE * 3];
  Real charges1[TILE_SIZE];  // 1536 bytes for params
  Real charges2[TILE_SIZE];
  unsigned char conn_ats1[MAX_N_CONN];  // 8 bytes
  unsigned char conn_ats2[MAX_N_CONN];
  unsigned char path_dist1[MAX_N_CONN * TILE_SIZE];  // 256 bytes
  unsigned char path_dist2[MAX_N_CONN * TILE_SIZE];
  unsigned char conn_seps[MAX_N_CONN * MAX_N_CONN];  // 64 bytes
};

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    int nt,
    typename Real>
void TMOL_DEVICE_FUNC elec_load_block_coords_and_charges_into_shared(
    TView<Vec<Real, 3>, 2, D> coords,
    TView<Real, 2, D> block_type_partial_charge,
    int pose_ind,
    ElecSingleResData<Real> &r_dat,
    int n_atoms_to_load,
    int start_atom) {
  // pre-condition: n_atoms_to_load < TILE_SIZE
  // Note that TILE_SIZE is not explicitly passed in, but is "present"
  // in r_dat.coords allocation
  DeviceDispatch<D>::template copy_contiguous_data<nt, 3>(
      r_dat.coords,
      reinterpret_cast<Real *>(
          &coords[pose_ind][r_dat.block_coord_offset + start_atom]),
      n_atoms_to_load * 3);
  DeviceDispatch<D>::template copy_contiguous_data<nt, 1>(
      r_dat.charges,
      &block_type_partial_charge[r_dat.block_type][start_atom],
      n_atoms_to_load);
}

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    int nt,
    int TILE_SIZE,
    typename Real,
    typename Int>
void TMOL_DEVICE_FUNC elec_load_block_into_shared(
    TView<Vec<Real, 3>, 2, D> coords,
    TView<Real, 2, D> block_type_partial_charge,
    TView<Int, 3, D> block_type_representative_path_distance,
    int pose_ind,
    ElecSingleResData<Real> &r_dat,
    int n_atoms_to_load,
    int start_atom,
    bool count_pair_striking_dist,
    unsigned char *__restrict__ conn_ats) {
  elec_load_block_coords_and_charges_into_shared<DeviceDispatch, D, nt>(
      coords,
      block_type_partial_charge,
      pose_ind,
      r_dat,
      n_atoms_to_load,
      start_atom);

  auto copy_path_dists = ([=](int tid) {
    for (int count = tid; count < n_atoms_to_load; count += nt) {
      int const atid = start_atom + count;
      for (int j = 0; j < r_dat.n_conn; ++j) {
        unsigned char ij_path_dist =
            block_type_representative_path_distance[r_dat.block_type]
                                                   [conn_ats[j]][atid];
        r_dat.path_dist[j * TILE_SIZE + count] = ij_path_dist;
      }
    }
  });
  if (count_pair_striking_dist) {
    DeviceDispatch<D>::template for_each_in_workgroup<nt>(copy_path_dists);
  }
}

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    int nt,
    typename Int,
    typename Real,
    int TILE_SIZE,
    int MAX_N_CONN>
void TMOL_DEVICE_FUNC elec_load_tile_invariant_interres_data(
    TView<Int, 2, D> pose_stack_block_coord_offset,
    TView<Int, 3, D> pose_stack_min_bond_separation,
    TView<Int, 1, D> block_type_n_interblock_bonds,
    TView<Int, 2, D> block_type_atoms_forming_chemical_bonds,
    TView<Int, 5, D> pose_stack_inter_block_bondsep,
    TView<ElecGlobalParams<Real>, 1, D> global_params,
    int const max_important_bond_separation,
    int pose_ind,
    int block_ind1,
    int block_ind2,
    int block_type1,
    int block_type2,
    int n_atoms1,
    int n_atoms2,
    ElecScoringData<Real> &inter_dat,
    ElecBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN> &shared_m) {
  inter_dat.pose_ind = pose_ind;
  inter_dat.r1.block_type = block_type1;
  inter_dat.r2.block_type = block_type2;
  inter_dat.r1.block_coord_offset =
      pose_stack_block_coord_offset[pose_ind][block_ind1];
  inter_dat.r2.block_coord_offset =
      pose_stack_block_coord_offset[pose_ind][block_ind2];
  inter_dat.max_important_bond_separation = max_important_bond_separation;
  inter_dat.min_separation =
      pose_stack_min_bond_separation[pose_ind][block_ind1][block_ind2];
  inter_dat.in_count_pair_striking_dist =
      inter_dat.min_separation <= max_important_bond_separation;
  inter_dat.r1.n_atoms = n_atoms1;
  inter_dat.r2.n_atoms = n_atoms2;
  inter_dat.r1.n_conn = block_type_n_interblock_bonds[block_type1];
  inter_dat.r2.n_conn = block_type_n_interblock_bonds[block_type2];

  // set the pointers in inter_dat to point at the shared-memory arrays
  inter_dat.r1.coords = shared_m.coords1;
  inter_dat.r2.coords = shared_m.coords2;
  inter_dat.r1.charges = shared_m.charges1;
  inter_dat.r2.charges = shared_m.charges2;
  inter_dat.r1.path_dist = shared_m.path_dist1;
  inter_dat.r2.path_dist = shared_m.path_dist2;
  inter_dat.conn_seps = shared_m.conn_seps;

  // Count pair setup that does not depend on which tile we are
  // operating on
  if (inter_dat.in_count_pair_striking_dist) {
    // Load data into shared arrays
    auto load_count_pair_conn_at_data = ([&](int tid) {
      int n_conn_tot = inter_dat.r1.n_conn + inter_dat.r2.n_conn
                       + inter_dat.r1.n_conn * inter_dat.r2.n_conn;
      for (int count = tid; count < n_conn_tot; count += nt) {
        if (count < inter_dat.r1.n_conn) {
          int const conn_ind = count;
          shared_m.conn_ats1[conn_ind] =
              block_type_atoms_forming_chemical_bonds[block_type1][conn_ind];
        } else if (count < inter_dat.r1.n_conn + inter_dat.r2.n_conn) {
          int const conn_ind = count - inter_dat.r1.n_conn;
          shared_m.conn_ats2[conn_ind] =
              block_type_atoms_forming_chemical_bonds[block_type2][conn_ind];
        } else {
          int const conn_ind =
              count - inter_dat.r1.n_conn - inter_dat.r2.n_conn;
          int conn1 = conn_ind / inter_dat.r2.n_conn;
          int conn2 = conn_ind % inter_dat.r2.n_conn;
          shared_m.conn_seps[conn_ind] =
              pose_stack_inter_block_bondsep[pose_ind][block_ind1][block_ind2]
                                            [conn1][conn2];
        }
      }
    });
    // On CPU: a for loop executed once; on GPU threads within the
    // workgroup working in parallel will just continue to work in
    // parallel
    DeviceDispatch<D>::template for_each_in_workgroup<nt>(
        load_count_pair_conn_at_data);
  }

  // Final data members
  inter_dat.global_params = global_params[0];
  inter_dat.total_elec = 0;
}

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    int nt,
    int TILE_SIZE,
    int MAX_N_CONN,
    typename Real,
    typename Int>
void TMOL_DEVICE_FUNC elec_load_interres1_tile_data_to_shared(
    TView<Vec<Real, 3>, 2, D> coords,
    TView<Real, 2, D> block_type_partial_charge,
    TView<Int, 3, D> block_type_representative_path_distance,
    int tile_ind,
    int start_atom1,
    int n_atoms_to_load1,
    ElecScoringData<Real> &inter_dat,
    ElecBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN> &shared_m) {
  elec_load_block_into_shared<DeviceDispatch, D, nt, TILE_SIZE>(
      coords,
      block_type_partial_charge,
      block_type_representative_path_distance,
      inter_dat.pose_ind,
      inter_dat.r1,
      n_atoms_to_load1,
      start_atom1,
      inter_dat.in_count_pair_striking_dist,
      shared_m.conn_ats1);
}

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    int nt,
    int TILE_SIZE,
    int MAX_N_CONN,
    typename Real,
    typename Int>
void TMOL_DEVICE_FUNC elec_load_interres2_tile_data_to_shared(
    TView<Vec<Real, 3>, 2, D> coords,
    TView<Real, 2, D> block_type_partial_charge,
    TView<Int, 3, D> block_type_representative_path_distance,
    int tile_ind,
    int start_atom2,
    int n_atoms_to_load2,
    ElecScoringData<Real> &inter_dat,
    ElecBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN> &shared_m) {
  elec_load_block_into_shared<DeviceDispatch, D, nt, TILE_SIZE>(
      coords,
      block_type_partial_charge,
      block_type_representative_path_distance,
      inter_dat.pose_ind,
      inter_dat.r2,
      n_atoms_to_load2,
      start_atom2,
      inter_dat.in_count_pair_striking_dist,
      shared_m.conn_ats2);
}

// template <int TILE_SIZE, int MAX_N_CONN, typename Real>
// void TMOL_DEVICE_FUNC elec_load_interres_data_from_shared(
//     ElecBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN> &shared_m,
//     ElecScoringData<Real> &inter_dat) {
//
// }

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    int nt,
    typename Int,
    typename Real,
    int TILE_SIZE,
    int MAX_N_CONN>
void TMOL_DEVICE_FUNC elec_load_tile_invariant_intrares_data(
    TView<Int, 2, D> pose_stack_block_coord_offset,
    TView<ElecGlobalParams<Real>, 1, D> global_params,
    int const max_important_bond_separation,
    int pose_ind,
    int block_ind1,
    int block_type1,
    int n_atoms1,
    ElecScoringData<Real> &intra_dat,
    ElecBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN> &shared_m) {
  intra_dat.pose_ind = pose_ind;
  intra_dat.r1.block_type = block_type1;
  intra_dat.r2.block_type = block_type1;
  intra_dat.r1.block_coord_offset =
      pose_stack_block_coord_offset[pose_ind][block_ind1];
  intra_dat.r2.block_coord_offset = intra_dat.r1.block_coord_offset;
  intra_dat.max_important_bond_separation = max_important_bond_separation;

  // we are not going to load count pair data into shared memory because
  // we are not going to use that data from shared memory
  intra_dat.min_separation = 0;
  intra_dat.in_count_pair_striking_dist = false;

  intra_dat.r1.n_atoms = n_atoms1;
  intra_dat.r2.n_atoms = n_atoms1;
  intra_dat.r1.n_conn =
      0;  // not used! block_type_n_interblock_bonds[block_type1];
  intra_dat.r2.n_conn = 0;  // not used! intra_dat.r1.n_conn;

  // set the pointers in intra_dat to point at the
  // shared-memory arrays. Note that these arrays will be reset
  // later because which shared memory arrays we will use depends on
  // which tile pair we are evaluating!
  intra_dat.r1.coords = shared_m.coords1;    // depends on tile pair!
  intra_dat.r2.coords = shared_m.coords2;    // depends on tile pair!
  intra_dat.r1.charges = shared_m.charges1;  // depends on tile pair!
  intra_dat.r2.charges = shared_m.charges2;  // depends on tile pair!

  // these count pair arrays are not going to be used
  intra_dat.r1.path_dist = 0;
  intra_dat.r2.path_dist = 0;
  intra_dat.conn_seps = 0;

  // Final data members
  intra_dat.global_params = global_params[0];
  intra_dat.total_elec = 0;
}

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    int nt,
    int TILE_SIZE,
    int MAX_N_CONN,
    typename Real>
void TMOL_DEVICE_FUNC elec_load_intrares1_tile_data_to_shared(
    TView<Vec<Real, 3>, 2, D> coords,
    TView<Real, 2, D> block_type_partial_charge,
    int tile_ind,
    int start_atom1,
    int n_atoms_to_load1,
    ElecScoringData<Real> &intra_dat,
    ElecBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN> &shared_m) {
  elec_load_block_coords_and_charges_into_shared<DeviceDispatch, D, nt>(
      coords,
      block_type_partial_charge,
      intra_dat.pose_ind,
      intra_dat.r1,
      n_atoms_to_load1,
      start_atom1);
}

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    int nt,
    int TILE_SIZE,
    int MAX_N_CONN,
    typename Real>
void TMOL_DEVICE_FUNC elec_load_intrares2_tile_data_to_shared(
    TView<Vec<Real, 3>, 2, D> coords,
    TView<Real, 2, D> block_type_partial_charge,
    int tile_ind,
    int start_atom2,
    int n_atoms_to_load2,
    ElecScoringData<Real> &intra_dat,
    ElecBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN> &shared_m) {
  elec_load_block_coords_and_charges_into_shared<DeviceDispatch, D, nt>(
      coords,
      block_type_partial_charge,
      intra_dat.pose_ind,
      intra_dat.r2,
      n_atoms_to_load2,
      start_atom2);
}

template <int TILE_SIZE, int MAX_N_CONN, typename Real>
void TMOL_DEVICE_FUNC elec_load_intrares_data_from_shared(
    int tile_ind1,
    int tile_ind2,
    ElecBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN> &shared_m,
    ElecScoringData<Real> &intra_dat) {
  // set the pointers in intra_dat to point at the shared-memory arrays
  // If we are evaluating the energies between atoms in the same tile
  // then only the "1" shared-memory arrays will be loaded with data;
  // we will point the "2" memory pointers at the "1" arrays
  bool same_tile = tile_ind1 == tile_ind2;
  intra_dat.r1.coords = shared_m.coords1;
  intra_dat.r2.coords = (same_tile ? shared_m.coords1 : shared_m.coords2);
  intra_dat.r1.charges = shared_m.charges1;
  intra_dat.r2.charges = (same_tile ? shared_m.charges1 : shared_m.charges2);
}

template <typename Real>
TMOL_DEVICE_FUNC Real elec_atom_energy(
    int atom_tile_ind1,
    int atom_tile_ind2,
    ElecScoringData<Real> const &score_dat,
    int cp_separation) {
  using Real3 = Eigen::Matrix<Real, 3, 1>;

  Real3 coord1 = coord_from_shared(score_dat.r1.coords, atom_tile_ind1);
  Real3 coord2 = coord_from_shared(score_dat.r2.coords, atom_tile_ind2);

  Real const dist = distance<Real>::V(coord1, coord2);
  return elec(
      dist,
      score_dat.r1.charges[atom_tile_ind1],
      score_dat.r2.charges[atom_tile_ind2],
      Real(cp_separation),
      score_dat.global_params.D,
      score_dat.global_params.D0,
      score_dat.global_params.S,
      score_dat.global_params.min_dis,
      score_dat.global_params.max_dis);
}

template <typename Real, tmol::Device D>
TMOL_DEVICE_FUNC Real elec_atom_energy_and_derivs_full(
    int atom_tile_ind1,
    int atom_tile_ind2,
    int start_atom1,
    int start_atom2,
    ElecScoringData<Real> const &score_dat,
    int cp_separation,
    TView<Eigen::Matrix<Real, 3, 1>, 2, D> dV_dcoords) {
  using Real3 = Eigen::Matrix<Real, 3, 1>;

  Real3 coord1 = coord_from_shared(score_dat.r1.coords, atom_tile_ind1);
  Real3 coord2 = coord_from_shared(score_dat.r2.coords, atom_tile_ind2);

  auto dist_r = distance<Real>::V_dV(coord1, coord2);
  auto &dist = dist_r.V;
  auto &ddist_dat1 = dist_r.dV_dA;
  auto &ddist_dat2 = dist_r.dV_dB;
  // auto lj = lj_score<Real>::V_dV(
  //     dist,
  //     cp_separation,
  //     score_dat.r1.params[atom_tile_ind1].lj_params(),
  //     score_dat.r2.params[atom_tile_ind2].lj_params(),
  //     score_dat.global_params);
  Real V(0.0), dV_ddist(0.0);
  tie(V, dV_ddist) = elec_delec_ddist(
      dist,
      score_dat.r1.charges[atom_tile_ind1],
      score_dat.r2.charges[atom_tile_ind2],
      Real(cp_separation),
      score_dat.global_params.D,
      score_dat.global_params.D0,
      score_dat.global_params.S,
      score_dat.global_params.min_dis,
      score_dat.global_params.max_dis);

  // all threads accumulate derivatives for atom 1 to global memory
  Vec<Real, 3> elec_dxyz_at1 = dV_ddist * ddist_dat1;
  for (int j = 0; j < 3; ++j) {
    if (elec_dxyz_at1[j] != 0) {
      accumulate<D, Real>::add(
          dV_dcoords[score_dat.pose_ind]
                    [score_dat.r1.block_coord_offset + atom_tile_ind1
                     + start_atom1][j],
          elec_dxyz_at1[j]);
    }
  }

  // all threads accumulate derivatives for atom 2 to global memory
  Vec<Real, 3> elec_dxyz_at2 = dV_ddist * ddist_dat2;
  for (int j = 0; j < 3; ++j) {
    if (elec_dxyz_at2[j] != 0) {
      accumulate<D, Real>::add(
          dV_dcoords[score_dat.pose_ind]
                    [score_dat.r2.block_coord_offset + atom_tile_ind2
                     + start_atom2][j],
          elec_dxyz_at2[j]);
    }
  }
  return V;
}

}  // namespace potentials
}  // namespace elec
}  // namespace score
}  // namespace tmol
