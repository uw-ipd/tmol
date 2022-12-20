#pragma once

#include <tmol/utility/tensor/TensorAccessor.h>

#include <tmol/score/common/data_loading.hh>
#include <tmol/score/hbond/potentials/params.hh>
#include <tmol/score/hbond/potentials/potentials.hh>

namespace tmol {
namespace score {
namespace hbond {
namespace potentials {

template <typename Real>
class HBondSingleResData {
 public:
  int block_ind;
  int block_type;
  int block_coord_offset;
  int n_atoms;
  int n_conn;
  Real *coords;
  unsigned char n_donH;
  unsigned char n_acc;
  unsigned char *donH_tile_inds;
  unsigned char *acc_tile_inds;
  unsigned char *donH_type;
  unsigned char *acc_type;
  unsigned char *acc_hybridization;
  unsigned char *path_dist;
};

template <tmol::Device Dev, typename Real, typename Int>
class HBondResPairData {
 public:
  int pose_ind;
  int max_important_bond_separation;
  int min_separation;
  bool in_count_pair_striking_dist;
  unsigned char *conn_seps;

  // load global params once; store totalE
  HBondGlobalParams<Real> global_params;
  Real total_hbond;

  // NOTE: the remaining data members of this class
  // "duplicate" the tensor argumentss to the kernel.
  // Fortunately, nvcc will be able to tell that these
  // are the same members that are passed in and will
  // not duplicate their allocation in register memory

  // If the hbond involves atoms from other residues, we need
  // to be able to retrieve their coordinates
  TView<Vec<Real, 3>, 2, Dev> coords;
  TView<Int, 2, Dev> pose_stack_block_coord_offset;
  TView<Int, 2, Dev> pose_stack_block_type;

  // For determining which atoms to retrieve from neighboring
  // residues we have to know how the blocks in the Pose
  // are connected
  TView<Vec<Int, 2>, 3, Dev> pose_stack_inter_residue_connections;

  // And we need to know the properties of the block types
  // that we are working with to iterating across chemical bonds
  TView<Int, 1, Dev> block_type_n_all_bonds;
  TView<Int, 3, Dev> block_type_all_bonds;
  TView<Int, 2, Dev> block_type_atom_all_bond_ranges;
  TView<Int, 2, Dev> block_type_atoms_forming_chemical_bonds;
  TView<Int, 2, Dev> block_type_atom_is_hydrogen;

  // Parameters that define the hbond energies
  TView<HBondPairParams<Real>, 2, Dev> pair_params;
  TView<HBondPolynomials<double>, 2, Dev> pair_polynomials;
};

template <tmol::Device Dev, typename Real, typename Int>
class HBondScoringData {
 public:
  HBondSingleResData<Real> r1;
  HBondSingleResData<Real> r2;
  HBondResPairData<Dev, Real, Int> pair_data;
};

template <typename Real, int TILE_SIZE, int MAX_N_CONN>
struct HBondBlockPairSharedData {
  Real coords1[TILE_SIZE * 3];  // 768 bytes for coords
  Real coords2[TILE_SIZE * 3];
  unsigned char n_donH1;  // 4 bytes for counts
  unsigned char n_donH2;
  unsigned char n_acc1;
  unsigned char n_acc2;
  unsigned char don_inds1[TILE_SIZE];  // 256 bytes for indices
  unsigned char don_inds2[TILE_SIZE];
  unsigned char acc_inds1[TILE_SIZE];
  unsigned char acc_inds2[TILE_SIZE];
  unsigned char don_type1[TILE_SIZE];
  unsigned char don_type2[TILE_SIZE];
  unsigned char acc_type1[TILE_SIZE];
  unsigned char acc_type2[TILE_SIZE];
  unsigned char acc_hybridization1[TILE_SIZE];
  unsigned char acc_hybridization2[TILE_SIZE];

  unsigned char conn_ats1[MAX_N_CONN];  // 8 bytes
  unsigned char conn_ats2[MAX_N_CONN];
  unsigned char path_dist1[MAX_N_CONN * TILE_SIZE];  // 256 bytes
  unsigned char path_dist2[MAX_N_CONN * TILE_SIZE];
  unsigned char conn_seps[MAX_N_CONN * MAX_N_CONN];  // 64 bytes
};

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    typename Real,
    typename Int>
void TMOL_DEVICE_FUNC hbond_load_block_coords_and_params_into_shared(
    TView<Vec<Real, 3>, 2, Dev> coords,
    TView<Int, 3, Dev> block_type_tile_donH_inds,
    TView<Int, 3, Dev> block_type_tile_acc_inds,
    TView<Int, 3, Dev> block_type_tile_donor_type,
    TView<Int, 3, Dev> block_type_tile_acceptor_type,
    TView<Int, 3, Dev> block_type_tile_hybridization,
    int pose_ind,
    int tile_ind,
    HBondSingleResData<Real> &r_dat,
    int n_atoms_to_load,
    int start_atom) {
  // pre-condition: n_atoms_to_load < TILE_SIZE
  // Note that TILE_SIZE is not explicitly passed in, but is "present"
  // in r_dat.coords allocation
  DeviceDispatch<Dev>::template copy_contiguous_data<nt, 3>(
      r_dat.coords,
      reinterpret_cast<Real *>(
          &coords[pose_ind][r_dat.block_coord_offset + start_atom]),
      n_atoms_to_load * 3);
  DeviceDispatch<Dev>::template copy_contiguous_data<nt, 1>(
      r_dat.donH_tile_inds,
      &block_type_tile_donH_inds[r_dat.block_type][tile_ind][0],
      r_dat.n_donH);
  DeviceDispatch<Dev>::template copy_contiguous_data<nt, 1>(
      r_dat.acc_tile_inds,
      &block_type_tile_acc_inds[r_dat.block_type][tile_ind][0],
      r_dat.n_acc);
  DeviceDispatch<Dev>::template copy_contiguous_data<nt, 1>(
      r_dat.don_type,
      &block_type_tile_donor_type[r_dat.block_type][tile_ind][0],
      r_dat.n_donH);
  DeviceDispatch<Dev>::template copy_contiguous_data<nt, 1>(
      r_dat.acc_type,
      &block_type_tile_acceptor_type[r_dat.block_type][tile_ind][0],
      r_dat.n_acc);
  DeviceDispatch<Dev>::template copy_contiguous_data<nt, 1>(
      r_dat.acc_tile_hybridization,
      &block_type_hybridization[r_dat.block_type][tile_ind][0],
      r_dat.n_acc);
}

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    int TILE_SIZE,
    typename Real,
    typename Int>
void TMOL_DEVICE_FUNC hbond_load_block_into_shared(
    TView<Vec<Real, 3>, 2, Dev> coords,
    TView<Int, 3, Dev> block_type_tile_donH_inds,
    TView<Int, 3, Dev> block_type_tile_acc_inds,
    TView<Int, 3, Dev> block_type_tile_donor_type,
    TView<Int, 3, Dev> block_type_tile_acceptor_type,
    TView<Int, 3, Dev> block_type_tile_hybridization,
    TView<Int, 3, Dev> block_type_path_distance,
    int pose_ind,
    int tile_ind,
    HBondSingleResData<Real> &r_dat,
    int n_atoms_to_load,
    int start_atom,
    bool count_pair_striking_dist,
    unsigned char *__restrict__ conn_ats) {
  hbond_load_block_coords_and_params_into_shared<DeviceDispatch, Dev, nt>(
      coords,
      block_type_tile_donH_inds,
      block_type_tile_acc_inds,
      block_type_tile_donor_type,
      block_type_tile_acceptor_type,
      block_type_tile_hybridization,
      pose_ind,
      tile_ind r_dat,
      n_atoms_to_load,
      start_atom);

  auto copy_path_dists = ([=](int tid) {
    for (int count = tid; count < n_atoms_to_load; count += nt) {
      int const atid = start_atom + count;
      for (int j = 0; j < r_dat.n_conn; ++j) {
        unsigned char ij_path_dist =
            block_type_path_distance[r_dat.block_type][conn_ats[j]][atid];
        r_dat.path_dist[j * TILE_SIZE + count] = ij_path_dist;
      }
    }
  });
  if (count_pair_striking_dist) {
    DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(copy_path_dists);
  }
}

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    typename Int,
    typename Real,
    int TILE_SIZE,
    int MAX_N_CONN>
void TMOL_DEVICE_FUNC hbond_load_tile_invariant_interres_data(
    TView<Vec<Real, 3>, 2, Dev> coords,
    TView<Int, 2, Dev> pose_stack_block_coord_offset,
    TView<Int, 2, Dev> pose_stack_block_type,
    TView<Vec<Int, 2>, 3, Dev> pose_stack_inter_residue_connections,
    TView<Int, 3, Dev> pose_stack_min_bond_separation,
    TView<Int, 5, Dev> pose_stack_inter_block_bondsep,

    TView<Int, 1, Dev> block_type_n_all_bonds,
    TView<Int, 3, Dev> block_type_all_bonds,
    TView<Int, 2, Dev> block_type_atom_all_bond_ranges,
    TView<Int, 1, Dev> block_type_n_interblock_bonds,
    TView<Int, 2, Dev> block_type_atoms_forming_chemical_bonds,
    TView<Int, 2, Dev> block_type_atom_is_hydrogen,

    TView<HBondPairParams<Real>, 2, Dev> pair_params,
    TView<HBondPolynomials<double>, 2, Dev> pair_polynomials,
    TView<HBondGlobalParams<Real>, 1, Dev> global_params,

    int const max_important_bond_separation,
    int pose_ind,
    int block_ind1,
    int block_ind2,
    int block_type1,
    int block_type2,
    int n_atoms1,
    int n_atoms2,
    HBondScoringData<Real> &inter_dat,
    HBondBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN> &shared_m) {
  inter_dat.pair_data.pose_ind = pose_ind;
  inter_dat.r1.block_type = block_type1;
  inter_dat.r2.block_type = block_type2;
  inter_dat.r1.block_coord_offset =
      pose_stack_block_coord_offset[pose_ind][block_ind1];
  inter_dat.r2.block_coord_offset =
      pose_stack_block_coord_offset[pose_ind][block_ind2];
  inter_dat.pair_data.max_important_bond_separation =
      max_important_bond_separation;
  inter_dat.pair_data.min_separation =
      pose_stack_min_bond_separation[pose_ind][block_ind1][block_ind2];
  inter_dat.pair_data.in_count_pair_striking_dist =
      inter_dat.pair_data.min_separation <= max_important_bond_separation;
  inter_dat.r1.n_atoms = n_atoms1;
  inter_dat.r2.n_atoms = n_atoms2;
  inter_dat.r1.n_conn = block_type_n_interblock_bonds[block_type1];
  inter_dat.r2.n_conn = block_type_n_interblock_bonds[block_type2];

  // set the pointers in inter_dat to point at the shared-memory arrays
  inter_dat.r1.coords = shared_m.coords1;
  inter_dat.r2.coords = shared_m.coords2;
  inter_dat.r1.donH_tile_inds = shared_m.don_inds1;
  inter_dat.r2.donH_tile_inds = shared_m.don_inds2;
  inter_dat.r1.acc_tile_inds = shared_m.acc_inds1;
  inter_dat.r2.acc_tile_inds = shared_m.acc_inds2;
  inter_dat.r1.donH_type = shared_m.don_type1;
  inter_dat.r2.donH_type = shared_m.don_type2;
  inter_dat.r1.acc_type = shared_m.acc_type1;
  inter_dat.r2.acc_type = shared_m.acc_type2;
  inter_dat.r1.acc_hybridization = shared_m.acc_hybridization1;
  inter_dat.r2.acc_hybridization = shared_m.acc_hybridization2;

  inter_dat.r1.path_dist = shared_m.path_dist1;
  inter_dat.r2.path_dist = shared_m.path_dist2;
  inter_dat.pair_data.conn_seps = shared_m.conn_seps;

  // Count pair setup that does not depend on which tile we are
  // operating on
  if (inter_dat.pair_data.in_count_pair_striking_dist) {
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
    DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(
        load_count_pair_conn_at_data);
  }

  // Final data members
  inter_dat.pair_data.global_params = global_params[0];
  inter_dat.pair_data.total_hbond = 0;

  // Keep a "copy" of the tensors needed during score evaluation
  // nvcc is smart enough not to duplicate the registers used here
  inter_dat.pair_data.coords = coords;
  inter_dat.pair_data.pose_stack_block_coord_offset =
      pose_stack_block_coord_offset;
  inter_dat.pair_data.pose_stack_block_type = pose_stack_block_type;
  inter_dat.pair_data.pose_stack_inter_residue_connections =
      pose_stack_inter_residue_connections;
  inter_dat.pair_data.block_type_n_all_bonds = block_type_n_all_bonds;
  inter_dat.pair_data.block_type_all_bonds = block_type_all_bonds;
  inter_dat.pair_data.block_type_atom_all_bond_ranges =
      block_type_atom_all_bond_ranges;
  inter_dat.pair_data.block_type_atoms_forming_chemical_bonds =
      block_type_atoms_forming_chemical_bonds;
  inter_dat.pair_data.block_type_atom_is_hydrogen = block_type_atom_is_hydrogen;
  inter_dat.pair_data.pair_params = pair_params;
  inter_dat.pair_data.pair_polynomials = pair_polynomials;
}

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    int TILE_SIZE,
    int MAX_N_CONN,
    typename Real,
    typename Int>
void TMOL_DEVICE_FUNC hbond_load_interres1_tile_data_to_shared(
    TView<Vec<Real, 3>, 2, Dev> coords,
    TView<Int, 2, Dev> block_type_tile_n_donH,
    TView<Int, 2, Dev> block_type_tile_n_acc,
    TView<Int, 3, Dev> block_type_tile_donH_inds,
    TView<Int, 3, Dev> block_type_tile_acc_inds,
    TView<Int, 3, Dev> block_type_tile_donor_type,
    TView<Int, 3, Dev> block_type_tile_acceptor_type,
    TView<Int, 3, Dev> block_type_tile_hybridization,
    TView<Int, 3, Dev> block_type_path_distance,
    int tile_ind,
    int start_atom1,
    int n_atoms_to_load1,
    HBondScoringData<Real> &inter_dat,
    HBondBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN> &shared_m) {
  auto store_n_don_n_acc1 = ([&](int tid) {
    int n_donH = block_type_tile_n_donH[inter_dat.r1.block_type][tile_ind];
    int n_acc = block_type_tile_n_acc[inter_dat.r1.block_type][tile_ind];
    inter_dat.r1.n_donH = n_donH;
    inter_dat.r1.n_acc = n_acc;
    if (tid == 0) {
      shared_m.n_donH1 = n_donH;
      shared_m.n_acc1 = n_acc;
    }
  });
  DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(store_n_don_n_acc1);

  hbond_load_block_into_shared<DeviceDispatch, Dev, nt, TILE_SIZE>(
      coords,
      block_type_tile_donH_inds,
      block_type_tile_acc_inds,
      block_type_tile_donor_type,
      block_type_tile_acceptor_type,
      block_type_tile_hybridization,
      block_type_path_distance,
      inter_dat.pair_dat.pose_ind,
      tile_ind,
      inter_dat.r1,
      n_atoms_to_load1,
      start_atom1,
      inter_dat.pair_dat.in_count_pair_striking_dist,
      shared_m.conn_ats1);
}

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    int TILE_SIZE,
    int MAX_N_CONN,
    typename Real,
    typename Int>
void TMOL_DEVICE_FUNC hbond_load_interres2_tile_data_to_shared(
    TView<Vec<Real, 3>, 2, Dev> coords,
    TView<Int, 2, Dev> block_type_tile_n_donH,
    TView<Int, 2, Dev> block_type_tile_n_acc,
    TView<Int, 3, Dev> block_type_tile_donH_inds,
    TView<Int, 3, Dev> block_type_tile_acc_inds,
    TView<Int, 3, Dev> block_type_tile_donor_type,
    TView<Int, 3, Dev> block_type_tile_acceptor_type,
    TView<Int, 3, Dev> block_type_tile_hybridization,
    TView<Int, 3, Dev> block_type_path_distance,
    int tile_ind,
    int start_atom2,
    int n_atoms_to_load2,
    HBondScoringData<Real> &inter_dat,
    HBondBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN> &shared_m) {
  auto store_n_don_n_acc2 = ([&](int tid) {
    int n_donH = block_type_tile_n_donH[inter_dat.r2.block_type][tile_ind];
    int n_acc = block_type_tile_n_acc[inter_dat.r2.block_type][tile_ind];
    inter_dat.r2.n_donH = n_donH;
    inter_dat.r2.n_acc = n_acc;
    if (tid == 0) {
      shared_m.n_donH2 = n_donH;
      shared_m.n_acc2 = n_acc;
    }
  });
  DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(store_n_don_n_acc2);
  hbond_load_block_into_shared<DeviceDispatch, Dev, nt, TILE_SIZE>(
      coords,
      block_type_tile_donH_inds,
      block_type_tile_acc_inds,
      block_type_tile_donor_type,
      block_type_tile_acceptor_type,
      block_type_tile_hybridization,
      block_type_path_distance,
      inter_dat.pair_dat.pose_ind,
      inter_dat.r2,
      n_atoms_to_load2,
      start_atom2,
      inter_dat.pair_dat.in_count_pair_striking_dist,
      shared_m.conn_ats2);
}

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    typename Int,
    typename Real,
    int TILE_SIZE,
    int MAX_N_CONN>
void TMOL_DEVICE_FUNC hbond_load_tile_invariant_intrares_data(
    TView<Vec<Real, 3>, 2, Dev> coords,
    TView<Int, 2, Dev> pose_stack_block_coord_offset,
    TView<Int, 2, Dev> pose_stack_block_type,
    TView<Vec<Int, 2>, 3, Dev> pose_stack_inter_residue_connections,
    TView<Int, 1, Dev> block_type_n_all_bonds,
    TView<Int, 3, Dev> block_type_all_bonds,
    TView<Int, 2, Dev> block_type_atom_all_bond_ranges,
    TView<Int, 2, Dev> block_type_atoms_forming_chemical_bonds,
    TView<Int, 2, Dev> block_type_atom_is_hydrogen,
    TView<HBondPairParams<Real>, 2, Dev> pair_params,
    TView<HBondPolynomials<double>, 2, Dev> pair_polynomials,
    TView<HBondGlobalParams<Real>, 1, Dev> global_params,
    int const max_important_bond_separation,
    int pose_ind,
    int block_ind1,
    int block_type1,
    int n_atoms1,
    HBondScoringData<Real> &intra_dat,
    HBondBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN> &shared_m) {
  intra_dat.pair_data.pose_ind = pose_ind;
  intra_dat.r1.block_type = block_type1;
  intra_dat.r2.block_type = block_type1;
  intra_dat.r1.block_coord_offset =
      pose_stack_block_coord_offset[pose_ind][block_ind1];
  intra_dat.r2.block_coord_offset = intra_dat.r1.block_coord_offset;
  intra_dat.pair_data.max_important_bond_separation =
      max_important_bond_separation;

  // we are not going to load count pair data into shared memory because
  // we are not going to use that data from shared memory
  intra_dat.pair_data.min_separation = 0;
  intra_dat.pair_data.in_count_pair_striking_dist = false;

  intra_dat.r1.n_atoms = n_atoms1;
  intra_dat.r2.n_atoms = n_atoms1;
  intra_dat.r1.n_conn = 0;
  intra_dat.r2.n_conn = 0;

  // set the pointers in intra_dat to point at the
  // shared-memory arrays. Note that these arrays will be reset
  // later because which shared memory arrays we will use depends on
  // which tile pair we are evaluating!
  intra_dat.r1.coords = shared_m.coords1;
  intra_dat.r2.coords = shared_m.coords2;
  intra_dat.r1.donH_tile_inds = shared_m.don_inds1;
  intra_dat.r2.donH_tile_inds = shared_m.don_inds2;
  intra_dat.r1.acc_tile_inds = shared_m.acc_inds1;
  intra_dat.r2.acc_tile_inds = shared_m.acc_inds2;
  intra_dat.r1.donH_type = shared_m.don_type1;
  intra_dat.r2.donH_type = shared_m.don_type2;
  intra_dat.r1.acc_type = shared_m.acc_type1;
  intra_dat.r2.acc_type = shared_m.acc_type2;
  intra_dat.r1.acc_hybridization = shared_m.acc_hybridization1;
  intra_dat.r2.acc_hybridization = shared_m.acc_hybridization2;

  // these count pair arrays are not going to be used
  intra_dat.r1.path_dist = 0;
  intra_dat.r2.path_dist = 0;
  intra_dat.pair_data.conn_seps = 0;

  // Final data members
  intra_dat.pair_data.global_params = global_params[0];
  intra_dat.pair_data.total_hbond = 0;

  // Keep a "copy" of the tensors needed during score evaluation
  // nvcc is smart enough not to duplicate the registers used here
  intra_dat.pair_data.coords = coords;
  intra_dat.pair_data.pose_stack_block_coord_offset =
      pose_stack_block_coord_offset;
  intra_dat.pair_data.pose_stack_block_type = pose_stack_block_type;
  intra_dat.pair_data.pose_stack_inter_residue_connections =
      pose_stack_inter_residue_connections;
  intra_dat.pair_data.block_type_n_all_bonds = block_type_n_all_bonds;
  intra_dat.pair_data.block_type_all_bonds = block_type_all_bonds;
  intra_dat.pair_data.block_type_atom_all_bond_ranges =
      block_type_atom_all_bond_ranges;
  intra_dat.pair_data.block_type_atoms_forming_chemical_bonds =
      block_type_atoms_forming_chemical_bonds;
  intra_dat.pair_data.block_type_atom_is_hydrogen = block_type_atom_is_hydrogen;
  intra_dat.pair_data.pair_params = pair_params;
  intra_dat.pair_data.pair_polynomials = pair_polynomials;
}

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    int TILE_SIZE,
    int MAX_N_CONN,
    typename Real,
    typename Int>
void TMOL_DEVICE_FUNC hbond_load_intrares1_tile_data_to_shared(
    TView<Vec<Real, 3>, 2, Dev> coords,
    TView<Int, 2, Dev> block_type_tile_n_donH_inds,
    TView<Int, 2, Dev> block_type_tile_n_acc_inds,
    TView<Int, 2, Dev> block_type_tile_donH_inds,
    TView<Int, 2, Dev> block_type_tile_acc_inds,
    TView<Int, 2, Dev> block_type_donor_type,
    TView<Int, 2, Dev> block_type_acceptor_type,
    TView<Int, 2, Dev> block_type_hybridization,
    int tile_ind,
    int start_atom1,
    int n_atoms_to_load1,
    HBondScoringData<Real> &intra_dat,
    HBondBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN> &shared_m) {
  hbond_load_block_coords_and_params_into_shared<DeviceDispatch, Dev, nt>(
      coords,
      block_type_tile_n_donH_inds,
      block_type_tile_n_acc_inds,
      block_type_tile_donH_inds,
      block_type_tile_acc_inds,
      block_type_donor_type,
      block_type_acceptor_type,
      block_type_hybridization,
      intra_dat.pair_data.pose_ind,
      intra_dat.r1,
      n_atoms_to_load1,
      start_atom1);
}

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    int TILE_SIZE,
    int MAX_N_CONN,
    typename Real,
    typename Int>
void TMOL_DEVICE_FUNC hbond_load_intrares2_tile_data_to_shared(
    TView<Vec<Real, 3>, 2, Dev> coords,
    TView<Int, 2, Dev> block_type_tile_n_donH_inds,
    TView<Int, 2, Dev> block_type_tile_n_acc_inds,
    TView<Int, 2, Dev> block_type_tile_donH_inds,
    TView<Int, 2, Dev> block_type_tile_acc_inds,
    TView<Int, 2, Dev> block_type_donor_type,
    TView<Int, 2, Dev> block_type_acceptor_type,
    TView<Int, 2, Dev> block_type_hybridization,
    TView<Int, 3, Dev> block_type_path_distance,
    int tile_ind,
    int start_atom2,
    int n_atoms_to_load2,
    HBondScoringData<Real> &intra_dat,
    HBondBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN> &shared_m) {
  hbond_load_block_coords_and_params_into_shared<DeviceDispatch, Dev, nt>(
      coords,
      block_type_tile_n_donH_inds,
      block_type_tile_n_acc_inds,
      block_type_tile_donH_inds,
      block_type_tile_acc_inds,
      block_type_donor_type,
      block_type_acceptor_type,
      block_type_hybridization,
      intra_dat.pair_data.pose_ind,
      intra_dat.r2,
      n_atoms_to_load2,
      start_atom2);
}

template <int TILE_SIZE, int MAX_N_CONN, typename Real>
void TMOL_DEVICE_FUNC hbond_load_intrares_data_from_shared(
    int tile_ind1,
    int tile_ind2,
    HBondBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN> &shared_m,
    HBondScoringData<Real> &intra_dat) {
  // set the pointers in intra_dat to point at the shared-memory arrays
  // If we are evaluating the energies between atoms in the same tile
  // then only the "1" shared-memory arrays will be loaded with data;
  // we will point the "2" memory pointers at the "1" arrays
  bool same_tile = tile_ind1 == tile_ind2;
  intra_dat.r1.coords = shared_m.coords1;
  intra_dat.r2.coords = (same_tile ? shared_m.coords1 : shared_m.coords2);
  intra_dat.r1.donH_tile_inds = shared_m.don_inds1;
  intra_dat.r1.acc_tile_inds = shared_m.acc_inds;
  intra_dat.r2.donH_tile_inds =
      (same_tile ? shared_m.don_inds1 : shared_m.don_inds2);
  intra_dat.r2.acc_tile_inds =
      (same_tile ? shared_m.acc_inds1 : shared_m.acc_inds2);
  intra_dat.r1.donH_type = shared_m.don_type1;
  intra_dat.r1.acc_type = shared_m.acc_type1;
  intra_dat.r2.donH_type =
      (same_tile ? shared_m.don_type1 : shared_m.don_type2);
  intra_dat.r2.acc_type = (same_tile ? shared_m.acc_type1 : shared_m.acc_type2);
  intra_dat.r1.acc_hybridization = shared_m.acc_hybridization1;
  intra_dat.r2.acc_hybridization =
      (same_tile ? shared_m.acc_hybridization1 : shared_m.acc_hybridization2);
}

// Some coordinates are available in shared memory, some we will
// have to go out to global memory for.
template <int TILE_SIZE, typename Real, typename Int, tmol::Device Dev>
Eigen::Matrix<Real, 3, 1> load_coord(
    BlockCentricAtom<Int> bcat,
    HBondSingleResData<Real> const &single_res_dat,
    HBondResPairData<Dev, Real, Int> const &respair_dat,
    int tile_start) {
  Eigen::Matrix<Real, 3, 1> xyz{Real(0), Real(0), Real(0)};
  if (bcat.atom != -1) {
    bool in_smem = false;
    if (bcat.block == single_res_dat.block_ind) {
      int bcat_tile_ind = bcat.atom - tile_start_start;
      if (bcat_tile_ind >= 0 && bcat_tile_ind < TILE_SIZE) {
        in_smem = true;
        xyz = coord_from_shared(single_res_dat.coords, bcat_tile_ind);
      }
    }
    if (!in_smem) {
      // outside of tile or on other res, retrieve from global coords
      int coord_offset =
          (bcat.block == single_res_dat.block_ind
               ? single_res_dat.block_coord_offset
               : respair_dat.pose_stack_block_coord_offset[respair_dat.pose_ind]
                                                          [bcat.block]);
      xyz = respair_dat
                .coords[respair_dat.pose_ind][don_bases.D.atom + coord_offset];
    }
  }
  return xyz;
}

template <int TILE_SIZE, typename Real, typename Int, tmol::Device Dev>
TMOL_DEVICE_FUNC Real hbond_atom_energy_full(
    int donH_ind,             // in [0:n_donH)
    int acc_ind,              // in [0:n_acc)
    int don_h_atom_tile_ind,  // in [0:TILE_SIZE)
    int acc_atom_tile_ind,    // in [0:TILE_SIZE)
    int don_start,
    int acc_start,
    HBondSingleResData<Real> const &don_dat,
    HBondSingleResData<Real> const &acc_dat,
    HBondResPairData<Dev, Real, Int> const &respair_dat,
    int cp_separation) {
  using Real3 = Eigen::Matrix<Real, 3, 1>;

  Real3 Hxyz = coord_from_shared(don_dat.coords, don_h_atom_tile_ind);
  Real3 Axyz = coord_from_shared(acc_dat.coords, acc_atom_tile_ind);

  Real const dist = distance<Real>::V(Hxyz, Axyz);
  if (dist < respair_dat.global_params.max_ha_dis) {
    BlockCentricAtom<Int> H{
        don_dat.block_ind,
        don_dat.block_type,
        don_start + don_h_atom_tile_ind,
    };
    BlockCentricAtom A{
        acc_dat.block_ind, acc_dat.block_type, acc_start + acc_atom_tile_ind};
    BlockCentricIndexedBonds<Int, Dev> bonds{
        respair_dat.pose_stack_inter_residue_connections[respair_dat.pose_ind],
        respair_dat.pose_stack_block_type[respair_dat.pose_ind],
        respair_dat.block_type_n_all_bonds,
        respair_dat.block_type_all_bonds,
        respair_dat.block_type_atom_all_bond_ranges,
        respair_dat.block_type_atoms_forming_chemical_bonds};
    auto acc_bases = BlockCentricAcceptorBases<Int>::for_acceptor(
        A,
        acc_dat.acc_hybridization[acc_ind],
        bonds,
        respair_dat.block_type_atom_is_hydrogen);
    auto don_bases = BlockCentricDonorBase<Int>::for_polar_H(H, bonds);

    Real3 Dxyz =
        load_coord<TILE_SIZE>(don_bases.D, don_dat, respair_dat, don_start);
    Real3 Bxyz =
        load_coord<TILE_SIZE>(acc_bases.B, acc_dat, respair_dat, acc_start);
    Real3 B0xyz =
        load_coord<TILE_SIZE>(acc_bases.B0, acc_dat, respair_dat, acc_start);

    unsigned char dt = don_dat.don_type[donH_ind];
    unsigned char at = acc_dat.acc_type[acc_ind];

    return hbond_score<Real, Int>::V(
        Dxyz,
        Hxyz,
        Axyz,
        Bxyz,
        B0xyz,
        respair_dat.pair_params[dt][at],
        respair_dat.pair_polynomials[dt][at],
        respair_dat.global_params);
  } else {
    return 0;
  }
}

template <typename Real, tmol::Device Dev>
TMOL_DEVICE_FUNC Real hbond_atom_energy_and_derivs_full(
    int donH_ind,             // in [0:n_donH)
    int acc_ind,              // in [0:n_acc)
    int don_h_atom_tile_ind,  // in [0:TILE_SIZE)
    int acc_atom_tile_ind,    // in [0:TILE_SIZE)
    int don_start,
    int acc_start,
    HBondSingleResData<Real> const &don_dat,
    HBondSingleResData<Real> const &acc_dat,
    HBondResPairData<Dev, Real, Int> const &respair_dat,
    int cp_separation,
    TView<Eigen::Matrix<Real, 3, 1>, 3, Dev> dV_dcoords) {
  using Real3 = Eigen::Matrix<Real, 3, 1>;

  Real3 Hxyz = coord_from_shared(don_dat.coords, don_h_atom_tile_ind);
  Real3 Axyz = coord_from_shared(acc_dat.coords, acc_atom_tile_ind);

  auto const dist_r = distance<Real>::V(Hxyz, Axyz);
  if (dist_r.V < respair_dat.global_params.max_ha_dis) {
    BlockCentricAtom<Int> H{
        don_dat.block_ind,
        don_dat.block_type,
        don_start + don_h_atom_tile_ind,
    };
    BlockCentricAtom A{
        acc_dat.block_ind, acc_dat.block_type, acc_start + acc_atom_tile_ind};
    BlockCentricIndexedBonds<Int, Dev> bonds{
        respair_dat.pose_stack_inter_residue_connections[respair_dat.pose_ind],
        respair_dat.pose_stack_block_type[respair_dat.pose_ind],
        respair_dat.block_type_n_all_bonds,
        respair_dat.block_type_all_bonds,
        respair_dat.block_type_atom_all_bond_ranges,
        respair_dat.block_type_atoms_forming_chemical_bonds};
    auto acc_bases = BlockCentricAcceptorBases<Int>::for_acceptor(
        A,
        acc_dat.acc_hybridization[acc_ind],
        bonds,
        respair_dat.block_type_atom_is_hydrogen);
    auto don_bases = BlockCentricDonorBase<Int>::for_polar_H(H, bonds);

    Real3 Dxyz =
        load_coord<TILE_SIZE>(don_bases.D, don_dat, respair_dat, don_start);
    Real3 Bxyz =
        load_coord<TILE_SIZE>(acc_bases.B, acc_dat, respair_dat, acc_start);
    Real3 B0xyz =
        load_coord<TILE_SIZE>(acc_bases.B0, acc_dat, respair_dat, acc_start);

    unsigned char dt = don_dat.don_type[donH_ind];
    unsigned char at = acc_dat.acc_type[acc_ind];

    auto hbond_V_dV = hbond_score<Real, Int>::V(
        Dxyz,
        Hxyz,
        Axyz,
        Bxyz,
        B0xyz,
        respair_dat.pair_params[dt][at],
        respair_dat.pair_polynomials[dt][at],
        respair_dat.global_params);

    // accumulate don D atom derivatives to global memory
    for (int j = 0; j < 3; ++j) {
      if (hbond_V_dV.dV_dD[j] != 0) {
        accumulate<Dev, Real>::add(
            dV_dcoords[0][respair_dat.pose_ind]
                      [don_dat.block_coord_offset + D.atom][j],
            hbond_V_dV.dV_dD[j]);
      }
    }
    // accumulate don H atom derivatives to global memory
    for (int j = 0; j < 3; ++j) {
      if (hbond_V_dV.dV_dH[j] != 0) {
        accumulate<Dev, Real>::add(
            dV_dcoords[0][respair_dat.pose_ind]
                      [don_dat.block_coord_offset + H.atom][j],
            hbond_V_dV.dV_dH[j]);
      }
    }

    // accumulate acc A atom derivatives to global memory
    for (int j = 0; j < 3; ++j) {
      if (hbond_V_dV.dV_dA[j] != 0) {
        accumulate<Dev, Real>::add(
            dV_dcoords[0][respair_dat.pose_ind]
                      [acc_dat.block_coord_offset + A.atom][j],
            hbond_V_dV.dV_dA[j]);
      }
    }
    // accumulate acc B atom derivatives to global memory
    for (int j = 0; j < 3; ++j) {
      if (hbond_V_dV.dV_dB[j] != 0 && B.atom >= 0) {
        accumulate<Dev, Real>::add(
            dV_dcoords[0][respair_dat.pose_ind]
                      [acc_dat.block_coord_offset + B.atom][j],
            hbond_V_dV.dV_dB[j]);
      }
    }
    // accumulate acc B0 atom derivatives to global memory
    for (int j = 0; j < 3; ++j) {
      if (hbond_V_dV.dV_dB0[j] != 0 && B0.atom >= 0) {
        accumulate<Dev, Real>::add(
            dV_dcoords[0][respair_dat.pose_ind]
                      [acc_dat.block_coord_offset + B0.atom][j],
            hbond_V_dV.dV_dB0[j]);
      }
    }

  } else {
    return 0;
  }

  return V;
}

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    typename Func,
    typename Real>
void TMOL_DEVICE_FUNC eval_interres_don_acc_pair_energies(
    HbondScoringData<Real> &inter_dat,
    int start_atom1,
    int start_atom2,
    Func f) {
  int const n_don_acc_pairs = inter_dat.r1.n_donH * inter_dat.r2.n_acc
                              + inter_dat.r1.n_acc * inter_dat.r2.n_donH;
  for (int i = tid; i < n_don_acc_pairs; i += nt) {
    bool r1_don = i < inter_dat.r1.n_donH * inter_dat.r2.n_acc;
    int pair_ind = r1_don ? i - inter_dat.r1.n_donH * inter_dat.r2.n_acc : i;
    HBondSingleResData const &don_dat = r1_don ? inter_dat.r1 : inter_dat.r2;
    HBondSingleResData const &acc_dat = r1_don ? inter_dat.r2 : inter_dat.r1;
    int don_ind = pair_ind / acc_dat.n_acc;
    int acc_ind = pair_ind % acc_dat.n_acc;
    int don_start = r1_don ? start_atom1 : start_atom2;
    int acc_start = r2_don ? start_atom2 : start_atom1;

    inter_dat.pair_data.total_hbond +=
        f(don_start, acc_start, don_ind, acc_ind, inter_dat, r1_don);
  }
  DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(
      eval_scores_for_don_acc_pairs);
}

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    typename Func,
    typename Real>
void TMOL_DEVICE_FUNC eval_intrares_don_acc_pair_energies(
    HbondScoringData<Real> &intra_dat,
    int start_atom1,
    int start_atom2,
    Func f) {
  auto eval_scores_for_don_acc_pairs = ([&](int tid) {
    if (start_atom1 == start_atom2) {
      int const n_don_acc_pairs = intra_dat.r1.n_donH * intra_dat.r2.n_acc;
      for (int i = tid; i < n_don_acc_pairs; i += nt) {
        int don_ind = i / acc_dat.n_acc;
        int acc_ind = i % acc_dat.n_acc;

        intra_dat.pair_data.total_hbond +=
            f(start_atom1, start_atom1, don_ind, acc_ind, intra_dat, true);
      }
    } else {
      int const n_don_acc_pairs = intra_dat.r1.n_donH * intra_dat.r2.n_acc
                                  + intra_dat.r1.n_acc * intra_dat.r2.n_donH;
      for (int i = tid; i < n_don_acc_pairs; i += nt) {
        bool r1_don = i < intra_dat.r1.n_donH * intra_dat.r2.n_acc;
        int pair_ind =
            r1_don ? i - intra_dat.r1.n_donH * intra_dat.r2.n_acc : i;
        // HBondSingleResData const & don_dat = r1_don ? intra_dat.r1 :
        // intra_dat.r2;
        HBondSingleResData const &acc_dat =
            r1_don ? intra_dat.r2 : intra_dat.r1;
        int don_ind = pair_ind / acc_dat.n_acc;
        int acc_ind = pair_ind % acc_dat.n_acc;
        int don_start = r1_don ? start_atom1 : start_atom2;
        int acc_start = r2_don ? start_atom2 : start_atom1;

        intra_dat.pair_data.total_hbond +=
            f(don_start, acc_start, don_ind, acc_ind, intra_dat, r1_don);
      }
    }
  });

  DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(
      eval_scores_for_don_acc_pairs);
}

}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol
