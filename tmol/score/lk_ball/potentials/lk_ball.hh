#pragma once

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/count_pair.hh>
#include <tmol/score/common/data_loading.hh>
#include <tmol/score/common/diamond_macros.hh>

#include "params.hh"
#include "scalar.hh"

namespace tmol {
namespace score {
namespace lk_ball {
namespace potentials {

template <typename Real>
class LKBallSingleResData {
 public:
  int rot_ind;
  int block_ind;
  int block_type;
  int rot_coord_offset;
  int n_atoms;
  int n_conn;
  Real* pose_coords;
  Real* water_coords;
  unsigned char n_polars;
  unsigned char n_occluders;
  unsigned char* pol_occ_tile_inds;
  LKBallTypeParams<Real>* lk_ball_params;
  unsigned char* path_dist;
};

template <typename Real>
class LKBallResPairData {
 public:
  int pose_ind;
  int max_important_bond_separation;
  int min_separation;
  bool in_count_pair_striking_dist;
  unsigned char* conn_seps;

  // load global params once; store totalE's
  LKBallGlobalParams<Real> global_params;
  Real total_lk_ball_iso;
  Real total_lk_ball;
  Real total_lk_bridge;
  Real total_lk_bridge_uncpl;
};

template <typename Real>
class LKBallScoringData {
 public:
  LKBallSingleResData<Real> r1;
  LKBallSingleResData<Real> r2;
  LKBallResPairData<Real> pair_data;
};

template <typename Real, int TILE_SIZE, int MAX_N_WATER, int MAX_N_CONN>
struct LKBallBlockPairSharedData {
  Real pose_coords1[TILE_SIZE * 3];  // 768 bytes for coords
  Real pose_coords2[TILE_SIZE * 3];
  Real water_coords1[TILE_SIZE * MAX_N_WATER * 3];  // 3072 bytes for coords
  Real water_coords2[TILE_SIZE * MAX_N_WATER * 3];

  unsigned char n_polars1;  // 4 bytes for counts
  unsigned char n_polars2;
  unsigned char n_occluders1;
  unsigned char n_occluders2;
  unsigned char pol_occ_tile_inds1[TILE_SIZE];  // 64 bytes for indices
  unsigned char pol_occ_tile_inds2[TILE_SIZE];
  LKBallTypeParams<Real> lk_ball_params1[TILE_SIZE];  // 192 bytes for params
  LKBallTypeParams<Real> lk_ball_params2[TILE_SIZE];

  unsigned char conn_ats1[MAX_N_CONN];  // 12 bytes
  unsigned char conn_ats2[MAX_N_CONN];
  unsigned char path_dist1[MAX_N_CONN * TILE_SIZE];  // 384 bytes
  unsigned char path_dist2[MAX_N_CONN * TILE_SIZE];
  unsigned char conn_seps[MAX_N_CONN * MAX_N_CONN];  // 144 bytes
};

#undef def

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    int MAX_N_WATER,
    typename Real,
    typename Int>
void TMOL_DEVICE_FUNC lk_ball_load_block_coords_and_params_into_shared(
    TView<Vec<Real, 3>, 1, Dev> pose_coords,
    TView<Vec<Real, 3>, 2, Dev> water_coords,
    TView<Int, 3, Dev> block_type_tile_pol_occ_inds,
    TView<LKBallTypeParams<Real>, 3, Dev> block_type_tile_lk_ball_params,
    int pose_ind,
    int tile_ind,
    LKBallSingleResData<Real>& r_dat,
    int n_atoms_to_load,
    int start_atom) {
  // pre-condition: n_atoms_to_load < TILE_SIZE
  // Note that TILE_SIZE is not explicitly passed in, but is "present"
  // in r_dat.coords allocation

  if constexpr (Dev == tmol::Device::CPU) {
    // CPU has no workgroup-shared memory to populate. Read immutable,
    // contiguous coordinates and parameters in place instead of copying each
    // rotamer pair through the CUDA-oriented staging arrays.
    r_dat.pose_coords = reinterpret_cast<Real*>(
        &pose_coords[r_dat.rot_coord_offset + start_atom]);
    r_dat.water_coords = reinterpret_cast<Real*>(
        &water_coords[r_dat.rot_coord_offset + start_atom][0]);
    r_dat.lk_ball_params =
        &block_type_tile_lk_ball_params[r_dat.block_type][tile_ind][0];
  } else {
    DeviceDispatch<Dev>::template copy_contiguous_data<nt, 3>(
        r_dat.pose_coords,
        reinterpret_cast<Real*>(
            &pose_coords[r_dat.rot_coord_offset + start_atom]),
        n_atoms_to_load * 3);
    DeviceDispatch<Dev>::template copy_contiguous_data<nt, MAX_N_WATER * 3>(
        r_dat.water_coords,
        reinterpret_cast<Real*>(
            &water_coords[r_dat.rot_coord_offset + start_atom][0]),
        n_atoms_to_load * MAX_N_WATER * 3);
    int const N_PARAMS = sizeof(LKBallTypeParams<Real>) / sizeof(Real);
    DeviceDispatch<Dev>::template copy_contiguous_data<nt, N_PARAMS>(
        reinterpret_cast<Real*>(r_dat.lk_ball_params),
        reinterpret_cast<Real*>(
            &block_type_tile_lk_ball_params[r_dat.block_type][tile_ind][0]),
        r_dat.n_occluders * N_PARAMS);
  }
  DeviceDispatch<Dev>::template copy_contiguous_data_and_cast<nt, 1>(
      r_dat.pol_occ_tile_inds,
      &block_type_tile_pol_occ_inds[r_dat.block_type][tile_ind][0],
      r_dat.n_occluders);
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    int TILE_SIZE,
    int MAX_N_WATER,
    typename Real,
    typename Int>
void TMOL_DEVICE_FUNC lk_ball_load_block_into_shared(
    TView<Vec<Real, 3>, 1, Dev> pose_coords,
    TView<Vec<Real, 3>, 2, Dev> water_coords,
    TView<Int, 3, Dev> block_type_tile_pol_occ_inds,
    TView<LKBallTypeParams<Real>, 3, Dev> block_type_tile_lk_ball_params,
    TView<Int, 3, Dev> block_type_path_distance,
    int pose_ind,
    int tile_ind,
    LKBallSingleResData<Real>& r_dat,
    int n_atoms_to_load,
    int start_atom,
    bool count_pair_striking_dist,
    unsigned char* __restrict__ conn_ats) {
  lk_ball_load_block_coords_and_params_into_shared<
      DeviceDispatch,
      Dev,
      nt,
      MAX_N_WATER>(
      pose_coords,
      water_coords,
      block_type_tile_pol_occ_inds,
      block_type_tile_lk_ball_params,
      pose_ind,
      tile_ind,
      r_dat,
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
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    typename Int,
    typename Real,
    int TILE_SIZE,
    int MAX_N_WATER,
    int MAX_N_CONN>
void TMOL_DEVICE_FUNC lk_ball_load_tile_invariant_interres_data(
    TView<Int, 1, Dev> rot_coord_offset,
    TView<Int, 1, Dev> block_type_ind_for_rot,
    TView<Vec<Int, 2>, 3, Dev> pose_stack_inter_residue_connections,
    TView<Int, 4, Dev> pose_stack_near_blocks,
    TView<int8_t, 5, Dev> pose_stack_inter_block_bondsep,
    TView<Int, 1, Dev> block_type_n_interblock_bonds,
    TView<Int, 2, Dev> block_type_atoms_forming_chemical_bonds,
    TView<LKBallGlobalParams<Real>, 1, Dev> global_params,
    int const max_important_bond_separation,
    int pose_ind,
    int rot_ind1,
    int rot_ind2,
    int block_ind1,
    int block_ind2,
    int block_type1,
    int block_type2,
    int n_atoms1,
    int n_atoms2,
    LKBallScoringData<Real>& inter_dat,
    LKBallBlockPairSharedData<Real, TILE_SIZE, MAX_N_WATER, MAX_N_CONN>&
        shared_m) {
  inter_dat.pair_data.pose_ind = pose_ind;
  inter_dat.r1.rot_ind = rot_ind1;
  inter_dat.r2.rot_ind = rot_ind2;
  inter_dat.r1.block_ind = block_ind1;
  inter_dat.r2.block_ind = block_ind2;
  inter_dat.r1.block_type = block_type1;
  inter_dat.r2.block_type = block_type2;
  inter_dat.r1.rot_coord_offset = rot_coord_offset[rot_ind1];
  inter_dat.r2.rot_coord_offset = rot_coord_offset[rot_ind2];
  inter_dat.pair_data.max_important_bond_separation =
      max_important_bond_separation;
  int const near_slot = common::count_pair::near_block_slot(
      pose_stack_near_blocks[pose_ind][block_ind1], block_ind2);
  inter_dat.pair_data.min_separation =
      pose_stack_near_blocks[pose_ind][block_ind1][near_slot][1];
  inter_dat.pair_data.in_count_pair_striking_dist =
      inter_dat.pair_data.min_separation <= max_important_bond_separation;
  inter_dat.r1.n_atoms = n_atoms1;
  inter_dat.r2.n_atoms = n_atoms2;
  inter_dat.r1.n_conn = block_type_n_interblock_bonds[block_type1];
  inter_dat.r2.n_conn = block_type_n_interblock_bonds[block_type2];

  // set the pointers in inter_dat to point at the shared-memory arrays
  inter_dat.r1.pose_coords = shared_m.pose_coords1;
  inter_dat.r2.pose_coords = shared_m.pose_coords2;
  inter_dat.r1.water_coords = shared_m.water_coords1;
  inter_dat.r2.water_coords = shared_m.water_coords2;
  inter_dat.r1.pol_occ_tile_inds = shared_m.pol_occ_tile_inds1;
  inter_dat.r2.pol_occ_tile_inds = shared_m.pol_occ_tile_inds2;
  inter_dat.r1.lk_ball_params = shared_m.lk_ball_params1;
  inter_dat.r2.lk_ball_params = shared_m.lk_ball_params2;

  inter_dat.r1.path_dist = shared_m.path_dist1;
  inter_dat.r2.path_dist = shared_m.path_dist2;
  inter_dat.pair_data.conn_seps = shared_m.conn_seps;

  // Count pair setup that does not depend on which tile we are
  // operating on; only necessary if r1 and r2 are within
  // a minimum number of chemical bonds separation
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
              pose_stack_inter_block_bondsep[pose_ind][block_ind1][near_slot]
                                            [conn1][conn2];
        }
      }
    });
    DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(
        load_count_pair_conn_at_data);
  }

  // Final data members
  inter_dat.pair_data.global_params = global_params[0];

  // Set initial energy totals to 0
  inter_dat.pair_data.total_lk_ball_iso = 0;
  inter_dat.pair_data.total_lk_ball = 0;
  inter_dat.pair_data.total_lk_bridge = 0;
  inter_dat.pair_data.total_lk_bridge_uncpl = 0;
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    int TILE_SIZE,
    int MAX_N_WATER,
    int MAX_N_CONN,
    typename Real,
    typename Int>
void TMOL_DEVICE_FUNC lk_ball_load_interres1_tile_data_to_shared(
    TView<Vec<Real, 3>, 1, Dev> pose_coords,
    TView<Vec<Real, 3>, 2, Dev> water_coords,
    TView<Int, 2, Dev> block_type_tile_n_polar_atoms,
    TView<Int, 2, Dev> block_type_tile_n_occluder_atoms,
    TView<Int, 3, Dev> block_type_tile_pol_occ_inds,
    TView<LKBallTypeParams<Real>, 3, Dev> block_type_tile_lk_ball_params,
    TView<Int, 3, Dev> block_type_path_distance,
    int tile_ind,
    int start_atom1,
    int n_atoms_to_load1,
    LKBallScoringData<Real>& inter_dat,
    LKBallBlockPairSharedData<Real, TILE_SIZE, MAX_N_WATER, MAX_N_CONN>&
        shared_m) {
  auto store_n_pol_n_occ1 = ([&](int tid) {
    int n_pol =
        block_type_tile_n_polar_atoms[inter_dat.r1.block_type][tile_ind];
    int n_occ =
        block_type_tile_n_occluder_atoms[inter_dat.r1.block_type][tile_ind];
    inter_dat.r1.n_polars = n_pol;
    inter_dat.r1.n_occluders = n_occ;
    if (tid == 0) {
      shared_m.n_polars1 = n_pol;
      shared_m.n_occluders1 = n_occ;
    }
  });
  DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(store_n_pol_n_occ1);

  lk_ball_load_block_into_shared<
      DeviceDispatch,
      Dev,
      nt,
      TILE_SIZE,
      MAX_N_WATER>(
      pose_coords,
      water_coords,
      block_type_tile_pol_occ_inds,
      block_type_tile_lk_ball_params,
      block_type_path_distance,
      inter_dat.pair_data.pose_ind,
      tile_ind,
      inter_dat.r1,
      n_atoms_to_load1,
      start_atom1,
      inter_dat.pair_data.in_count_pair_striking_dist,
      shared_m.conn_ats1);
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    int TILE_SIZE,
    int MAX_N_WATER,
    int MAX_N_CONN,
    typename Real,
    typename Int>
void TMOL_DEVICE_FUNC lk_ball_load_interres2_tile_data_to_shared(
    TView<Vec<Real, 3>, 1, Dev> pose_coords,
    TView<Vec<Real, 3>, 2, Dev> water_coords,
    TView<Int, 2, Dev> block_type_tile_n_polar_atoms,
    TView<Int, 2, Dev> block_type_tile_n_occluder_atoms,
    TView<Int, 3, Dev> block_type_tile_pol_occ_inds,
    TView<LKBallTypeParams<Real>, 3, Dev> block_type_tile_lk_ball_params,
    TView<Int, 3, Dev> block_type_path_distance,
    int tile_ind,
    int start_atom2,
    int n_atoms_to_load2,
    LKBallScoringData<Real>& inter_dat,
    LKBallBlockPairSharedData<Real, TILE_SIZE, MAX_N_WATER, MAX_N_CONN>&
        shared_m) {
  auto store_n_pol_n_occ2 = ([&](int tid) {
    int n_pol =
        block_type_tile_n_polar_atoms[inter_dat.r2.block_type][tile_ind];
    int n_occ =
        block_type_tile_n_occluder_atoms[inter_dat.r2.block_type][tile_ind];
    inter_dat.r2.n_polars = n_pol;
    inter_dat.r2.n_occluders = n_occ;
    if (tid == 0) {
      shared_m.n_polars2 = n_pol;
      shared_m.n_occluders2 = n_occ;
    }
  });
  DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(store_n_pol_n_occ2);

  lk_ball_load_block_into_shared<
      DeviceDispatch,
      Dev,
      nt,
      TILE_SIZE,
      MAX_N_WATER>(
      pose_coords,
      water_coords,
      block_type_tile_pol_occ_inds,
      block_type_tile_lk_ball_params,
      block_type_path_distance,
      inter_dat.pair_data.pose_ind,
      tile_ind,
      inter_dat.r2,
      n_atoms_to_load2,
      start_atom2,
      inter_dat.pair_data.in_count_pair_striking_dist,
      shared_m.conn_ats2);
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    typename Int,
    typename Real,
    int TILE_SIZE,
    int MAX_N_WATER,
    int MAX_N_CONN>
void TMOL_DEVICE_FUNC lk_ball_load_tile_invariant_intrares_data(
    TView<Int, 1, Dev> rot_coord_offset,
    TView<Int, 1, Dev> block_type_ind_for_rot,
    TView<LKBallGlobalParams<Real>, 1, Dev> global_params,
    int const max_important_bond_separation,
    int pose_ind,
    int rot_ind1,
    int block_ind1,
    int block_type1,
    int n_atoms1,
    LKBallScoringData<Real>& intra_dat,
    LKBallBlockPairSharedData<Real, TILE_SIZE, MAX_N_WATER, MAX_N_CONN>&
        shared_m) {
  intra_dat.pair_data.pose_ind = pose_ind;
  intra_dat.r1.block_ind = block_ind1;
  intra_dat.r2.block_ind = block_ind1;
  intra_dat.r1.block_type = block_type1;
  intra_dat.r2.block_type = block_type1;
  intra_dat.r1.rot_coord_offset = rot_coord_offset[rot_ind1];
  intra_dat.r2.rot_coord_offset = intra_dat.r1.rot_coord_offset;
  intra_dat.pair_data.max_important_bond_separation =
      max_important_bond_separation;

  // we are not going to load count pair data into shared memory because
  // we are not going to use that data from shared memory; setting
  // in_count_pair_striking_distance to false prevents downstream loading
  // of count-pair data, which waste memory bandwidth
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
  intra_dat.r1.pose_coords = shared_m.pose_coords1;
  intra_dat.r2.pose_coords = shared_m.pose_coords2;
  intra_dat.r1.water_coords = shared_m.water_coords1;
  intra_dat.r2.water_coords = shared_m.water_coords2;
  intra_dat.r1.pol_occ_tile_inds = shared_m.pol_occ_tile_inds1;
  intra_dat.r2.pol_occ_tile_inds = shared_m.pol_occ_tile_inds2;
  intra_dat.r1.lk_ball_params = shared_m.lk_ball_params1;
  intra_dat.r2.lk_ball_params = shared_m.lk_ball_params2;

  // these count pair arrays are not going to be used
  intra_dat.r1.path_dist = 0;
  intra_dat.r2.path_dist = 0;
  intra_dat.pair_data.conn_seps = 0;

  // Final data members
  intra_dat.pair_data.global_params = global_params[0];
  intra_dat.pair_data.total_lk_ball_iso = 0;
  intra_dat.pair_data.total_lk_ball = 0;
  intra_dat.pair_data.total_lk_bridge = 0;
  intra_dat.pair_data.total_lk_bridge_uncpl = 0;
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    int TILE_SIZE,
    int MAX_N_WATER,
    int MAX_N_CONN,
    typename Real,
    typename Int>
void TMOL_DEVICE_FUNC lk_ball_load_intrares1_tile_data_to_shared(
    TView<Vec<Real, 3>, 1, Dev> pose_coords,
    TView<Vec<Real, 3>, 2, Dev> water_coords,
    TView<Int, 2, Dev> block_type_tile_n_polar_atoms,
    TView<Int, 2, Dev> block_type_tile_n_occluder_atoms,
    TView<Int, 3, Dev> block_type_tile_pol_occ_inds,
    TView<LKBallTypeParams<Real>, 3, Dev> block_type_tile_lk_ball_params,
    int tile_ind,
    int start_atom1,
    int n_atoms_to_load1,
    LKBallScoringData<Real>& intra_dat,
    LKBallBlockPairSharedData<Real, TILE_SIZE, MAX_N_WATER, MAX_N_CONN>&
        shared_m) {
  auto store_n_pol_n_occ1 = ([&](int tid) {
    int n_pol =
        block_type_tile_n_polar_atoms[intra_dat.r1.block_type][tile_ind];
    int n_occ =
        block_type_tile_n_occluder_atoms[intra_dat.r1.block_type][tile_ind];
    intra_dat.r1.n_polars = n_pol;
    intra_dat.r1.n_occluders = n_occ;
    if (tid == 0) {
      shared_m.n_polars1 = n_pol;
      shared_m.n_occluders1 = n_occ;
    }
  });
  DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(store_n_pol_n_occ1);
  lk_ball_load_block_coords_and_params_into_shared<
      DeviceDispatch,
      Dev,
      nt,
      MAX_N_WATER>(
      pose_coords,
      water_coords,
      block_type_tile_pol_occ_inds,
      block_type_tile_lk_ball_params,
      intra_dat.pair_data.pose_ind,
      tile_ind,
      intra_dat.r1,
      n_atoms_to_load1,
      start_atom1);
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    int TILE_SIZE,
    int MAX_N_WATER,
    int MAX_N_CONN,
    typename Real,
    typename Int>
void TMOL_DEVICE_FUNC lk_ball_load_intrares2_tile_data_to_shared(
    TView<Vec<Real, 3>, 1, Dev> pose_coords,
    TView<Vec<Real, 3>, 2, Dev> water_coords,
    TView<Int, 2, Dev> block_type_tile_n_polar_atoms,
    TView<Int, 2, Dev> block_type_tile_n_occluder_atoms,
    TView<Int, 3, Dev> block_type_tile_pol_occ_inds,
    TView<LKBallTypeParams<Real>, 3, Dev> block_type_tile_lk_ball_params,
    int tile_ind,
    int start_atom2,
    int n_atoms_to_load2,
    LKBallScoringData<Real>& intra_dat,
    LKBallBlockPairSharedData<Real, TILE_SIZE, MAX_N_WATER, MAX_N_CONN>&
        shared_m) {
  // A prior same-tile lk_ball_load_intrares_data_from_shared call may have
  // aliased r2's pointers to the "1" shared-memory arrays. Reset them to
  // the "2" arrays so the load below writes to the correct destination.
  intra_dat.r2.pose_coords = shared_m.pose_coords2;
  intra_dat.r2.water_coords = shared_m.water_coords2;
  intra_dat.r2.pol_occ_tile_inds = shared_m.pol_occ_tile_inds2;
  intra_dat.r2.lk_ball_params = shared_m.lk_ball_params2;
  auto store_n_pol_n_occ2 = ([&](int tid) {
    int n_pol =
        block_type_tile_n_polar_atoms[intra_dat.r2.block_type][tile_ind];
    int n_occ =
        block_type_tile_n_occluder_atoms[intra_dat.r2.block_type][tile_ind];
    intra_dat.r2.n_polars = n_pol;
    intra_dat.r2.n_occluders = n_occ;
    if (tid == 0) {
      shared_m.n_polars2 = n_pol;
      shared_m.n_occluders2 = n_occ;
    }
  });
  DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(store_n_pol_n_occ2);
  lk_ball_load_block_coords_and_params_into_shared<
      DeviceDispatch,
      Dev,
      nt,
      MAX_N_WATER>(
      pose_coords,
      water_coords,
      block_type_tile_pol_occ_inds,
      block_type_tile_lk_ball_params,
      intra_dat.pair_data.pose_ind,
      tile_ind,
      intra_dat.r2,
      n_atoms_to_load2,
      start_atom2);
}

template <
    tmol::Device Dev,
    int TILE_SIZE,
    int MAX_N_WATER,
    int MAX_N_CONN,
    typename Real>
void TMOL_DEVICE_FUNC lk_ball_load_intrares_data_from_shared(
    int tile_ind1,
    int tile_ind2,
    LKBallBlockPairSharedData<Real, TILE_SIZE, MAX_N_WATER, MAX_N_CONN>&
        shared_m,
    LKBallScoringData<Real>& intra_dat) {
  // set the pointers in intra_dat to point at the shared-memory arrays
  // If we are evaluating the energies between atoms in the same tile
  // then only the "1" shared-memory arrays will be loaded with data;
  // we will point the "2" memory pointers at the "1" arrays
  bool same_tile = tile_ind1 == tile_ind2;
  if constexpr (Dev == tmol::Device::CPU) {
    if (same_tile) {
      intra_dat.r2.pose_coords = intra_dat.r1.pose_coords;
      intra_dat.r2.water_coords = intra_dat.r1.water_coords;
      intra_dat.r2.lk_ball_params = intra_dat.r1.lk_ball_params;
    }
  } else {
    intra_dat.r1.pose_coords = shared_m.pose_coords1;
    intra_dat.r2.pose_coords =
        (same_tile ? shared_m.pose_coords1 : shared_m.pose_coords2);
    intra_dat.r1.water_coords = shared_m.water_coords1;
    intra_dat.r2.water_coords =
        (same_tile ? shared_m.water_coords1 : shared_m.water_coords2);
    intra_dat.r1.lk_ball_params = shared_m.lk_ball_params1;
    intra_dat.r2.lk_ball_params =
        (same_tile ? shared_m.lk_ball_params1 : shared_m.lk_ball_params2);
  }
  intra_dat.r1.pol_occ_tile_inds = shared_m.pol_occ_tile_inds1;
  intra_dat.r2.pol_occ_tile_inds =
      (same_tile ? shared_m.pol_occ_tile_inds1 : shared_m.pol_occ_tile_inds2);
  if (same_tile) {
    intra_dat.r2.n_polars = intra_dat.r1.n_polars;
    intra_dat.r2.n_occluders = intra_dat.r1.n_occluders;
  }
}

template <int MAX_N_WATER, typename Real>
TMOL_DEVICE_FUNC lk_ball_Vt<Real> lk_ball_atom_energy_full(
    int polar_ind,               // in [0:n_polar)
    int occluder_ind,            // in [0:n_occluders)
    int polar_atom_tile_ind,     // in [0:TILE_SIZE)
    int occluder_atom_tile_ind,  // in [0:TILE_SIZE)
    int polar_start,
    int occluder_start,
    LKBallSingleResData<Real> const& polar_block_dat,
    LKBallSingleResData<Real> const& occluder_block_dat,
    LKBallResPairData<Real> const& block_pair_dat,
    int cp_separation) {
  using tmol::score::common::coord_from_shared;
  using Real3 = Eigen::Matrix<Real, 3, 1>;

  if (cp_separation <= 3) {
    return {0, 0, 0, 0};
  }

  Real3 polar_xyz =
      coord_from_shared(polar_block_dat.pose_coords, polar_atom_tile_ind);
  Real3 occluder_xyz =
      coord_from_shared(occluder_block_dat.pose_coords, occluder_atom_tile_ind);

  Real const distance_threshold =
      block_pair_dat.global_params.distance_threshold;
  if ((polar_xyz - occluder_xyz).squaredNorm()
      >= distance_threshold * distance_threshold) {
    return {0, 0, 0, 0};
  }

  Eigen::Matrix<Real, 4, 3> wmat_polar;
  Eigen::Matrix<Real, 4, 3> wmat_occluder;

  for (int wi = 0; wi < MAX_N_WATER; wi++) {
    wmat_polar.row(wi) = coord_from_shared(
        polar_block_dat.water_coords, MAX_N_WATER * polar_atom_tile_ind + wi);
    wmat_occluder.row(wi) = coord_from_shared(
        occluder_block_dat.water_coords,
        MAX_N_WATER * occluder_atom_tile_ind + wi);
  }

  return lk_ball_score<Real, MAX_N_WATER>::V(
      polar_xyz,
      occluder_xyz,
      wmat_polar,
      wmat_occluder,
      cp_separation,
      polar_block_dat.lk_ball_params[polar_ind],
      occluder_block_dat.lk_ball_params[occluder_ind],
      block_pair_dat.global_params);
}

// Calculate and write to global memory only the derivatives for the two
// indicated atoms. Does not return the score
template <
    int TILE_SIZE,
    int MAX_N_WATER,
    typename Real,
    tmol::Device Dev,
    bool UseSharedScratch = false>
TMOL_DEVICE_FUNC void lk_ball_atom_derivs_full(
    int polar_ind,               // in [0:n_polar)
    int occluder_ind,            // in [0:n_occluders)
    int polar_atom_tile_ind,     // in [0:TILE_SIZE)
    int occluder_atom_tile_ind,  // in [0:TILE_SIZE)
    int polar_start,
    int occluder_start,
    LKBallSingleResData<Real> const& polar_block_dat,
    LKBallSingleResData<Real> const& occluder_block_dat,
    LKBallResPairData<Real> const& block_pair_dat,
    int cp_separation,
    Vec<Real, 4> dTdV,
    TView<Eigen::Matrix<Real, 3, 1>, 1, Dev> dV_d_pose_coords,
    TView<Eigen::Matrix<Real, 3, 1>, 2, Dev> dV_d_water_coords) {
  using Real3 = Eigen::Matrix<Real, 3, 1>;
  using tmol::score::common::accumulate;
  using tmol::score::common::coord_from_shared;

  if (cp_separation <= 3) {
    return;
  }

  Real3 polar_xyz =
      coord_from_shared(polar_block_dat.pose_coords, polar_atom_tile_ind);
  Real3 occluder_xyz =
      coord_from_shared(occluder_block_dat.pose_coords, occluder_atom_tile_ind);

  Real const distance_threshold =
      block_pair_dat.global_params.distance_threshold;
  if ((polar_xyz - occluder_xyz).squaredNorm()
      >= distance_threshold * distance_threshold)
    return;

  Eigen::Matrix<Real, MAX_N_WATER, 3> wmat_polar;
  Eigen::Matrix<Real, MAX_N_WATER, 3> wmat_occluder;

  for (int wi = 0; wi < MAX_N_WATER; wi++) {
    wmat_polar.row(wi) = coord_from_shared(
        polar_block_dat.water_coords, MAX_N_WATER * polar_atom_tile_ind + wi);
    wmat_occluder.row(wi) = coord_from_shared(
        occluder_block_dat.water_coords,
        MAX_N_WATER * occluder_atom_tile_ind + wi);
  }

  auto dV =
      lk_ball_score<Real, MAX_N_WATER>::template weighted_dV<UseSharedScratch>(
          polar_xyz,
          occluder_xyz,
          wmat_polar,
          wmat_occluder,
          cp_separation,
          polar_block_dat.lk_ball_params[polar_ind],
          occluder_block_dat.lk_ball_params[occluder_ind],
          block_pair_dat.global_params,
          dTdV);

  auto accum_derivs = ([&] TMOL_DEVICE_FUNC(
                           LKBallSingleResData<Real> const& block_dat,
                           int atom_ind,
                           Real3 const& derivs) {
    for (int j = 0; j < 3; ++j) {
      if (derivs[j] != 0) {
        accumulate<Dev, Real>::add(
            dV_d_pose_coords[block_dat.rot_coord_offset + atom_ind][j],
            derivs[j]);
      }
    }
  });

  auto water_accum_derivs =
      ([&] TMOL_DEVICE_FUNC(
           LKBallSingleResData<Real> const& block_dat,
           int atom_ind,
           int water_ind,
           Eigen::Matrix<Real, MAX_N_WATER, 3> const& derivs) {
        for (int j = 0; j < 3; ++j) {
          if (derivs(water_ind, j) != 0) {
            accumulate<Dev, Real>::add(
                dV_d_water_coords[block_dat.rot_coord_offset + atom_ind]
                                 [water_ind][j],
                derivs(water_ind, j));
          }
        }
      });

  accum_derivs(polar_block_dat, polar_start + polar_atom_tile_ind, dV.dI);
  accum_derivs(
      occluder_block_dat, occluder_start + occluder_atom_tile_ind, dV.dJ);
  for (int i = 0; i < MAX_N_WATER; ++i) {
    if (std::isnan(
            polar_block_dat
                .water_coords[3 * (MAX_N_WATER * polar_atom_tile_ind + i)]))
      continue;
    water_accum_derivs(
        polar_block_dat, polar_start + polar_atom_tile_ind, i, dV.dWI);
  }
  for (int i = 0; i < MAX_N_WATER; ++i) {
    if (std::isnan(
            occluder_block_dat
                .water_coords[3 * (MAX_N_WATER * occluder_atom_tile_ind + i)]))
      continue;
    water_accum_derivs(
        occluder_block_dat, occluder_start + occluder_atom_tile_ind, i, dV.dWJ);
  }
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    typename Func,
    typename Real>
void TMOL_DEVICE_FUNC eval_interres_pol_occ_pair_energies(
    LKBallScoringData<Real>& inter_dat,
    int start_atom1,
    int start_atom2,
    Func f) {
  if constexpr (Dev == tmol::Device::CPU) {
    // CPU workgroup lanes run serially; direct loops avoid flattened-index
    // division and modulo for every polar/occluder pair.
    for (int pol_ind = 0; pol_ind < inter_dat.r1.n_polars; ++pol_ind) {
      for (int occ_ind = 0; occ_ind < inter_dat.r2.n_occluders; ++occ_ind) {
        f(start_atom1, start_atom2, pol_ind, occ_ind, inter_dat, true);
      }
    }
    for (int pol_ind = 0; pol_ind < inter_dat.r2.n_polars; ++pol_ind) {
      for (int occ_ind = 0; occ_ind < inter_dat.r1.n_occluders; ++occ_ind) {
        f(start_atom2, start_atom1, pol_ind, occ_ind, inter_dat, false);
      }
    }
    return;
  }

  auto eval_scores_for_pol_occ_pairs = ([&](int tid) {
    int const n_pol_occ_pairs =
        inter_dat.r1.n_polars * inter_dat.r2.n_occluders
        + inter_dat.r1.n_occluders * inter_dat.r2.n_polars;
    for (int i = tid; i < n_pol_occ_pairs; i += nt) {
      bool r1_polar = i < inter_dat.r1.n_polars * inter_dat.r2.n_occluders;
      int pair_ind =
          r1_polar ? i : i - inter_dat.r1.n_polars * inter_dat.r2.n_occluders;
      LKBallSingleResData<Real> const& pol_dat =
          r1_polar ? inter_dat.r1 : inter_dat.r2;
      LKBallSingleResData<Real> const& occ_dat =
          r1_polar ? inter_dat.r2 : inter_dat.r1;
      int pol_ind = pair_ind / occ_dat.n_occluders;
      int occ_ind = pair_ind % occ_dat.n_occluders;
      int pol_start = r1_polar ? start_atom1 : start_atom2;
      int occ_start = r1_polar ? start_atom2 : start_atom1;

      // Now that we have the pair of polar/occluder atoms to focus on,
      // invoke the lambda function that will operate on them
      f(pol_start, occ_start, pol_ind, occ_ind, inter_dat, r1_polar);
    }
  });
  DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(
      eval_scores_for_pol_occ_pairs);
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    typename Func,
    typename Real>
void TMOL_DEVICE_FUNC eval_intrares_pol_occ_pair_energies(
    LKBallScoringData<Real>& intra_dat,
    int start_atom1,
    int start_atom2,
    Func f) {
  if constexpr (Dev == tmol::Device::CPU) {
    if (start_atom1 == start_atom2) {
      for (int pol_ind = 0; pol_ind < intra_dat.r1.n_polars; ++pol_ind) {
        for (int occ_ind = 0; occ_ind < intra_dat.r1.n_occluders; ++occ_ind) {
          if (pol_ind == occ_ind) continue;
          f(start_atom1, start_atom1, pol_ind, occ_ind, intra_dat, true);
        }
      }
    } else {
      for (int pol_ind = 0; pol_ind < intra_dat.r1.n_polars; ++pol_ind) {
        for (int occ_ind = 0; occ_ind < intra_dat.r2.n_occluders; ++occ_ind) {
          f(start_atom1, start_atom2, pol_ind, occ_ind, intra_dat, true);
        }
      }
      for (int pol_ind = 0; pol_ind < intra_dat.r2.n_polars; ++pol_ind) {
        for (int occ_ind = 0; occ_ind < intra_dat.r1.n_occluders; ++occ_ind) {
          f(start_atom2, start_atom1, pol_ind, occ_ind, intra_dat, false);
        }
      }
    }
    return;
  }

  auto eval_scores_for_pol_occ_pairs = ([&](int tid) {
    if (start_atom1 == start_atom2) {
      int const n_pol_occ_pairs =
          intra_dat.r1.n_polars * intra_dat.r1.n_occluders;
      for (int i = tid; i < n_pol_occ_pairs; i += nt) {
        int pol_ind = i / intra_dat.r1.n_occluders;
        int occ_ind = i % intra_dat.r1.n_occluders;

        // No atom can occlude itself
        if (pol_ind == occ_ind) continue;

        // Now that we have the pair of polar/occluder atoms to focus on,
        // invoke the lambda function that will operate on them
        f(start_atom1, start_atom1, pol_ind, occ_ind, intra_dat, true);
      }
    } else {
      // Similar to the inter-residue case; all atoms of tile X
      // are evaluated against all atoms of tile Y
      int const n_pol_occ_pairs =
          intra_dat.r1.n_polars * intra_dat.r2.n_occluders
          + intra_dat.r1.n_occluders * intra_dat.r2.n_polars;
      for (int i = tid; i < n_pol_occ_pairs; i += nt) {
        bool r1_polar = i < intra_dat.r1.n_polars * intra_dat.r2.n_occluders;
        int pair_ind =
            r1_polar ? i : i - intra_dat.r1.n_polars * intra_dat.r2.n_occluders;
        LKBallSingleResData<Real> const& occ_dat =
            r1_polar ? intra_dat.r2 : intra_dat.r1;
        int pol_ind = pair_ind / occ_dat.n_occluders;
        int occ_ind = pair_ind % occ_dat.n_occluders;
        int pol_start = r1_polar ? start_atom1 : start_atom2;
        int occ_start = r1_polar ? start_atom2 : start_atom1;

        // Now that we have the pair of polar/occluder atoms to focus on,
        // invoke the lambda function that will operate on them
        f(pol_start, occ_start, pol_ind, occ_ind, intra_dat, r1_polar);
      }
    }
  });

  DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(
      eval_scores_for_pol_occ_pairs);
}

}  // namespace potentials
}  // namespace lk_ball
}  // namespace score
}  // namespace tmol
