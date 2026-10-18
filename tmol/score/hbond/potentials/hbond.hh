#pragma once

#include <tmol/utility/portable/array.hh>
#include <tmol/utility/portable/matrix.hh>
#include <tmol/utility/portable/metal_view.hh>
#include <tmol/utility/portable/qualifiers.hh>

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/count_pair.hh>
#include <tmol/score/common/data_loading.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/workgroup_lane.hh>
#include <tmol/score/hbond/potentials/params_types.hh>
#include <tmol/score/hbond/potentials/potentials.hh>

namespace tmol {
namespace score {
namespace hbond {
namespace potentials {

using tmol::score::common::WorkgroupLane;

using tmol::portable::Vec;

template <typename Real>
class HBondSingleResData {
 public:
  int rot_ind;
  int block_ind;
  int block_type;
  int rot_coord_offset;
  int n_atoms;
  int n_conn;
  // These point into the workgroup's shared tile buffers, so they carry the
  // shared address space on backends that have one. See metal_view.hh.
  TMOL_SHARED_PTR(Real) coords;
  unsigned char n_donH;
  unsigned char n_acc;
  TMOL_SHARED_PTR(unsigned char) donH_tile_inds;
  TMOL_SHARED_PTR(unsigned char) acc_tile_inds;
  TMOL_SHARED_PTR(unsigned char) donH_type;
  TMOL_SHARED_PTR(unsigned char) acc_type;
  TMOL_SHARED_PTR(unsigned char) acc_hybridization;
  TMOL_SHARED_PTR(unsigned char) path_dist;
};

// The per-residue-pair state a tile's worth of hbond evaluation needs.
//
// The derived-atom and parameter tensors used to live here as TViews. They are
// passed to the atom-energy functions directly instead: a Metal translation
// unit cannot include the torch tensor layer at all, and the only consumers
// were those three functions, which already take every other input
// explicitly. The views reach them from the enclosing kernel body's scope,
// the same way the energy output tensor does.
template <typename Real>
class HBondRotPairData {
 public:
  int pose_ind;
  int max_important_bond_separation;
  int min_separation;
  bool in_count_pair_striking_dist;
  TMOL_SHARED_PTR(unsigned char) conn_seps;

  // load global params once; store totalE
  HBondGlobalParams<Real> global_params;
  Real total_hbond;
};

template <typename Real>
class HBondScoringData {
 public:
  HBondSingleResData<Real> r1;
  HBondSingleResData<Real> r2;
  HBondRotPairData<Real> pair_data;
};

template <typename Real, int TILE_SIZE, int MAX_N_CONN>
struct HBondBlockPairSharedData {
  Real coords1[TILE_SIZE * 3];  // 768 bytes for coords
  Real coords2[TILE_SIZE * 3];
  unsigned char n_donH1;  // 4 bytes for counts
  unsigned char n_donH2;
  unsigned char n_acc1;
  unsigned char n_acc2;
  unsigned char don_inds1[TILE_SIZE];  // 320 bytes for indices
  unsigned char don_inds2[TILE_SIZE];
  unsigned char acc_inds1[TILE_SIZE];
  unsigned char acc_inds2[TILE_SIZE];
  unsigned char don_type1[TILE_SIZE];
  unsigned char don_type2[TILE_SIZE];
  unsigned char acc_type1[TILE_SIZE];
  unsigned char acc_type2[TILE_SIZE];
  unsigned char acc_hybridization1[TILE_SIZE];
  unsigned char acc_hybridization2[TILE_SIZE];

  unsigned char conn_ats1[MAX_N_CONN];  // 12 bytes
  unsigned char conn_ats2[MAX_N_CONN];
  unsigned char path_dist1[MAX_N_CONN * TILE_SIZE];  // 384 bytes
  unsigned char path_dist2[MAX_N_CONN * TILE_SIZE];
  unsigned char conn_seps[MAX_N_CONN * MAX_N_CONN];  // 144 bytes
};

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    typename Real,
    typename CoordsView,
    typename TileDonHIndsView,
    typename TileAccIndsView,
    typename TileDonorTypeView,
    typename TileAcceptorTypeView,
    typename TileHybridizationView>
void TMOL_DEVICE_FUNC hbond_load_block_coords_and_params_into_shared(
    CoordsView coords,
    TileDonHIndsView block_type_tile_donH_inds,
    TileAccIndsView block_type_tile_acc_inds,
    TileDonorTypeView block_type_tile_donor_type,
    TileAcceptorTypeView block_type_tile_acceptor_type,
    TileHybridizationView block_type_tile_hybridization,
    int pose_ind,
    int tile_ind,
    TMOL_THREAD HBondSingleResData<Real>& r_dat,
    int n_atoms_to_load,
    int start_atom,
    // Trailing lane handle: empty and omissible on CPU and CUDA, required by
    // the Metal backend. See workgroup_lane.hh.
    WorkgroupLane<Dev> lane = {}) {
  // pre-condition: n_atoms_to_load < TILE_SIZE
  // Note that TILE_SIZE is not explicitly passed in, but is "present"
  // in r_dat.coords allocation

  DeviceDispatch<Dev>::template copy_contiguous_data<nt, 3>(
      r_dat.coords,
      reinterpret_cast<TMOL_DEVICE_PTR(Real)>(
          &coords[r_dat.rot_coord_offset + start_atom]),
      n_atoms_to_load * 3,
      lane);
  DeviceDispatch<Dev>::template copy_contiguous_data_and_cast<nt, 1>(
      r_dat.donH_tile_inds,
      &block_type_tile_donH_inds[r_dat.block_type][tile_ind][0],
      r_dat.n_donH,
      lane);
  DeviceDispatch<Dev>::template copy_contiguous_data_and_cast<nt, 1>(
      r_dat.acc_tile_inds,
      &block_type_tile_acc_inds[r_dat.block_type][tile_ind][0],
      r_dat.n_acc,
      lane);
  DeviceDispatch<Dev>::template copy_contiguous_data_and_cast<nt, 1>(
      r_dat.donH_type,
      &block_type_tile_donor_type[r_dat.block_type][tile_ind][0],
      r_dat.n_donH,
      lane);
  DeviceDispatch<Dev>::template copy_contiguous_data_and_cast<nt, 1>(
      r_dat.acc_type,
      &block_type_tile_acceptor_type[r_dat.block_type][tile_ind][0],
      r_dat.n_acc,
      lane);
  DeviceDispatch<Dev>::template copy_contiguous_data_and_cast<nt, 1>(
      r_dat.acc_hybridization,
      &block_type_tile_hybridization[r_dat.block_type][tile_ind][0],
      r_dat.n_acc,
      lane);
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    int TILE_SIZE,
    typename Real,
    typename CoordsView,
    typename TileDonHIndsView,
    typename TileAccIndsView,
    typename TileDonorTypeView,
    typename TileAcceptorTypeView,
    typename TileHybridizationView,
    typename PathDistanceView>
void TMOL_DEVICE_FUNC hbond_load_block_into_shared(
    CoordsView coords,
    TileDonHIndsView block_type_tile_donH_inds,
    TileAccIndsView block_type_tile_acc_inds,
    TileDonorTypeView block_type_tile_donor_type,
    TileAcceptorTypeView block_type_tile_acceptor_type,
    TileHybridizationView block_type_tile_hybridization,
    PathDistanceView block_type_path_distance,
    int pose_ind,
    int tile_ind,
    TMOL_THREAD HBondSingleResData<Real>& r_dat,
    int n_atoms_to_load,
    int start_atom,
    bool count_pair_striking_dist,
    TMOL_SHARED_PTR(unsigned char) conn_ats,
    // Trailing lane handle: empty and omissible on CPU and CUDA, required by
    // the Metal backend. See workgroup_lane.hh.
    WorkgroupLane<Dev> lane = {}) {
  hbond_load_block_coords_and_params_into_shared<DeviceDispatch, Dev, nt, Real>(
      coords,
      block_type_tile_donH_inds,
      block_type_tile_acc_inds,
      block_type_tile_donor_type,
      block_type_tile_acceptor_type,
      block_type_tile_hybridization,
      pose_ind,
      tile_ind,
      r_dat,
      n_atoms_to_load,
      start_atom,
      lane);

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
    DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(
        copy_path_dists, lane);
  }
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    typename Real,
    int TILE_SIZE,
    int MAX_N_CONN,
    typename RotCoordOffsetView,
    typename PoseStackNearBlocksView,
    typename InterBlockBondsepView,
    typename BlockTypeNInterblockBondsView,
    typename BlockTypeAtomsFormingChemicalBondsView,
    typename GlobalParamsView>
void TMOL_DEVICE_FUNC hbond_load_tile_invariant_interres_data(
    RotCoordOffsetView rot_coord_offset,
    PoseStackNearBlocksView pose_stack_near_blocks,
    InterBlockBondsepView pose_stack_inter_block_bondsep,

    BlockTypeNInterblockBondsView block_type_n_interblock_bonds,
    BlockTypeAtomsFormingChemicalBondsView
        block_type_atoms_forming_chemical_bonds,

    GlobalParamsView global_params,

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
    TMOL_THREAD HBondScoringData<Real>& inter_dat,
    TMOL_SHARED_REF(HBondBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN>)
        shared_m,
    // Trailing lane handle: empty and omissible on CPU and CUDA, required by
    // the Metal backend. See workgroup_lane.hh.
    WorkgroupLane<Dev> lane = {}) {
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
    // On CPU: a for loop executed once; on GPU threads within the
    // workgroup working in parallel will just continue to work in
    // parallel
    DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(
        load_count_pair_conn_at_data, lane);
  }

  // Final data members
  inter_dat.pair_data.global_params = global_params[0];
  inter_dat.pair_data.total_hbond = 0;
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    int TILE_SIZE,
    int MAX_N_CONN,
    typename Real,
    typename CoordsView,
    typename TileNDonHView,
    typename TileNAccView,
    typename TileDonHIndsView,
    typename TileAccIndsView,
    typename TileDonorTypeView,
    typename TileAcceptorTypeView,
    typename TileHybridizationView,
    typename PathDistanceView>
void TMOL_DEVICE_FUNC hbond_load_interres1_tile_data_to_shared(
    CoordsView coords,
    TileNDonHView block_type_tile_n_donH,
    TileNAccView block_type_tile_n_acc,
    TileDonHIndsView block_type_tile_donH_inds,
    TileAccIndsView block_type_tile_acc_inds,
    TileDonorTypeView block_type_tile_donor_type,
    TileAcceptorTypeView block_type_tile_acceptor_type,
    TileHybridizationView block_type_tile_hybridization,
    PathDistanceView block_type_path_distance,
    int tile_ind,
    int start_atom1,
    int n_atoms_to_load1,
    TMOL_THREAD HBondScoringData<Real>& inter_dat,
    TMOL_SHARED_REF(HBondBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN>)
        shared_m,
    // Trailing lane handle: empty and omissible on CPU and CUDA, required by
    // the Metal backend. See workgroup_lane.hh.
    WorkgroupLane<Dev> lane = {}) {
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
  DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(
      store_n_don_n_acc1, lane);
  hbond_load_block_into_shared<DeviceDispatch, Dev, nt, TILE_SIZE, Real>(
      coords,
      block_type_tile_donH_inds,
      block_type_tile_acc_inds,
      block_type_tile_donor_type,
      block_type_tile_acceptor_type,
      block_type_tile_hybridization,
      block_type_path_distance,
      inter_dat.pair_data.pose_ind,
      tile_ind,
      inter_dat.r1,
      n_atoms_to_load1,
      start_atom1,
      inter_dat.pair_data.in_count_pair_striking_dist,
      shared_m.conn_ats1,
      lane);
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    int TILE_SIZE,
    int MAX_N_CONN,
    typename Real,
    typename CoordsView,
    typename TileNDonHView,
    typename TileNAccView,
    typename TileDonHIndsView,
    typename TileAccIndsView,
    typename TileDonorTypeView,
    typename TileAcceptorTypeView,
    typename TileHybridizationView,
    typename PathDistanceView>
void TMOL_DEVICE_FUNC hbond_load_interres2_tile_data_to_shared(
    CoordsView coords,
    TileNDonHView block_type_tile_n_donH,
    TileNAccView block_type_tile_n_acc,
    TileDonHIndsView block_type_tile_donH_inds,
    TileAccIndsView block_type_tile_acc_inds,
    TileDonorTypeView block_type_tile_donor_type,
    TileAcceptorTypeView block_type_tile_acceptor_type,
    TileHybridizationView block_type_tile_hybridization,
    PathDistanceView block_type_path_distance,
    int tile_ind,
    int start_atom2,
    int n_atoms_to_load2,
    TMOL_THREAD HBondScoringData<Real>& inter_dat,
    TMOL_SHARED_REF(HBondBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN>)
        shared_m,
    // Trailing lane handle: empty and omissible on CPU and CUDA, required by
    // the Metal backend. See workgroup_lane.hh.
    WorkgroupLane<Dev> lane = {}) {
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
  DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(
      store_n_don_n_acc2, lane);
  hbond_load_block_into_shared<DeviceDispatch, Dev, nt, TILE_SIZE, Real>(
      coords,
      block_type_tile_donH_inds,
      block_type_tile_acc_inds,
      block_type_tile_donor_type,
      block_type_tile_acceptor_type,
      block_type_tile_hybridization,
      block_type_path_distance,
      inter_dat.pair_data.pose_ind,
      tile_ind,
      inter_dat.r2,
      n_atoms_to_load2,
      start_atom2,
      inter_dat.pair_data.in_count_pair_striking_dist,
      shared_m.conn_ats2,
      lane);
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    typename Real,
    int TILE_SIZE,
    int MAX_N_CONN,
    typename RotCoordOffsetView,
    typename GlobalParamsView>
void TMOL_DEVICE_FUNC hbond_load_tile_invariant_intrares_data(
    RotCoordOffsetView rot_coord_offset,
    GlobalParamsView global_params,
    int const max_important_bond_separation,
    int pose_ind,
    int rot_ind1,
    int block_ind1,
    int block_type1,
    int n_atoms1,
    TMOL_THREAD HBondScoringData<Real>& intra_dat,
    TMOL_SHARED_REF(HBondBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN>)
        shared_m,
    // Trailing lane handle: empty and omissible on CPU and CUDA, required by
    // the Metal backend. See workgroup_lane.hh.
    WorkgroupLane<Dev> lane = {}) {
  intra_dat.pair_data.pose_ind = pose_ind;
  intra_dat.r1.rot_ind = rot_ind1;
  intra_dat.r2.rot_ind = rot_ind1;
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
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    int TILE_SIZE,
    int MAX_N_CONN,
    typename Real,
    typename CoordsView,
    typename TileNDonHView,
    typename TileNAccView,
    typename TileDonHIndsView,
    typename TileAccIndsView,
    typename DonorTypeView,
    typename AcceptorTypeView,
    typename HybridizationView>
void TMOL_DEVICE_FUNC hbond_load_intrares1_tile_data_to_shared(
    CoordsView coords,
    TileNDonHView block_type_tile_n_donH,
    TileNAccView block_type_tile_n_acc,
    TileDonHIndsView block_type_tile_donH_inds,
    TileAccIndsView block_type_tile_acc_inds,
    DonorTypeView block_type_donor_type,
    AcceptorTypeView block_type_acceptor_type,
    HybridizationView block_type_hybridization,
    int tile_ind,
    int start_atom1,
    int n_atoms_to_load1,
    TMOL_THREAD HBondScoringData<Real>& intra_dat,
    TMOL_SHARED_REF(HBondBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN>)
        shared_m,
    // Trailing lane handle: empty and omissible on CPU and CUDA, required by
    // the Metal backend. See workgroup_lane.hh.
    WorkgroupLane<Dev> lane = {}) {
  auto store_n_don_n_acc1 = ([&](int tid) {
    int n_donH = block_type_tile_n_donH[intra_dat.r1.block_type][tile_ind];
    int n_acc = block_type_tile_n_acc[intra_dat.r1.block_type][tile_ind];
    intra_dat.r1.n_donH = n_donH;
    intra_dat.r1.n_acc = n_acc;
    if (tid == 0) {
      shared_m.n_donH1 = n_donH;
      shared_m.n_acc1 = n_acc;
    }
  });
  DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(
      store_n_don_n_acc1, lane);
  hbond_load_block_coords_and_params_into_shared<
      DeviceDispatch,
      Dev,
      nt,
      Real>(
      coords,
      block_type_tile_donH_inds,
      block_type_tile_acc_inds,
      block_type_donor_type,
      block_type_acceptor_type,
      block_type_hybridization,
      intra_dat.pair_data.pose_ind,
      tile_ind,
      intra_dat.r1,
      n_atoms_to_load1,
      start_atom1,
      lane);
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    int TILE_SIZE,
    int MAX_N_CONN,
    typename Real,
    typename CoordsView,
    typename TileNDonHView,
    typename TileNAccView,
    typename TileDonHIndsView,
    typename TileAccIndsView,
    typename DonorTypeView,
    typename AcceptorTypeView,
    typename HybridizationView>
void TMOL_DEVICE_FUNC hbond_load_intrares2_tile_data_to_shared(
    CoordsView coords,
    TileNDonHView block_type_tile_n_donH,
    TileNAccView block_type_tile_n_acc,
    TileDonHIndsView block_type_tile_donH_inds,
    TileAccIndsView block_type_tile_acc_inds,
    DonorTypeView block_type_donor_type,
    AcceptorTypeView block_type_acceptor_type,
    HybridizationView block_type_hybridization,
    int tile_ind,
    int start_atom2,
    int n_atoms_to_load2,
    TMOL_THREAD HBondScoringData<Real>& intra_dat,
    TMOL_SHARED_REF(HBondBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN>)
        shared_m,
    // Trailing lane handle: empty and omissible on CPU and CUDA, required by
    // the Metal backend. See workgroup_lane.hh.
    WorkgroupLane<Dev> lane = {}) {
  // A prior same-tile hbond_load_intrares_data_from_shared call may have
  // aliased r2's pointers to the "1" shared-memory arrays. Reset them to
  // the "2" arrays so the load below writes to the correct destination.
  intra_dat.r2.donH_tile_inds = shared_m.don_inds2;
  intra_dat.r2.acc_tile_inds = shared_m.acc_inds2;
  intra_dat.r2.donH_type = shared_m.don_type2;
  intra_dat.r2.acc_type = shared_m.acc_type2;
  intra_dat.r2.acc_hybridization = shared_m.acc_hybridization2;
  intra_dat.r2.coords = shared_m.coords2;
  auto store_n_don_n_acc2 = ([&](int tid) {
    int n_donH = block_type_tile_n_donH[intra_dat.r2.block_type][tile_ind];
    int n_acc = block_type_tile_n_acc[intra_dat.r2.block_type][tile_ind];
    intra_dat.r2.n_donH = n_donH;
    intra_dat.r2.n_acc = n_acc;
    if (tid == 0) {
      shared_m.n_donH2 = n_donH;
      shared_m.n_acc2 = n_acc;
    }
  });
  DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(
      store_n_don_n_acc2, lane);
  hbond_load_block_coords_and_params_into_shared<
      DeviceDispatch,
      Dev,
      nt,
      Real>(
      coords,
      block_type_tile_donH_inds,
      block_type_tile_acc_inds,
      block_type_donor_type,
      block_type_acceptor_type,
      block_type_hybridization,
      intra_dat.pair_data.pose_ind,
      tile_ind,
      intra_dat.r2,
      n_atoms_to_load2,
      start_atom2,
      lane);
}

template <int TILE_SIZE, int MAX_N_CONN, typename Real>
void TMOL_DEVICE_FUNC hbond_load_intrares_data_from_shared(
    int tile_ind1,
    int tile_ind2,
    TMOL_SHARED_REF(HBondBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN>)
        shared_m,
    TMOL_THREAD HBondScoringData<Real>& intra_dat) {
  // set the pointers in intra_dat to point at the shared-memory arrays
  // If we are evaluating the energies between atoms in the same tile
  // then only the "1" shared-memory arrays will be loaded with data;
  // we will point the "2" memory pointers at the "1" arrays
  bool same_tile = tile_ind1 == tile_ind2;
  intra_dat.r1.coords = shared_m.coords1;
  intra_dat.r2.coords = (same_tile ? shared_m.coords1 : shared_m.coords2);
  intra_dat.r1.donH_tile_inds = shared_m.don_inds1;
  intra_dat.r1.acc_tile_inds = shared_m.acc_inds1;
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
  if (same_tile) {
    intra_dat.r2.n_donH = intra_dat.r1.n_donH;
    intra_dat.r2.n_acc = intra_dat.r1.n_acc;
  }
}

// Count-pair separation for an inter-residue atom pair.
//
// Lives here rather than in hbond_pose_score.impl.hh -- which it used to share
// a file with -- because the kernel bodies call it, and they are compiled by
// the Metal toolchain, which cannot see the torch tensor layer that the
// dispatch entry points reach.
template <int TILE, typename InterEnergyData>
TMOL_HD int interres_count_pair_separation(
    TMOL_THREAD InterEnergyData const& inter_dat,
    int atom_tile_ind1,
    int atom_tile_ind2) {
  int separation = inter_dat.pair_data.min_separation;
  if (separation <= inter_dat.pair_data.max_important_bond_separation) {
    separation = common::count_pair::shared_mem_inter_block_separation<TILE>(
        inter_dat.pair_data.max_important_bond_separation,
        atom_tile_ind1,
        atom_tile_ind2,
        inter_dat.r1.n_conn,
        inter_dat.r2.n_conn,
        inter_dat.r1.path_dist,
        inter_dat.r2.path_dist,
        inter_dat.pair_data.conn_seps);
  }
  return separation;
}

template <
    int TILE_SIZE,
    typename Real,
    typename Int,
    typename DerivedCoordsView,
    typename DerivedAtomIndsView,
    typename PairParamsView,
    typename PairPolynomialsView>
TMOL_DEVICE_FUNC Real hbond_atom_energy_full(
    int donH_ind,             // in [0:n_donH)
    int acc_ind,              // in [0:n_acc)
    int don_h_atom_tile_ind,  // in [0:TILE_SIZE)
    int acc_atom_tile_ind,    // in [0:TILE_SIZE)
    int don_start,
    int acc_start,
    TMOL_THREAD HBondSingleResData<Real> const& don_dat,
    TMOL_THREAD HBondSingleResData<Real> const& acc_dat,
    TMOL_THREAD HBondRotPairData<Real> const& respair_dat,
    int cp_separation,
    // The derived-atom tensors produced by the hbond pre-pass kernel, indexed
    // by the global pose-atom index (the same index that addresses
    // rot_coords / dV_dcoords):
    //   derived_coords[atom_ind][0] : D  (donor heavy) for donor H atoms
    //   derived_coords[atom_ind][1] : B  (acceptor base) for acceptor atoms
    //   derived_coords[atom_ind][2] : B0 (acceptor base 2) for acceptor atoms
    // derived_atom_inds mirrors that layout, holding the source atom's global
    // pose-atom index (-1 if unused) so gradients can be attributed back
    // without re-walking the bond graph.
    DerivedCoordsView derived_coords,
    DerivedAtomIndsView derived_atom_inds,
    // The parameters that define the hbond energies, by donor and acceptor
    // type.
    PairParamsView pair_params,
    PairPolynomialsView pair_polynomials) {
  using Real3 = tmol::portable::Vec<Real, 3>;

  Real3 Hxyz = coord_from_shared(don_dat.coords, don_h_atom_tile_ind);
  Real3 Axyz = coord_from_shared(acc_dat.coords, acc_atom_tile_ind);

  Real const max_ha_dis = respair_dat.global_params.max_ha_dis;
  if ((Hxyz - Axyz).squaredNorm() < max_ha_dis * max_ha_dis) {
    int const H_pose_atom_ind =
        don_dat.rot_coord_offset + don_start + don_h_atom_tile_ind;
    int const A_pose_atom_ind =
        acc_dat.rot_coord_offset + acc_start + acc_atom_tile_ind;
    // An undefined donor or acceptor base (index -1) leaves NaN derived
    // coordinates; the pair scores 0 but its gradient would be NaN * 0.
    if (derived_atom_inds[H_pose_atom_ind][0] < 0
        || derived_atom_inds[A_pose_atom_ind][1] < 0) {
      return 0;
    }

    Real3 Dxyz = derived_coords[H_pose_atom_ind][0];
    Real3 Bxyz = derived_coords[A_pose_atom_ind][1];
    Real3 B0xyz = derived_coords[A_pose_atom_ind][2];

    unsigned char dt = don_dat.donH_type[donH_ind];
    unsigned char at = acc_dat.acc_type[acc_ind];

    return hbond_score<Real, Int>::V(
        Dxyz,
        Hxyz,
        Axyz,
        Bxyz,
        B0xyz,
        pair_params[dt][at],
        pair_polynomials[dt][at],
        respair_dat.global_params);
  } else {
    return 0;
  }
}

template <
    int TILE_SIZE,
    tmol::Device Dev,
    typename Real,
    typename Int,
    typename DerivedCoordsView,
    typename DerivedAtomIndsView,
    typename PairParamsView,
    typename PairPolynomialsView,
    typename DVDCoordsView>
TMOL_DEVICE_FUNC Real hbond_atom_derivs(
    int donH_ind,             // in [0:n_donH)
    int acc_ind,              // in [0:n_acc)
    int don_h_atom_tile_ind,  // in [0:TILE_SIZE)
    int acc_atom_tile_ind,    // in [0:TILE_SIZE)
    int don_start,
    int acc_start,
    TMOL_THREAD HBondSingleResData<Real> const& don_dat,
    TMOL_THREAD HBondSingleResData<Real> const& acc_dat,
    TMOL_THREAD HBondRotPairData<Real> const& respair_dat,
    int cp_separation,
    Real dTdV,
    // The derived-atom tensors produced by the hbond pre-pass kernel, indexed
    // by the global pose-atom index (the same index that addresses
    // rot_coords / dV_dcoords):
    //   derived_coords[atom_ind][0] : D  (donor heavy) for donor H atoms
    //   derived_coords[atom_ind][1] : B  (acceptor base) for acceptor atoms
    //   derived_coords[atom_ind][2] : B0 (acceptor base 2) for acceptor atoms
    // derived_atom_inds mirrors that layout, holding the source atom's global
    // pose-atom index (-1 if unused) so gradients can be attributed back
    // without re-walking the bond graph.
    DerivedCoordsView derived_coords,
    DerivedAtomIndsView derived_atom_inds,
    // The parameters that define the hbond energies, by donor and acceptor
    // type.
    PairParamsView pair_params,
    PairPolynomialsView pair_polynomials,
    DVDCoordsView dV_dcoords) {
  using Real3 = tmol::portable::Vec<Real, 3>;

  Real3 Hxyz = coord_from_shared(don_dat.coords, don_h_atom_tile_ind);
  Real3 Axyz = coord_from_shared(acc_dat.coords, acc_atom_tile_ind);

  Real const max_ha_dis = respair_dat.global_params.max_ha_dis;
  if ((Hxyz - Axyz).squaredNorm() < max_ha_dis * max_ha_dis) {
    int const H_pose_atom_ind =
        don_dat.rot_coord_offset + don_start + don_h_atom_tile_ind;
    int const A_pose_atom_ind =
        acc_dat.rot_coord_offset + acc_start + acc_atom_tile_ind;
    // An undefined donor or acceptor base (index -1) leaves NaN derived
    // coordinates; the pair scores 0 but its gradient would be NaN * 0.
    if (derived_atom_inds[H_pose_atom_ind][0] < 0
        || derived_atom_inds[A_pose_atom_ind][1] < 0) {
      return 0;
    }

    Real3 Dxyz = derived_coords[H_pose_atom_ind][0];
    Real3 Bxyz = derived_coords[A_pose_atom_ind][1];
    Real3 B0xyz = derived_coords[A_pose_atom_ind][2];

    int const D_pose_atom_ind =
        derived_atom_inds[H_pose_atom_ind][0];
    int const B_pose_atom_ind =
        derived_atom_inds[A_pose_atom_ind][1];
    int const B0_pose_atom_ind =
        derived_atom_inds[A_pose_atom_ind][2];

    unsigned char dt = don_dat.donH_type[donH_ind];
    unsigned char at = acc_dat.acc_type[acc_ind];

    auto hbond_V_dV = hbond_score<Real, Int>::V_dV(
        Dxyz,
        Hxyz,
        Axyz,
        Bxyz,
        B0xyz,
        pair_params[dt][at],
        pair_polynomials[dt][at],
        respair_dat.global_params);

    Real dVdT_m = dTdV;

    auto accum_at = ([&] TMOL_DEVICE_FUNC(int atom_ind, Real3 dV_dat) {
      if (atom_ind < 0) return;
      for (int j = 0; j < 3; ++j) {
        if (dV_dat[j] != 0) {
          accumulate<Dev, Real>::add(
              dV_dcoords[atom_ind][j], dVdT_m * dV_dat[j]);
        }
      }
    });

    accum_at(D_pose_atom_ind, hbond_V_dV.dV_dD);
    accum_at(H_pose_atom_ind, hbond_V_dV.dV_dH);
    accum_at(A_pose_atom_ind, hbond_V_dV.dV_dA);
    accum_at(B_pose_atom_ind, hbond_V_dV.dV_dB);
    accum_at(B0_pose_atom_ind, hbond_V_dV.dV_dB0);
    return hbond_V_dV.V;
  } else {
    return 0;
  }
}

template <
    int TILE_SIZE,
    tmol::Device Dev,
    typename Real,
    typename Int,
    typename DerivedCoordsView,
    typename DerivedAtomIndsView,
    typename PairParamsView,
    typename PairPolynomialsView,
    typename DVDCoordsView>
TMOL_DEVICE_FUNC Real hbond_atom_energy_and_derivs_full(
    int donH_ind,             // in [0:n_donH)
    int acc_ind,              // in [0:n_acc)
    int don_h_atom_tile_ind,  // in [0:TILE_SIZE)
    int acc_atom_tile_ind,    // in [0:TILE_SIZE)
    int don_start,
    int acc_start,
    TMOL_THREAD HBondSingleResData<Real> const& don_dat,
    TMOL_THREAD HBondSingleResData<Real> const& acc_dat,
    TMOL_THREAD HBondRotPairData<Real> const& respair_dat,
    int cp_separation,
    // The derived-atom tensors produced by the hbond pre-pass kernel, indexed
    // by the global pose-atom index (the same index that addresses
    // rot_coords / dV_dcoords):
    //   derived_coords[atom_ind][0] : D  (donor heavy) for donor H atoms
    //   derived_coords[atom_ind][1] : B  (acceptor base) for acceptor atoms
    //   derived_coords[atom_ind][2] : B0 (acceptor base 2) for acceptor atoms
    // derived_atom_inds mirrors that layout, holding the source atom's global
    // pose-atom index (-1 if unused) so gradients can be attributed back
    // without re-walking the bond graph.
    DerivedCoordsView derived_coords,
    DerivedAtomIndsView derived_atom_inds,
    // The parameters that define the hbond energies, by donor and acceptor
    // type.
    PairParamsView pair_params,
    PairPolynomialsView pair_polynomials,
    DVDCoordsView dV_dcoords) {
  using Real3 = tmol::portable::Vec<Real, 3>;

  Real3 Hxyz = coord_from_shared(don_dat.coords, don_h_atom_tile_ind);
  Real3 Axyz = coord_from_shared(acc_dat.coords, acc_atom_tile_ind);

  Real const max_ha_dis = respair_dat.global_params.max_ha_dis;
  if ((Hxyz - Axyz).squaredNorm() < max_ha_dis * max_ha_dis) {
    int const H_pose_atom_ind =
        don_dat.rot_coord_offset + don_start + don_h_atom_tile_ind;
    int const A_pose_atom_ind =
        acc_dat.rot_coord_offset + acc_start + acc_atom_tile_ind;
    // An undefined donor or acceptor base (index -1) leaves NaN derived
    // coordinates; the pair scores 0 but its gradient would be NaN * 0.
    if (derived_atom_inds[H_pose_atom_ind][0] < 0
        || derived_atom_inds[A_pose_atom_ind][1] < 0) {
      return 0;
    }

    Real3 Dxyz = derived_coords[H_pose_atom_ind][0];
    Real3 Bxyz = derived_coords[A_pose_atom_ind][1];
    Real3 B0xyz = derived_coords[A_pose_atom_ind][2];

    int const D_pose_atom_ind =
        derived_atom_inds[H_pose_atom_ind][0];
    int const B_pose_atom_ind =
        derived_atom_inds[A_pose_atom_ind][1];
    int const B0_pose_atom_ind =
        derived_atom_inds[A_pose_atom_ind][2];

    unsigned char dt = don_dat.donH_type[donH_ind];
    unsigned char at = acc_dat.acc_type[acc_ind];

    auto hbond_V_dV = hbond_score<Real, Int>::V_dV(
        Dxyz,
        Hxyz,
        Axyz,
        Bxyz,
        B0xyz,
        pair_params[dt][at],
        pair_polynomials[dt][at],
        respair_dat.global_params);

    auto accum_at = ([&] TMOL_DEVICE_FUNC(int atom_ind, Real3 dV_dat) {
      if (atom_ind < 0) return;
      for (int j = 0; j < 3; ++j) {
        if (dV_dat[j] != 0) {
          accumulate<Dev, Real>::add(dV_dcoords[0][atom_ind][j], dV_dat[j]);
        }
      }
    });

    accum_at(D_pose_atom_ind, hbond_V_dV.dV_dD);
    accum_at(H_pose_atom_ind, hbond_V_dV.dV_dH);
    accum_at(A_pose_atom_ind, hbond_V_dV.dV_dA);
    accum_at(B_pose_atom_ind, hbond_V_dV.dV_dB);
    accum_at(B0_pose_atom_ind, hbond_V_dV.dV_dB0);
    return hbond_V_dV.V;
  } else {
    return 0;
  }
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    typename Func,
    typename Real>
void TMOL_DEVICE_FUNC eval_interres_don_acc_pair_energies(
    TMOL_THREAD HBondScoringData<Real>& inter_dat,
    int start_atom1,
    int start_atom2,
    Func f,
    // Trailing lane handle: empty and omissible on CPU and CUDA, required by
    // the Metal backend. See workgroup_lane.hh.
    WorkgroupLane<Dev> lane = {}) {
  auto eval_scores_for_don_acc_pairs = ([&](int tid) {
    int const n_don_acc_pairs = inter_dat.r1.n_donH * inter_dat.r2.n_acc
                                + inter_dat.r1.n_acc * inter_dat.r2.n_donH;
    for (int i = tid; i < n_don_acc_pairs; i += nt) {
      bool r1_don = i < inter_dat.r1.n_donH * inter_dat.r2.n_acc;
      int pair_ind = r1_don ? i : i - inter_dat.r1.n_donH * inter_dat.r2.n_acc;
      TMOL_THREAD HBondSingleResData<Real> const& don_dat =
          r1_don ? inter_dat.r1 : inter_dat.r2;
      TMOL_THREAD HBondSingleResData<Real> const& acc_dat =
          r1_don ? inter_dat.r2 : inter_dat.r1;
      int don_ind = pair_ind / acc_dat.n_acc;
      int acc_ind = pair_ind % acc_dat.n_acc;
      int don_start = r1_don ? start_atom1 : start_atom2;
      int acc_start = r1_don ? start_atom2 : start_atom1;

      Real E = tmol::portable::get<0>(
          f(don_start, acc_start, don_ind, acc_ind, inter_dat, r1_don));
      inter_dat.pair_data.total_hbond += E;
    }
  });
  DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(
      eval_scores_for_don_acc_pairs, lane);
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    typename Func,
    typename Real>
void TMOL_DEVICE_FUNC eval_intrares_don_acc_pair_energies(
    TMOL_THREAD HBondScoringData<Real>& intra_dat,
    int start_atom1,
    int start_atom2,
    Func f,
    // Trailing lane handle: empty and omissible on CPU and CUDA, required by
    // the Metal backend. See workgroup_lane.hh.
    WorkgroupLane<Dev> lane = {}) {
  auto eval_scores_for_don_acc_pairs = ([&](int tid) {
    if (start_atom1 == start_atom2) {
      int const n_don_acc_pairs = intra_dat.r1.n_donH * intra_dat.r1.n_acc;
      for (int i = tid; i < n_don_acc_pairs; i += nt) {
        int don_ind = i / intra_dat.r1.n_acc;
        int acc_ind = i % intra_dat.r1.n_acc;

        intra_dat.pair_data.total_hbond += tmol::portable::get<0>(
            f(start_atom1, start_atom1, don_ind, acc_ind, intra_dat, true));
      }
    } else {
      int const n_don_acc_pairs = intra_dat.r1.n_donH * intra_dat.r2.n_acc
                                  + intra_dat.r1.n_acc * intra_dat.r2.n_donH;
      for (int i = tid; i < n_don_acc_pairs; i += nt) {
        bool r1_don = i < intra_dat.r1.n_donH * intra_dat.r2.n_acc;
        int pair_ind =
            r1_don ? i : i - intra_dat.r1.n_donH * intra_dat.r2.n_acc;
        TMOL_THREAD HBondSingleResData<Real> const& acc_dat =
            r1_don ? intra_dat.r2 : intra_dat.r1;
        int don_ind = pair_ind / acc_dat.n_acc;
        int acc_ind = pair_ind % acc_dat.n_acc;
        int don_start = r1_don ? start_atom1 : start_atom2;
        int acc_start = r1_don ? start_atom2 : start_atom1;

        intra_dat.pair_data.total_hbond += tmol::portable::get<0>(
            f(don_start, acc_start, don_ind, acc_ind, intra_dat, r1_don));
      }
    }
  });

  DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(
      eval_scores_for_don_acc_pairs, lane);
}

}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol
