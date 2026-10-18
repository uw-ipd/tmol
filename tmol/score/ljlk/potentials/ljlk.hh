#pragma once


#include <tmol/utility/portable/array.hh>
#include <tmol/utility/portable/metal_view.hh>
#include <tmol/score/common/count_pair.hh>
#include <tmol/score/common/data_loading.hh>
#include <tmol/score/ljlk/potentials/common.hh>
#include <tmol/score/ljlk/potentials/lj.hh>
#include <tmol/score/ljlk/potentials/lk_isotropic.hh>

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

using tmol::score::common::accumulate_grad;

template <typename Real>
class LJLKSingleResData {
 public:
  int rot_ind;
  int block_type;
  int rot_coord_offset;
  int n_atoms;
  int n_conn;
  // These point into the workgroup's shared tile buffers, so they carry the
  // shared address space on backends that have one. See metal_view.hh.
  TMOL_SHARED_PTR(Real) coords;
  // Atom type indices, not parameter records. The tile used to hold a copy
  // of LJLKTypeParams per atom -- 52 bytes each, 3328 bytes of the block's
  // 5032 -- and threadgroup memory is what bounds this kernel's occupancy:
  // padding the block by 4 KB costs 39% and by 12 KB costs 148%. An index is
  // two bytes, so the two tiles cost 128 bytes instead of 3328, and the
  // parameters are read from the device table in the inner loop. That table
  // is a few kilobytes shared by every workgroup, so it sits in cache.
  TMOL_SHARED_PTR(short) atom_types;
  TMOL_SHARED_PTR(unsigned char) path_dist;
};

template <typename Real>
class LJLKScoringData {
 public:
  int pose_ind;
  int rot_ind1;
  int rot_ind2;
  int block_ind1;
  int block_ind2;
  LJLKSingleResData<Real> r1;
  LJLKSingleResData<Real> r2;
  int max_important_bond_separation;
  int min_separation;
  bool in_count_pair_striking_dist;
  TMOL_SHARED_PTR(unsigned char) conn_seps;
  LJGlobalParams<Real> global_params;
  Real total_ljatr;
  Real total_ljrep;
  Real total_lk;
};

template <typename Real, int TILE_SIZE, int MAX_N_CONN>
struct LJLKBlockPairSharedData {
  Real coords1[TILE_SIZE * 3];  // 384 bytes for coords
  Real coords2[TILE_SIZE * 3];
  short atom_types1[TILE_SIZE];  // 64 bytes of type indices, not params
  short atom_types2[TILE_SIZE];
  unsigned char conn_ats1[MAX_N_CONN];  // 12 bytes
  unsigned char conn_ats2[MAX_N_CONN];
  unsigned char path_dist1[MAX_N_CONN * TILE_SIZE];  // 384 bytes
  unsigned char path_dist2[MAX_N_CONN * TILE_SIZE];
  unsigned char conn_seps[MAX_N_CONN * MAX_N_CONN];  // 144 bytes
};

// Count-pair separation for an inter-residue atom pair.
//
// Lives here rather than in ljlk_pose_score.impl.hh -- which it used to share
// a file with -- because the kernel bodies call it, and they are compiled by
// the Metal toolchain, which cannot see the torch tensor layer that the
// dispatch entry points reach.
template <int TILE, template <typename> typename InterEnergyData, typename Real>
TMOL_HD int interres_count_pair_separation(
    TMOL_THREAD InterEnergyData<Real> const& inter_dat,
    int atom_tile_ind1,
    int atom_tile_ind2,
    bool crossover_3full) {
  int separation = inter_dat.min_separation;
  if (separation <= inter_dat.max_important_bond_separation) {
    separation = common::count_pair::shared_mem_inter_block_separation<TILE>(
        inter_dat.max_important_bond_separation,
        atom_tile_ind1,
        atom_tile_ind2,
        inter_dat.r1.n_conn,
        inter_dat.r2.n_conn,
        inter_dat.r1.path_dist,
        inter_dat.r2.path_dist,
        inter_dat.conn_seps);
  }
  if (crossover_3full && (separation == 3 || separation == 4)) separation = 5;
  return separation;
}

//// kernel macros
//    these define functions that are used in multiple lambda captures
//    variables that are expected to be captured for each macro are specified

// SCORE_INTER_LJ_ATOM_PAIR
// input argument:  a function with signature (
//     int atom_tile_idx1
//     int atom_tile_idx2
//     int start_atom1
//     int start_atom2
//     LJLKScoringData<Real> const &score_dat
//     int cp_separation)
//   ->tmol::portable::array<Real, 2>

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    int nt,
    typename Real,
    typename Int,
    typename CoordsView,
    typename BlockTypeAtomTypesView,
    typename TypeParamsView>
void TMOL_DEVICE_FUNC ljlk_load_block_coords_and_params_into_shared(
    CoordsView coords,
    BlockTypeAtomTypesView block_type_atom_types,
    TypeParamsView type_params,
    int pose_ind,
    TMOL_THREAD LJLKSingleResData<Real>& r_dat,
    int n_atoms_to_load,
    int start_atom,
    // Trailing lane handle: empty and omissible on CPU and CUDA, required
    // by the Metal backend. See workgroup_lane.hh.
    WorkgroupLane<D> lane = {}) {
  // pre-condition: n_atoms_to_load < TILE_SIZE
  // Note that TILE_SIZE is not explicitly passed in, but is "present"
  // in r_dat.coords allocation
  DeviceDispatch<D>::template copy_contiguous_data<nt, 3>(
      r_dat.coords,
      reinterpret_cast<TMOL_DEVICE_PTR(Real)>(
          &coords[r_dat.rot_coord_offset + start_atom]),
      n_atoms_to_load * 3,
      lane);
  auto copy_atom_types = ([=](int tid) {
    for (int count = tid; count < n_atoms_to_load; count += nt) {
      if (count < n_atoms_to_load) {
        int const atid = start_atom + count;
        int const attype = block_type_atom_types[r_dat.block_type][atid];
        // Negative means the slot has no atom; the index is carried through
        // so the reader sees the same condition the parameter copy used to.
        r_dat.atom_types[count] = short(attype);
      }
    }
  });
  DeviceDispatch<D>::template for_each_in_workgroup<nt>(copy_atom_types, lane);
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    int nt,
    int TILE_SIZE,
    typename Real,
    typename Int,
    typename CoordsView,
    typename BlockTypeAtomTypesView,
    typename TypeParamsView,
    typename BlockTypePathDistanceView>
void TMOL_DEVICE_FUNC ljlk_load_block_into_shared(
    CoordsView coords,
    BlockTypeAtomTypesView block_type_atom_types,
    TypeParamsView type_params,
    BlockTypePathDistanceView block_type_path_distance,
    int pose_ind,
    TMOL_THREAD LJLKSingleResData<Real>& r_dat,
    int n_atoms_to_load,
    int start_atom,
    bool count_pair_striking_dist,
    TMOL_SHARED_PTR(unsigned char) conn_ats,
    // Trailing lane handle: empty and omissible on CPU and CUDA, required
    // by the Metal backend. See workgroup_lane.hh.
    WorkgroupLane<D> lane = {}) {
  ljlk_load_block_coords_and_params_into_shared<DeviceDispatch, D, nt, Real, Int>(
      coords,
      block_type_atom_types,
      type_params,
      pose_ind,
      r_dat,
      n_atoms_to_load,
      start_atom,
      lane);

  auto copy_path_dists = ([=](int tid) {
    for (int count = tid; count < n_atoms_to_load; count += nt) {
      if (count_pair_striking_dist) {
        int const atid = start_atom + count;
        for (int j = 0; j < r_dat.n_conn; ++j) {
          unsigned char ij_path_dist =
              block_type_path_distance[r_dat.block_type][conn_ats[j]][atid];
          r_dat.path_dist[j * TILE_SIZE + count] = ij_path_dist;
        }
      }
    }
  });
  DeviceDispatch<D>::template for_each_in_workgroup<nt>(copy_path_dists, lane);
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    int nt,
    typename Int,
    typename Real,
    int TILE_SIZE,
    int MAX_N_CONN,
    typename RotCoordOffsetView,
    typename PoseStackNearBlocksView,
    typename BlockTypeNInterblockBondsView,
    typename BlockTypeAtomsFormingChemicalBondsView,
    typename PoseStackInterBlockBondsepView,
    typename GlobalParamsView>
void TMOL_DEVICE_FUNC ljlk_load_tile_invariant_interres_data(
    RotCoordOffsetView rot_coord_offset,
    PoseStackNearBlocksView pose_stack_near_blocks,
    BlockTypeNInterblockBondsView block_type_n_interblock_bonds,
    BlockTypeAtomsFormingChemicalBondsView block_type_atoms_forming_chemical_bonds,
    PoseStackInterBlockBondsepView pose_stack_inter_block_bondsep,
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
    TMOL_THREAD LJLKScoringData<Real>& inter_dat,
    TMOL_SHARED_REF(LJLKBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN>) shared_m,
    // Trailing lane handle: empty and omissible on CPU and CUDA, required
    // by the Metal backend. See workgroup_lane.hh.
    WorkgroupLane<D> lane = {}) {
  inter_dat.pose_ind = pose_ind;
  inter_dat.r1.rot_ind = rot_ind1;
  inter_dat.r2.rot_ind = rot_ind2;
  inter_dat.block_ind1 = block_ind1;
  inter_dat.block_ind2 = block_ind2;
  inter_dat.r1.block_type = block_type1;
  inter_dat.r2.block_type = block_type2;
  inter_dat.r1.rot_coord_offset = rot_coord_offset[rot_ind1];
  inter_dat.r2.rot_coord_offset = rot_coord_offset[rot_ind2];
  inter_dat.max_important_bond_separation = max_important_bond_separation;
  int const near_slot = common::count_pair::near_block_slot(
      pose_stack_near_blocks[pose_ind][block_ind1], block_ind2);
  inter_dat.min_separation =
      pose_stack_near_blocks[pose_ind][block_ind1][near_slot][1];
  inter_dat.in_count_pair_striking_dist =
      inter_dat.min_separation <= max_important_bond_separation;
  inter_dat.r1.n_atoms = n_atoms1;
  inter_dat.r2.n_atoms = n_atoms2;
  inter_dat.r1.n_conn = block_type_n_interblock_bonds[block_type1];
  inter_dat.r2.n_conn = block_type_n_interblock_bonds[block_type2];

  // set the pointers in inter_dat to point at the shared-memory arrays
  inter_dat.r1.coords = shared_m.coords1;
  inter_dat.r2.coords = shared_m.coords2;
  inter_dat.r1.atom_types = shared_m.atom_types1;
  inter_dat.r2.atom_types = shared_m.atom_types2;
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
              pose_stack_inter_block_bondsep[pose_ind][block_ind1][near_slot]
                                            [conn1][conn2];
        }
      }
    });
    // On CPU: a for loop executed once; on GPU threads within the
    // workgroup working in parallel will just continue to work in
    // parallel
    DeviceDispatch<D>::template for_each_in_workgroup<nt>(
        load_count_pair_conn_at_data, lane);
  }

  // Final data members
  inter_dat.global_params = global_params[0];
  inter_dat.total_ljatr = 0;
  inter_dat.total_ljrep = 0;
  inter_dat.total_lk = 0;
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    int nt,
    int TILE_SIZE,
    int MAX_N_CONN,
    typename Real,
    typename Int,
    typename CoordsView,
    typename BlockTypeAtomTypesView,
    typename TypeParamsView,
    typename BlockTypePathDistanceView>
void TMOL_DEVICE_FUNC ljlk_load_interres1_tile_data_to_shared(
    CoordsView coords,
    BlockTypeAtomTypesView block_type_atom_types,
    TypeParamsView type_params,
    BlockTypePathDistanceView block_type_path_distance,
    int,
    int start_atom1,
    int n_atoms_to_load1,
    TMOL_THREAD LJLKScoringData<Real>& inter_dat,
    TMOL_SHARED_REF(LJLKBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN>) shared_m,
    // Trailing lane handle: empty and omissible on CPU and CUDA, required
    // by the Metal backend. See workgroup_lane.hh.
    WorkgroupLane<D> lane = {}) {
  ljlk_load_block_into_shared<DeviceDispatch, D, nt, TILE_SIZE, Real, Int>(
      coords,
      block_type_atom_types,
      type_params,
      block_type_path_distance,
      inter_dat.pose_ind,
      inter_dat.r1,
      n_atoms_to_load1,
      start_atom1,
      inter_dat.in_count_pair_striking_dist,
      shared_m.conn_ats1,
      lane);
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    int nt,
    int TILE_SIZE,
    int MAX_N_CONN,
    typename Real,
    typename Int,
    typename CoordsView,
    typename BlockTypeAtomTypesView,
    typename TypeParamsView,
    typename BlockTypePathDistanceView>
void TMOL_DEVICE_FUNC ljlk_load_interres2_tile_data_to_shared(
    CoordsView coords,
    BlockTypeAtomTypesView block_type_atom_types,
    TypeParamsView type_params,
    BlockTypePathDistanceView block_type_path_distance,
    int,
    int start_atom2,
    int n_atoms_to_load2,
    TMOL_THREAD LJLKScoringData<Real>& inter_dat,
    TMOL_SHARED_REF(LJLKBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN>) shared_m,
    // Trailing lane handle: empty and omissible on CPU and CUDA, required
    // by the Metal backend. See workgroup_lane.hh.
    WorkgroupLane<D> lane = {}) {
  ljlk_load_block_into_shared<DeviceDispatch, D, nt, TILE_SIZE, Real, Int>(
      coords,
      block_type_atom_types,
      type_params,
      block_type_path_distance,
      inter_dat.pose_ind,
      inter_dat.r2,
      n_atoms_to_load2,
      start_atom2,
      inter_dat.in_count_pair_striking_dist,
      shared_m.conn_ats2,
      lane);
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    int nt,
    typename Int,
    typename Real,
    int TILE_SIZE,
    int MAX_N_CONN,
    typename RotCoordOffsetView,
    typename GlobalParamsView>
void TMOL_DEVICE_FUNC ljlk_load_tile_invariant_intrares_data(
    RotCoordOffsetView rot_coord_offset,
    GlobalParamsView global_params,
    int const max_important_bond_separation,
    int pose_ind,
    int rot_ind1,
    int block_ind1,
    int block_type1,
    int n_atoms1,
    TMOL_THREAD LJLKScoringData<Real>& intra_dat,
    TMOL_SHARED_REF(LJLKBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN>) shared_m,
    // Unused here -- this loader reaches no workgroup primitive -- but taken
    // for uniformity with its siblings, whose call sites in ljlk_kernels.hh
    // are generated by the same macro.
    WorkgroupLane<D> lane = {}) {
  intra_dat.pose_ind = pose_ind;
  intra_dat.r1.rot_ind = rot_ind1;
  intra_dat.r2.rot_ind = rot_ind1;
  intra_dat.block_ind1 = block_ind1;
  intra_dat.block_ind2 = block_ind1;
  intra_dat.r1.block_type = block_type1;
  intra_dat.r2.block_type = block_type1;
  intra_dat.r1.rot_coord_offset = rot_coord_offset[rot_ind1];
  intra_dat.r2.rot_coord_offset = intra_dat.r1.rot_coord_offset;
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
  intra_dat.r1.coords = shared_m.coords1;  // depends on tile pair!
  intra_dat.r2.coords = shared_m.coords2;  // depends on tile pair!
  intra_dat.r1.atom_types = shared_m.atom_types1;  // depends on tile pair!
  intra_dat.r2.atom_types = shared_m.atom_types2;  // depends on tile pair!

  // these count pair arrays are not going to be used
  intra_dat.r1.path_dist = 0;
  intra_dat.r2.path_dist = 0;
  intra_dat.conn_seps = 0;

  // Final data members
  intra_dat.global_params = global_params[0];
  intra_dat.total_ljatr = 0;
  intra_dat.total_ljrep = 0;
  intra_dat.total_lk = 0;
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    int nt,
    int TILE_SIZE,
    int MAX_N_CONN,
    typename Real,
    typename Int,
    typename CoordsView,
    typename BlockTypeAtomTypesView,
    typename TypeParamsView>
void TMOL_DEVICE_FUNC ljlk_load_intrares1_tile_data_to_shared(
    CoordsView coords,
    BlockTypeAtomTypesView block_type_atom_types,
    TypeParamsView type_params,
    int,
    int start_atom1,
    int n_atoms_to_load1,
    TMOL_THREAD LJLKScoringData<Real>& intra_dat,
    TMOL_SHARED_REF(LJLKBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN>) shared_m,
    // Trailing lane handle: empty and omissible on CPU and CUDA, required
    // by the Metal backend. See workgroup_lane.hh.
    WorkgroupLane<D> lane = {}) {
  ljlk_load_block_coords_and_params_into_shared<DeviceDispatch, D, nt, Real, Int>(
      coords,
      block_type_atom_types,
      type_params,
      intra_dat.pose_ind,
      intra_dat.r1,
      n_atoms_to_load1,
      start_atom1,
      lane);
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    int nt,
    int TILE_SIZE,
    int MAX_N_CONN,
    typename Real,
    typename Int,
    typename CoordsView,
    typename BlockTypeAtomTypesView,
    typename TypeParamsView>
void TMOL_DEVICE_FUNC ljlk_load_intrares2_tile_data_to_shared(
    CoordsView coords,
    BlockTypeAtomTypesView block_type_atom_types,
    TypeParamsView type_params,
    int,
    int start_atom2,
    int n_atoms_to_load2,
    TMOL_THREAD LJLKScoringData<Real>& intra_dat,
    TMOL_SHARED_REF(LJLKBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN>) shared_m,
    // Trailing lane handle: empty and omissible on CPU and CUDA, required
    // by the Metal backend. See workgroup_lane.hh.
    WorkgroupLane<D> lane = {}) {
  // A prior same-tile ljlk_load_intrares_data_from_shared call may have
  // aliased r2's pointers to the "1" shared-memory arrays. Reset them to
  // the "2" arrays so the load below writes to the correct destination.
  intra_dat.r2.coords = shared_m.coords2;
  intra_dat.r2.atom_types = shared_m.atom_types2;
  ljlk_load_block_coords_and_params_into_shared<DeviceDispatch, D, nt, Real, Int>(
      coords,
      block_type_atom_types,
      type_params,
      intra_dat.pose_ind,
      intra_dat.r2,
      n_atoms_to_load2,
      start_atom2,
      lane);
}

template <int TILE_SIZE, int MAX_N_CONN, typename Real>
void TMOL_DEVICE_FUNC ljlk_load_intrares_data_from_shared(
    int tile_ind1,
    int tile_ind2,
    TMOL_SHARED_REF(LJLKBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN>) shared_m,
    TMOL_THREAD LJLKScoringData<Real>& intra_dat) {
  // set the pointers in intra_dat to point at the shared-memory arrays
  // If we are evaluating the energies between atoms in the same tile
  // then only the "1" shared-memory arrays will be loaded with data;
  // we will point the "2" memory pointers at the "1" arrays
  bool same_tile = tile_ind1 == tile_ind2;
  intra_dat.r1.coords = shared_m.coords1;
  intra_dat.r2.coords = (same_tile ? shared_m.coords1 : shared_m.coords2);
  intra_dat.r1.atom_types = shared_m.atom_types1;
  intra_dat.r2.atom_types =
      (same_tile ? shared_m.atom_types1 : shared_m.atom_types2);
}

// Fused LJ/LK energy for pose scoring. Heavy-atom pairs reuse the coordinate
// loads, distance, and count-pair separation for both potentials.
template <typename Real, tmol::Device D,
    typename TypeParamsView>
TMOL_DEVICE_FUNC tmol::portable::array<Real, 3> ljlk_atom_energy(
    int atom_tile_ind1,
    int atom_tile_ind2,
    TMOL_THREAD LJLKScoringData<Real> const& score_dat,
    int cp_separation,
    TypeParamsView type_params) {
  if constexpr (D == Device::CUDA) {
    if (cp_separation < 4) return {0.0, 0.0, 0.0};
  }

  using Real3 = tmol::portable::Vec<Real, 3>;

  Real3 coord1 = coord_from_shared(score_dat.r1.coords, atom_tile_ind1);
  Real3 coord2 = coord_from_shared(score_dat.r2.coords, atom_tile_ind2);
  Real3 const delta = coord1 - coord2;
  Real const dist2 = delta.squaredNorm();
  Real const max_dis = score_dat.global_params.max_dis;
  if (dist2 > max_dis * max_dis) {
    return {0.0, 0.0, 0.0};
  }
  Real const dist = tmol::portable::math::sqrt(dist2);
  LJLKTypeParams<Real> const p1 =
      type_params[score_dat.r1.atom_types[atom_tile_ind1]];
  LJLKTypeParams<Real> const p2 =
      type_params[score_dat.r2.atom_types[atom_tile_ind2]];

  auto const lj = lj_score<Real>::V(
      dist,
      cp_separation,
      p1.lj_params(),
      p2.lj_params(),
      score_dat.global_params);
  Real const lk = (!p1.is_hydrogen && !p2.is_hydrogen)
                      ? lk_isotropic_score<Real>::V(
                            dist,
                            cp_separation,
                            p1.lk_params(),
                            p2.lk_params(),
                            score_dat.global_params)
                      : Real(0);
  return {lj[0], lj[1], lk};
}

// Weighted LJ/LK derivatives. Heavy-atom pairs reuse the LJ traversal,
// coordinate loads, distance, and count-pair separation for LK.
template <typename Real, tmol::Device D,
    typename DvDcoordsView,
    typename TypeParamsView>
TMOL_DEVICE_FUNC void ljlk_atom_derivs(
    int atom_tile_ind1,
    int atom_tile_ind2,
    int start_atom1,
    int start_atom2,
    TMOL_THREAD LJLKScoringData<Real> const& score_dat,
    int cp_separation,
    Real dTdV_atr,
    Real dTdV_rep,
    Real dTdV_lk,
    DvDcoordsView dV_dcoords,
    TypeParamsView type_params) {
  if constexpr (D == Device::CUDA) {
    if (cp_separation < 4) return;
  }

  using Real3 = tmol::portable::Vec<Real, 3>;

  Real3 const coord1 = coord_from_shared(score_dat.r1.coords, atom_tile_ind1);
  Real3 const coord2 = coord_from_shared(score_dat.r2.coords, atom_tile_ind2);
  Real3 const delta = coord1 - coord2;
  Real const dist2 = delta.squaredNorm();
  Real const max_dis = score_dat.global_params.max_dis;
  if (dist2 >= max_dis * max_dis) return;
  Real const dist = tmol::portable::math::sqrt(dist2);
  Real3 unit_delta({0.0, 0.0, 0.0});
  if (dist != 0) unit_delta = delta / dist;

  LJLKTypeParams<Real> const p1 =
      type_params[score_dat.r1.atom_types[atom_tile_ind1]];
  LJLKTypeParams<Real> const p2 =
      type_params[score_dat.r2.atom_types[atom_tile_ind2]];
  auto const lj = lj_score<Real>::V_dV(
      dist,
      cp_separation,
      p1.lj_params(),
      p2.lj_params(),
      score_dat.global_params);

  Real weighted_dV_ddist =
      dTdV_atr * lj.dVatr_ddist + dTdV_rep * lj.dVrep_ddist;
  if (!p1.is_hydrogen && !p2.is_hydrogen) {
    auto const lk = lk_isotropic_score<Real>::V_dV(
        dist,
        cp_separation,
        p1.lk_params(),
        p2.lk_params(),
        score_dat.global_params);
    weighted_dV_ddist += dTdV_lk * lk.dV_ddist;
  }

  Real3 const dxyz_at1 = weighted_dV_ddist * unit_delta;
#pragma unroll
  for (int j = 0; j < 3; ++j) {
    if (dxyz_at1[j] == 0) continue;
    accumulate_grad<D, Real>::add(
        dV_dcoords[score_dat.r1.rot_coord_offset + atom_tile_ind1 + start_atom1]
                  [j],
        dxyz_at1[j]);
    accumulate_grad<D, Real>::add(
        dV_dcoords[score_dat.r2.rot_coord_offset + atom_tile_ind2 + start_atom2]
                  [j],
        -dxyz_at1[j]);
  }
}

template <typename Real, tmol::Device D,
    typename DvDcoordsView,
    typename TypeParamsView>
TMOL_DEVICE_FUNC tmol::portable::array<Real, 3> ljlk_atom_energy_and_derivs_full(
    int atom_tile_ind1,
    int atom_tile_ind2,
    int start_atom1,
    int start_atom2,
    TMOL_THREAD LJLKScoringData<Real> const& score_dat,
    int cp_separation,
    DvDcoordsView dV_dcoords,
    TypeParamsView type_params) {
  if constexpr (D == Device::CUDA) {
    if (cp_separation < 4) return {0.0, 0.0, 0.0};
  }

  using Real3 = tmol::portable::Vec<Real, 3>;

  Real3 coord1 = coord_from_shared(score_dat.r1.coords, atom_tile_ind1);
  Real3 coord2 = coord_from_shared(score_dat.r2.coords, atom_tile_ind2);
  Real3 const delta = coord1 - coord2;
  Real const dist2 = delta.squaredNorm();
  Real const max_dis = score_dat.global_params.max_dis;
  if (dist2 >= max_dis * max_dis) {
    return {0.0, 0.0, 0.0};
  }
  Real const dist = tmol::portable::math::sqrt(dist2);
  Real3 ddist_dat1({0.0, 0.0, 0.0});
  if (dist != 0) {
    ddist_dat1 = delta / dist;
  }
  LJLKTypeParams<Real> const p1 =
      type_params[score_dat.r1.atom_types[atom_tile_ind1]];
  LJLKTypeParams<Real> const p2 =
      type_params[score_dat.r2.atom_types[atom_tile_ind2]];

  auto const lj = lj_score<Real>::V_dV(
      dist,
      cp_separation,
      p1.lj_params(),
      p2.lj_params(),
      score_dat.global_params);
  typename lk_isotropic_score<Real>::V_dV_t lk{0.0, 0.0};
  if (!p1.is_hydrogen && !p2.is_hydrogen) {
    lk = lk_isotropic_score<Real>::V_dV(
        dist,
        cp_separation,
        p1.lk_params(),
        p2.lk_params(),
        score_dat.global_params);
  }

  Real const radial_derivs[3] = {lj.dVatr_ddist, lj.dVrep_ddist, lk.dV_ddist};
#pragma unroll
  for (int score_type = 0; score_type < 3; ++score_type) {
    Real3 const dxyz_at1 = radial_derivs[score_type] * ddist_dat1;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      if (dxyz_at1[j] != 0) {
        accumulate_grad<D, Real>::add(
            dV_dcoords[score_type]
                      [score_dat.r1.rot_coord_offset + atom_tile_ind1
                       + start_atom1][j],
            dxyz_at1[j]);
      }
      if (dxyz_at1[j] != 0) {
        accumulate_grad<D, Real>::add(
            dV_dcoords[score_type]
                      [score_dat.r2.rot_coord_offset + atom_tile_ind2
                       + start_atom2][j],
            -dxyz_at1[j]);
      }
    }
  }

  return {lj.Vatr, lj.Vrep, lk.V};
}

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
