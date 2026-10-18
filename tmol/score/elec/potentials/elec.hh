#pragma once

#include <tmol/score/common/diamond_macros.hh>
#include <tmol/utility/portable/matrix.hh>

#include <tmol/score/common/count_pair.hh>
#include <tmol/score/common/workgroup_lane.hh>
#include <tmol/utility/portable/metal_view.hh>
#include <tmol/score/common/data_loading.hh>
#include <tmol/score/elec/potentials/params_types.hh>
#include <tmol/score/elec/potentials/potentials.hh>

namespace tmol {
namespace score {
namespace elec {
namespace potentials {

using tmol::score::common::accumulate_grad;

using tmol::portable::Vec;

template <typename Real>
class ElecSingleResData {
 public:
  int rot_ind;
  int block_type;
  int rot_coord_offset;
  int n_atoms;
  int n_conn;
  // These point into the workgroup's shared tile buffers, so they carry the
  // shared address space on backends that have one. See metal_view.hh.
  TMOL_SHARED_PTR(Real) coords;
  // Partial charges are not tiled. They sit contiguously in the block type's
  // own row of the charge table, so the tile can point at that row directly
  // instead of copying it: 256 bytes of threadgroup memory and one
  // cooperative copy per tile saved, and threadgroup memory is what bounds
  // this kernel's occupancy. The table is a few kilobytes shared by every
  // workgroup, so the reads hit cache. Coordinates still have to be tiled --
  // they are gathered from a per-pose array, not a per-block-type one.
  TMOL_DEVICE_PTR(Real) charges;
  TMOL_SHARED_PTR(unsigned char) path_dist;
};

template <typename Real>
class ElecScoringData {
 public:
  int pose_ind;
  int rot_ind1;
  int rot_ind2;
  int block_ind1;
  int block_ind2;
  ElecSingleResData<Real> r1;
  ElecSingleResData<Real> r2;
  int max_important_bond_separation;
  int min_separation;
  bool in_count_pair_striking_dist;
  TMOL_SHARED_PTR(unsigned char) conn_seps;
  ElecGlobalParams<Real> global_params;
  Real total_elec;
};

template <typename Real, int TILE_SIZE, int MAX_N_CONN>
struct ElecBlockPairSharedData {
  Real coords1[TILE_SIZE * 3];  // 768 bytes for coords
  Real coords2[TILE_SIZE * 3];
  unsigned char conn_ats1[MAX_N_CONN];  // 12 bytes
  unsigned char conn_ats2[MAX_N_CONN];
  unsigned char path_dist1[MAX_N_CONN * TILE_SIZE];  // 384 bytes
  unsigned char path_dist2[MAX_N_CONN * TILE_SIZE];
  unsigned char conn_seps[MAX_N_CONN * MAX_N_CONN];  // 144 bytes
};

// Count-pair separation for an inter-residue atom pair.
//
// Lives here rather than in elec_pose_score.impl.hh -- which it used to share
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

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    int nt,
    typename Real,
    typename CoordsView,
    typename BlockTypePartialChargeView>
void TMOL_DEVICE_FUNC elec_load_block_coords_and_charges_into_shared(
    CoordsView coords,
    BlockTypePartialChargeView block_type_partial_charge,
    int pose_ind,
    TMOL_THREAD ElecSingleResData<Real>& r_dat,
    int n_atoms_to_load,
    int start_atom,
    // Trailing lane handle: empty and omissible on CPU and CUDA, required by
    // the Metal backend. See workgroup_lane.hh.
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
  r_dat.charges = &block_type_partial_charge[r_dat.block_type][start_atom];
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    int nt,
    int TILE_SIZE,
    typename Real,
    typename Int,
    typename CoordsView,
    typename BlockTypePartialChargeView,
    typename BlockTypeRepresentativePathDistanceView>
void TMOL_DEVICE_FUNC elec_load_block_into_shared(
    CoordsView coords,
    BlockTypePartialChargeView block_type_partial_charge,
    BlockTypeRepresentativePathDistanceView
        block_type_representative_path_distance,
    int pose_ind,
    TMOL_THREAD ElecSingleResData<Real>& r_dat,
    int n_atoms_to_load,
    int start_atom,
    bool count_pair_striking_dist,
    TMOL_SHARED_PTR(unsigned char) conn_ats,
    // Trailing lane handle: empty and omissible on CPU and CUDA, required by
    // the Metal backend. See workgroup_lane.hh.
    WorkgroupLane<D> lane = {}) {
  elec_load_block_coords_and_charges_into_shared<DeviceDispatch, D, nt, Real>(
      coords,
      block_type_partial_charge,
      pose_ind,
      r_dat,
      n_atoms_to_load,
      start_atom,
      lane);

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
    DeviceDispatch<D>::template for_each_in_workgroup<nt>(
        copy_path_dists, lane);
  }
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
void TMOL_DEVICE_FUNC elec_load_tile_invariant_interres_data(
    RotCoordOffsetView rot_coord_offset,
    PoseStackNearBlocksView pose_stack_near_blocks,
    BlockTypeNInterblockBondsView block_type_n_interblock_bonds,
    BlockTypeAtomsFormingChemicalBondsView
        block_type_atoms_forming_chemical_bonds,
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
    TMOL_THREAD ElecScoringData<Real>& inter_dat,
    TMOL_SHARED_REF(ElecBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN>)
        shared_m,
    // Trailing lane handle: empty and omissible on CPU and CUDA, required by
    // the Metal backend. See workgroup_lane.hh.
    WorkgroupLane<D> lane = {}) {
  inter_dat.pose_ind = pose_ind;
  inter_dat.r1.rot_ind = rot_ind1;
  inter_dat.r2.rot_ind = rot_ind2;
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
  inter_dat.block_ind1 = block_ind1;
  inter_dat.block_ind2 = block_ind2;

  // set the pointers in inter_dat to point at the shared-memory arrays
  inter_dat.r1.coords = shared_m.coords1;
  inter_dat.r2.coords = shared_m.coords2;
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
  inter_dat.total_elec = 0;
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
    typename BlockTypePartialChargeView,
    typename BlockTypeRepresentativePathDistanceView>
void TMOL_DEVICE_FUNC elec_load_interres1_tile_data_to_shared(
    CoordsView coords,
    BlockTypePartialChargeView block_type_partial_charge,
    BlockTypeRepresentativePathDistanceView
        block_type_representative_path_distance,
    int tile_ind,
    int start_atom1,
    int n_atoms_to_load1,
    TMOL_THREAD ElecScoringData<Real>& inter_dat,
    TMOL_SHARED_REF(ElecBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN>)
        shared_m,
    // Trailing lane handle: empty and omissible on CPU and CUDA, required by
    // the Metal backend. See workgroup_lane.hh.
    WorkgroupLane<D> lane = {}) {
  elec_load_block_into_shared<DeviceDispatch, D, nt, TILE_SIZE, Real, Int>(
      coords,
      block_type_partial_charge,
      block_type_representative_path_distance,
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
    typename BlockTypePartialChargeView,
    typename BlockTypeRepresentativePathDistanceView>
void TMOL_DEVICE_FUNC elec_load_interres2_tile_data_to_shared(
    CoordsView coords,
    BlockTypePartialChargeView block_type_partial_charge,
    BlockTypeRepresentativePathDistanceView
        block_type_representative_path_distance,
    int tile_ind,
    int start_atom2,
    int n_atoms_to_load2,
    TMOL_THREAD ElecScoringData<Real>& inter_dat,
    TMOL_SHARED_REF(ElecBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN>)
        shared_m,
    // Trailing lane handle: empty and omissible on CPU and CUDA, required by
    // the Metal backend. See workgroup_lane.hh.
    WorkgroupLane<D> lane = {}) {
  elec_load_block_into_shared<DeviceDispatch, D, nt, TILE_SIZE, Real, Int>(
      coords,
      block_type_partial_charge,
      block_type_representative_path_distance,
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
void TMOL_DEVICE_FUNC elec_load_tile_invariant_intrares_data(
    RotCoordOffsetView rot_coord_offset,
    GlobalParamsView global_params,
    int const max_important_bond_separation,
    int pose_ind,
    int rot_ind1,
    int block_ind1,
    int block_type1,
    int n_atoms1,
    TMOL_THREAD ElecScoringData<Real>& intra_dat,
    TMOL_SHARED_REF(ElecBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN>)
        shared_m,
    // Trailing lane handle: empty and omissible on CPU and CUDA, required by
    // the Metal backend. See workgroup_lane.hh.
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
  intra_dat.r1.n_conn = 0;
  intra_dat.r2.n_conn = 0;

  // set the pointers in intra_dat to point at the
  // shared-memory arrays. Note that these arrays will be reset
  // later because which shared memory arrays we will use depends on
  // which tile pair we are evaluating!
  intra_dat.r1.coords = shared_m.coords1;    // depends on tile pair!
  intra_dat.r2.coords = shared_m.coords2;    // depends on tile pair!

  // these count pair arrays are not going to be used
  intra_dat.r1.path_dist = 0;
  intra_dat.r2.path_dist = 0;
  intra_dat.conn_seps = 0;

  // Final data members
  intra_dat.global_params = global_params[0];
  intra_dat.total_elec = 0;
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    int nt,
    int TILE_SIZE,
    int MAX_N_CONN,
    typename Real,
    typename CoordsView,
    typename BlockTypePartialChargeView>
void TMOL_DEVICE_FUNC elec_load_intrares1_tile_data_to_shared(
    CoordsView coords,
    BlockTypePartialChargeView block_type_partial_charge,
    int tile_ind,
    int start_atom1,
    int n_atoms_to_load1,
    TMOL_THREAD ElecScoringData<Real>& intra_dat,
    TMOL_SHARED_REF(ElecBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN>)
        shared_m,
    // Trailing lane handle: empty and omissible on CPU and CUDA, required by
    // the Metal backend. See workgroup_lane.hh.
    WorkgroupLane<D> lane = {}) {
  elec_load_block_coords_and_charges_into_shared<DeviceDispatch, D, nt, Real>(
      coords,
      block_type_partial_charge,
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
    typename CoordsView,
    typename BlockTypePartialChargeView>
void TMOL_DEVICE_FUNC elec_load_intrares2_tile_data_to_shared(
    CoordsView coords,
    BlockTypePartialChargeView block_type_partial_charge,
    int tile_ind,
    int start_atom2,
    int n_atoms_to_load2,
    TMOL_THREAD ElecScoringData<Real>& intra_dat,
    TMOL_SHARED_REF(ElecBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN>)
        shared_m,
    // Trailing lane handle: empty and omissible on CPU and CUDA, required by
    // the Metal backend. See workgroup_lane.hh.
    WorkgroupLane<D> lane = {}) {
  // A prior same-tile elec_load_intrares_data_from_shared call may have
  // aliased r2's pointers to the "1" shared-memory arrays. Reset them to
  // the "2" arrays so the load below writes to the correct destination.
  intra_dat.r2.coords = shared_m.coords2;
  elec_load_block_coords_and_charges_into_shared<DeviceDispatch, D, nt, Real>(
      coords,
      block_type_partial_charge,
      intra_dat.pose_ind,
      intra_dat.r2,
      n_atoms_to_load2,
      start_atom2,
      lane);
}

template <int TILE_SIZE, int MAX_N_CONN, typename Real>
void TMOL_DEVICE_FUNC elec_load_intrares_data_from_shared(
    int tile_ind1,
    int tile_ind2,
    TMOL_SHARED_REF(ElecBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN>)
        shared_m,
    TMOL_THREAD ElecScoringData<Real>& intra_dat) {
  // set the pointers in intra_dat to point at the shared-memory arrays
  // If we are evaluating the energies between atoms in the same tile
  // then only the "1" shared-memory arrays will be loaded with data;
  // we will point the "2" memory pointers at the "1" arrays
  bool same_tile = tile_ind1 == tile_ind2;
  intra_dat.r1.coords = shared_m.coords1;
  intra_dat.r2.coords = (same_tile ? shared_m.coords1 : shared_m.coords2);
  // Charges are not tiled, so there is no "2" array to alias; when both
  // halves are the same tile, nothing loaded r2's charge pointer and it has
  // to be pointed at r1's. Mirrors the coordinate aliasing above.
  if (same_tile) {
    intra_dat.r2.charges = intra_dat.r1.charges;
  }
}

template <typename Real>
TMOL_DEVICE_FUNC Real elec_atom_energy(
    int atom_tile_ind1,
    int atom_tile_ind2,
    TMOL_THREAD ElecScoringData<Real> const& score_dat,
    int cp_separation) {
  using Real3 = tmol::portable::Vec<Real, 3>;

  Real3 coord1 =
      coord_from_shared<Real, Real3>(score_dat.r1.coords, atom_tile_ind1);
  Real3 coord2 =
      coord_from_shared<Real, Real3>(score_dat.r2.coords, atom_tile_ind2);

  Real const q1 = score_dat.r1.charges[atom_tile_ind1];
  Real const q2 = score_dat.r2.charges[atom_tile_ind2];
  Real const dist = distance<Real>::V(coord1, coord2);
  Real const result =
      elec(dist, q1, q2, Real(cp_separation), score_dat.global_params);
  return result;
}

template <typename Real, tmol::Device D, typename DvDcoordsView>
TMOL_DEVICE_FUNC void elec_atom_derivs(
    int atom_tile_ind1,
    int atom_tile_ind2,
    int start_atom1,
    int start_atom2,
    TMOL_THREAD ElecScoringData<Real> const& score_dat,
    int cp_separation,
    Real dTdV,
    DvDcoordsView dV_dcoords) {
  using Real3 = tmol::portable::Vec<Real, 3>;

  Real3 coord1 =
      coord_from_shared<Real, Real3>(score_dat.r1.coords, atom_tile_ind1);
  Real3 coord2 =
      coord_from_shared<Real, Real3>(score_dat.r2.coords, atom_tile_ind2);

  auto dist_r = distance<Real>::V_dV(coord1, coord2);
  TMOL_THREAD auto& dist = dist_r.V;
  TMOL_THREAD auto& ddist_dat1 = dist_r.dV_dA;
  Real V(0.0), dV_ddist(0.0);
  tie(V, dV_ddist) = elec_delec_ddist(
      dist,
      score_dat.r1.charges[atom_tile_ind1],
      score_dat.r2.charges[atom_tile_ind2],
      Real(cp_separation),
      score_dat.global_params);

  // all threads accumulate derivatives for atom 1 to global memory
  // Remove "0.5 *" as the energy is no longer split between the i,j and j,i
  // pair
  Vec<Real, 3> elec_dxyz_at1 = dTdV * dV_ddist * ddist_dat1;
  for (int j = 0; j < 3; ++j) {
    if (elec_dxyz_at1[j] != 0) {
      accumulate_grad<D, Real>::add(
          dV_dcoords[0]
                    [score_dat.r1.rot_coord_offset + atom_tile_ind1
                     + start_atom1][j],
          elec_dxyz_at1[j]);
    }
  }

  // all threads accumulate derivatives for atom 2 to global memory
  Vec<Real, 3> elec_dxyz_at2 = -elec_dxyz_at1;
  for (int j = 0; j < 3; ++j) {
    if (elec_dxyz_at2[j] != 0) {
      accumulate_grad<D, Real>::add(
          dV_dcoords[0]
                    [score_dat.r2.rot_coord_offset + atom_tile_ind2
                     + start_atom2][j],
          elec_dxyz_at2[j]);
    }
  }
}

template <typename Real, tmol::Device D, typename DvDcoordsView>
TMOL_DEVICE_FUNC Real elec_atom_energy_and_derivs_full(
    int atom_tile_ind1,
    int atom_tile_ind2,
    int start_atom1,
    int start_atom2,
    TMOL_THREAD ElecScoringData<Real> const& score_dat,
    int cp_separation,
    DvDcoordsView dV_dcoords) {
  using Real3 = tmol::portable::Vec<Real, 3>;

  Real3 coord1 =
      coord_from_shared<Real, Real3>(score_dat.r1.coords, atom_tile_ind1);
  Real3 coord2 =
      coord_from_shared<Real, Real3>(score_dat.r2.coords, atom_tile_ind2);

  auto dist_r = distance<Real>::V_dV(coord1, coord2);
  TMOL_THREAD auto& dist = dist_r.V;
  TMOL_THREAD auto& ddist_dat1 = dist_r.dV_dA;
  Real V(0.0), dV_ddist(0.0);
  tie(V, dV_ddist) = elec_delec_ddist(
      dist,
      score_dat.r1.charges[atom_tile_ind1],
      score_dat.r2.charges[atom_tile_ind2],
      Real(cp_separation),
      score_dat.global_params);

  // all threads accumulate derivatives for atom 1 to global memory
  Vec<Real, 3> elec_dxyz_at1 = dV_ddist * ddist_dat1;
  for (int j = 0; j < 3; ++j) {
    if (elec_dxyz_at1[j] != 0) {
      accumulate_grad<D, Real>::add(
          dV_dcoords[0]
                    [score_dat.r1.rot_coord_offset + atom_tile_ind1
                     + start_atom1][j],
          elec_dxyz_at1[j]);
    }
  }

  // all threads accumulate derivatives for atom 2 to global memory
  Vec<Real, 3> elec_dxyz_at2 = -elec_dxyz_at1;
  for (int j = 0; j < 3; ++j) {
    if (elec_dxyz_at2[j] != 0) {
      accumulate_grad<D, Real>::add(
          dV_dcoords[0]
                    [score_dat.r2.rot_coord_offset + atom_tile_ind2
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
