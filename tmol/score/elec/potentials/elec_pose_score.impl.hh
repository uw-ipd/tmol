#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/count_pair.hh>
#include <tmol/score/common/data_loading.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/geom.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/common/sphere_overlap.impl.hh>
#include <tmol/score/common/tile_atom_pair_evaluation.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/upper_triangle_indices.hh>
#include <tmol/score/common/warp_segreduce.hh>
#include <tmol/score/common/warp_stride_reduce.hh>

#include <tmol/score/elec/potentials/elec.hh>
#include <tmol/score/elec/potentials/params.hh>
#include <tmol/score/elec/potentials/elec_pose_score.hh>

// Operator definitions; safe for CPU compilation
#include <moderngpu/operators.hxx>

#include <chrono>

// The maximum number of inter-residue chemical bonds
#define MAX_N_CONN 4
#define TILE_SIZE 32

namespace tmol {
namespace score {
namespace elec {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <int TILE, template <typename> typename InterEnergyData, typename Real>
EIGEN_DEVICE_FUNC int interres_count_pair_separation(
    InterEnergyData<Real> const& inter_dat,
    int atom_tile_ind1,
    int atom_tile_ind2) {
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
  return separation;
}

// MACROS

#define SCORE_INTER_ELEC_ATOM_PAIR                              \
  TMOL_DEVICE_FUNC(                                             \
      int start_atom1,                                          \
      int start_atom2,                                          \
      int atom_tile_ind1,                                       \
      int atom_tile_ind2,                                       \
      ElecScoringData<Real> const& inter_dat)                   \
      ->std::array<Real, 1> {                                   \
    int separation = interres_count_pair_separation<TILE_SIZE>( \
        inter_dat, atom_tile_ind1, atom_tile_ind2);             \
    Real elec = elec_atom_energy_and_derivs(                    \
        atom_tile_ind1,                                         \
        atom_tile_ind2,                                         \
        start_atom1,                                            \
        start_atom2,                                            \
        inter_dat,                                              \
        separation);                                            \
    return {elec};                                              \
  }

#define SCORE_INTRA_ELEC_ATOM_PAIR                                   \
  TMOL_DEVICE_FUNC(                                                  \
      int start_atom1,                                               \
      int start_atom2,                                               \
      int atom_tile_ind1,                                            \
      int atom_tile_ind2,                                            \
      ElecScoringData<Real> const& intra_dat)                        \
      ->std::array<Real, 1> {                                        \
    int const atom_ind1 = start_atom1 + atom_tile_ind1;              \
    int const atom_ind2 = start_atom2 + atom_tile_ind2;              \
    int const separation =                                           \
        block_type_intra_repr_path_distance[intra_dat.r1.block_type] \
                                           [atom_ind1][atom_ind2];   \
    Real elec = elec_atom_energy_and_derivs(                         \
        atom_tile_ind1,                                              \
        atom_tile_ind2,                                              \
        start_atom1,                                                 \
        start_atom2,                                                 \
        intra_dat,                                                   \
        separation);                                                 \
    return {elec};                                                   \
  }

#define LOAD_BLOCK_COORDS_AND_PARAMS_INTO_SHARED                           \
  TMOL_DEVICE_FUNC(                                                        \
      int pose_ind,                                                        \
      ElecSingleResData<Real>& r_dat,                                      \
      int n_atoms_to_load,                                                 \
      int start_atom) {                                                    \
    elec_load_block_coords_and_charges_into_shared<DeviceDispatch, D, nt>( \
        rot_coords,                                                        \
        block_type_partial_charge,                                         \
        pose_ind,                                                          \
        r_dat,                                                             \
        n_atoms_to_load,                                                   \
        start_atom);                                                       \
  }

#define LOAD_BLOCK_INTO_SHARED                                     \
  TMOL_DEVICE_FUNC(                                                \
      int pose_ind,                                                \
      ElecSingleResData<Real>& r_dat,                              \
      int n_atoms_to_load,                                         \
      int start_atom,                                              \
      bool count_pair_striking_dist,                               \
      unsigned char* __restrict__ conn_ats) {                      \
    elec_load_block_into_shared<DeviceDispatch, D, nt, TILE_SIZE>( \
        rot_coords,                                                \
        block_type_partial_charge,                                 \
        block_type_inter_repr_path_distance,                       \
        pose_ind,                                                  \
        r_dat,                                                     \
        n_atoms_to_load,                                           \
        start_atom,                                                \
        count_pair_striking_dist,                                  \
        conn_ats);                                                 \
  }

#define LOAD_TILE_INVARIANT_INTERRES_DATA                          \
  TMOL_DEVICE_FUNC(                                                \
      int pose_ind,                                                \
      int rot_ind1,                                                \
      int rot_ind2,                                                \
      int block_ind1,                                              \
      int block_ind2,                                              \
      int block_type1,                                             \
      int block_type2,                                             \
      int n_atoms1,                                                \
      int n_atoms2,                                                \
      ElecScoringData<Real>& inter_dat,                            \
      shared_mem_union& shared) {                                  \
    elec_load_tile_invariant_interres_data<DeviceDispatch, D, nt>( \
        rot_coord_offset,                                          \
        pose_stack_min_bond_separation,                            \
        block_type_n_interblock_bonds,                             \
        block_type_atoms_forming_chemical_bonds,                   \
        pose_stack_inter_block_bondsep,                            \
        global_params,                                             \
        max_important_bond_separation,                             \
        pose_ind,                                                  \
        rot_ind1,                                                  \
        rot_ind2,                                                  \
        block_ind1,                                                \
        block_ind2,                                                \
        block_type1,                                               \
        block_type2,                                               \
        n_atoms1,                                                  \
        n_atoms2,                                                  \
        inter_dat,                                                 \
        shared.m);                                                 \
  }

#define LOAD_INTERRES1_TILE_DATA_TO_SHARED                          \
  TMOL_DEVICE_FUNC(                                                 \
      int tile_ind,                                                 \
      int start_atom1,                                              \
      int n_atoms_to_load1,                                         \
      ElecScoringData<Real>& inter_dat,                             \
      shared_mem_union& shared) {                                   \
    elec_load_interres1_tile_data_to_shared<DeviceDispatch, D, nt>( \
        rot_coords,                                                 \
        block_type_partial_charge,                                  \
        block_type_inter_repr_path_distance,                        \
        tile_ind,                                                   \
        start_atom1,                                                \
        n_atoms_to_load1,                                           \
        inter_dat,                                                  \
        shared.m);                                                  \
  }

#define LOAD_INTERRES2_TILE_DATA_TO_SHARED                          \
  TMOL_DEVICE_FUNC(                                                 \
      int tile_ind,                                                 \
      int start_atom2,                                              \
      int n_atoms_to_load2,                                         \
      ElecScoringData<Real>& inter_dat,                             \
      shared_mem_union& shared) {                                   \
    elec_load_interres2_tile_data_to_shared<DeviceDispatch, D, nt>( \
        rot_coords,                                                 \
        block_type_partial_charge,                                  \
        block_type_inter_repr_path_distance,                        \
        tile_ind,                                                   \
        start_atom2,                                                \
        n_atoms_to_load2,                                           \
        inter_dat,                                                  \
        shared.m);                                                  \
  }

#define LOAD_INTERRES_DATA_FROM_SHARED \
  TMOL_DEVICE_FUNC(int, int, shared_mem_union&, ElecScoringData<Real>&) {}

#define EVAL_INTERRES_ATOM_PAIR_SCORES                                      \
  TMOL_DEVICE_FUNC(                                                         \
      ElecScoringData<Real>& inter_dat, int start_atom1, int start_atom2) { \
    auto eval_scores_for_atom_pairs = ([&](int tid) {                       \
      auto elecE = tmol::score::common::InterResBlockEvaluation<            \
          ElecScoringData,                                                  \
          AllAtomPairSelector,                                              \
          D,                                                                \
          TILE_SIZE,                                                        \
          nt,                                                               \
          1,                                                                \
          Real,                                                             \
          Int>::                                                            \
          eval_interres_atom_pair(                                          \
              tid,                                                          \
              start_atom1,                                                  \
              start_atom2,                                                  \
              score_inter_elec_atom_pair,                                   \
              inter_dat);                                                   \
      inter_dat.total_elec += std::get<0>(elecE);                           \
    });                                                                     \
    DeviceDispatch<D>::template for_each_in_workgroup<nt>(                  \
        eval_scores_for_atom_pairs);                                        \
  }

#define STORE_CALCULATED_POSE_ENERGIES                                      \
  TMOL_DEVICE_FUNC(                                                         \
      ElecScoringData<Real>& score_dat, shared_mem_union& shared) {         \
    auto reduce_energies = ([&](int tid) {                                  \
      Real const cta_total_elec =                                           \
          DeviceDispatch<D>::template reduce_in_workgroup<nt>(              \
              score_dat.total_elec, shared, mgpu::plus_t<Real>());          \
      if (tid == 0) {                                                       \
        if (!output_block_pair_energies) {                                  \
          accumulate<D, Real>::add(                                         \
              output[0][score_dat.pose_ind][0][0], cta_total_elec);         \
        } else {                                                            \
          int const b1 = score_dat.block_ind1;                              \
          int const b2 = score_dat.block_ind2;                              \
          output[0][cta][b1][b2] = cta_total_elec;                          \
        }                                                                   \
      }                                                                     \
    });                                                                     \
    DeviceDispatch<D>::template for_each_in_workgroup<nt>(reduce_energies); \
  }

#define STORE_CALCULATED_ROTAMER_ENERGIES                                   \
  TMOL_DEVICE_FUNC(                                                         \
      ElecScoringData<Real>& score_dat, shared_mem_union& shared) {         \
    auto reduce_energies = ([&](int tid) {                                  \
      Real const cta_total_elec =                                           \
          DeviceDispatch<D>::template reduce_in_workgroup<nt>(              \
              score_dat.total_elec, shared, mgpu::plus_t<Real>());          \
      if (tid == 0) {                                                       \
        output[0][cta] = cta_total_elec;                                    \
      }                                                                     \
    });                                                                     \
    DeviceDispatch<D>::template for_each_in_workgroup<nt>(reduce_energies); \
  }

#define LOAD_TILE_INVARIANT_INTRARES_DATA                          \
  TMOL_DEVICE_FUNC(                                                \
      int pose_ind,                                                \
      int rot_ind1,                                                \
      int block_ind1,                                              \
      int block_type1,                                             \
      int n_atoms1,                                                \
      ElecScoringData<Real>& intra_dat,                            \
      shared_mem_union& shared) {                                  \
    elec_load_tile_invariant_intrares_data<DeviceDispatch, D, nt>( \
        rot_coord_offset,                                          \
        global_params,                                             \
        max_important_bond_separation,                             \
        pose_ind,                                                  \
        rot_ind1,                                                  \
        block_ind1,                                                \
        block_type1,                                               \
        n_atoms1,                                                  \
        intra_dat,                                                 \
        shared.m);                                                 \
  }

#define LOAD_INTRARES1_TILE_DATA_TO_SHARED                          \
  TMOL_DEVICE_FUNC(                                                 \
      int tile_ind,                                                 \
      int start_atom1,                                              \
      int n_atoms_to_load1,                                         \
      ElecScoringData<Real>& intra_dat,                             \
      shared_mem_union& shared) {                                   \
    elec_load_intrares1_tile_data_to_shared<DeviceDispatch, D, nt>( \
        rot_coords,                                                 \
        block_type_partial_charge,                                  \
        tile_ind,                                                   \
        start_atom1,                                                \
        n_atoms_to_load1,                                           \
        intra_dat,                                                  \
        shared.m);                                                  \
  }

#define LOAD_INTRARES2_TILE_DATA_TO_SHARED                          \
  TMOL_DEVICE_FUNC(                                                 \
      int tile_ind,                                                 \
      int start_atom2,                                              \
      int n_atoms_to_load2,                                         \
      ElecScoringData<Real>& intra_dat,                             \
      shared_mem_union& shared) {                                   \
    elec_load_intrares2_tile_data_to_shared<DeviceDispatch, D, nt>( \
        rot_coords,                                                 \
        block_type_partial_charge,                                  \
        tile_ind,                                                   \
        start_atom2,                                                \
        n_atoms_to_load2,                                           \
        intra_dat,                                                  \
        shared.m);                                                  \
  }

#define LOAD_INTRARES_DATA_FROM_SHARED              \
  TMOL_DEVICE_FUNC(                                 \
      int tile_ind1,                                \
      int tile_ind2,                                \
      shared_mem_union& shared,                     \
      ElecScoringData<Real>& intra_dat) {           \
    elec_load_intrares_data_from_shared(            \
        tile_ind1, tile_ind2, shared.m, intra_dat); \
  }

#define EVAL_INTRARES_ATOM_PAIR_SCORES                                      \
  TMOL_DEVICE_FUNC(                                                         \
      ElecScoringData<Real>& intra_dat, int start_atom1, int start_atom2) { \
    auto eval_scores_for_atom_pairs = ([&](int tid) {                       \
      auto elecE = tmol::score::common::IntraResBlockEvaluation<            \
          ElecScoringData,                                                  \
          AllAtomPairSelector,                                              \
          D,                                                                \
          TILE_SIZE,                                                        \
          nt,                                                               \
          1,                                                                \
          Real,                                                             \
          Int>::                                                            \
          eval_intrares_atom_pairs(                                         \
              tid,                                                          \
              start_atom1,                                                  \
              start_atom2,                                                  \
              score_intra_elec_atom_pair,                                   \
              intra_dat);                                                   \
      intra_dat.total_elec += std::get<0>(elecE);                           \
    });                                                                     \
    DeviceDispatch<D>::template for_each_in_workgroup<nt>(                  \
        eval_scores_for_atom_pairs);                                        \
  }

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto ElecPoseScoreDispatch<DeviceDispatch, D, Real, Int>::forward(
    // common params
    TView<Vec<Real, 3>, 1, D> rot_coords,
    TView<Int, 1, D> rot_coord_offset,
    TView<Int, 1, D> pose_ind_for_atom,
    TView<Int, 2, D> first_rot_for_block,
    TView<Int, 2, D> first_rot_block_type,
    TView<Int, 1, D> block_ind_for_rot,
    TView<Int, 1, D> pose_ind_for_rot,
    TView<Int, 1, D> block_type_ind_for_rot,
    TView<Int, 1, D> n_rots_for_pose,
    TView<Int, 1, D> rot_offset_for_pose,
    TView<Int, 2, D> n_rots_for_block,
    TView<Int, 2, D> rot_offset_for_block,
    Int max_n_rots_per_pose,

    // dims: n-poses x max-n-blocks x max-n-blocks
    // Quick lookup: given the inds of two blocks, ask: what is the minimum
    // number of chemical bonds that separate any pair of atoms in those
    // blocks? If this minimum is greater than the crossover, then no further
    // logic for deciding whether two atoms in those blocks should have their
    // interaction energies calculated: all should. intentionally small to
    // (possibly) fit in constant cache
    TView<Int, 3, D> pose_stack_min_bond_separation,

    // dims: n-poses x max-n-blocks x max-n-blocks x
    // max-n-interblock-connections x max-n-interblock-connections
    TView<Int, 5, D> pose_stack_inter_block_bondsep,

    //////////////////////
    // Chemical properties
    // how many atoms for a given block
    // Dimsize n_block_types
    TView<Int, 1, D> block_type_n_atoms,

    TView<Real, 2, D> block_type_partial_charge,

    // how many inter-block chemical bonds are there
    // Dimsize: n_block_types
    TView<Int, 1, D> block_type_n_interblock_bonds,

    // what atoms form the inter-block chemical bonds
    // Dimsize: n_block_types x max_n_interblock_bonds
    TView<Int, 2, D> block_type_atoms_forming_chemical_bonds,

    // what is the path distance between pairs of atoms in the block
    // denormalized by their count-pair representative; used for
    // inter-block chemical-bond separation determination. Entry
    // i, j stores path_dist[i, rep(j)]
    // Dimsize: n_block_types x max_n_atoms x max_n_atoms
    TView<Int, 3, D> block_type_inter_repr_path_distance,

    // what is the path distance between pairs of atoms in the block
    // denormalized (twice!) by their count-pair representative;
    // used for intra-block chemical-bond separation determination.
    // Entry i, j stores path_dist[rep(i), rep(j)]
    // Dimsize: n_block_types x max_n_atoms x max_n_atoms
    TView<Int, 3, D> block_type_intra_repr_path_distance,
    //////////////////////

    // LJ parameters
    TView<ElecGlobalParams<Real>, 1, D> global_params,
    bool output_block_pair_energies,
    bool compute_derivs) -> std::
    tuple<TPack<Real, 4, D>, TPack<Vec<Real, 3>, 2, D>, TPack<Int, 3, D> > {
  using tmol::score::common::accumulate;
  using Real3 = Vec<Real, 3>;

  int const n_atoms = rot_coords.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const n_rots = rot_coord_offset.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);

  // int const max_n_pose_atoms = coords.size(1);
  int const max_n_block_atoms = block_type_partial_charge.size(1);
  int const n_block_types = block_type_n_atoms.size(0);
  int const max_n_interblock_bonds =
      block_type_atoms_forming_chemical_bonds.size(1);

  assert(max_n_interblock_bonds <= MAX_N_CONN);

  assert(first_rot_block_type.size(0) == n_poses);
  assert(first_rot_block_type.size(1) == max_n_blocks);

  assert(block_ind_for_rot.size(0) == n_rots);
  assert(pose_ind_for_rot.size(0) == n_rots);
  assert(block_type_ind_for_rot.size(0) == n_rots);

  assert(n_rots_for_pose.size(0) == n_poses);
  assert(rot_offset_for_pose.size(0) == n_poses);

  assert(n_rots_for_block.size(0) == n_poses);
  assert(n_rots_for_block.size(1) == max_n_blocks);

  assert(rot_offset_for_block.size(0) == n_poses);
  assert(rot_offset_for_block.size(1) == max_n_blocks);

  assert(pose_stack_min_bond_separation.size(0) == n_poses);
  assert(pose_stack_min_bond_separation.size(1) == max_n_blocks);
  assert(pose_stack_min_bond_separation.size(2) == max_n_blocks);

  assert(pose_stack_inter_block_bondsep.size(0) == n_poses);
  assert(pose_stack_inter_block_bondsep.size(1) == max_n_blocks);
  assert(pose_stack_inter_block_bondsep.size(2) == max_n_blocks);
  assert(pose_stack_inter_block_bondsep.size(3) == max_n_interblock_bonds);
  assert(pose_stack_inter_block_bondsep.size(4) == max_n_interblock_bonds);

  assert(block_type_n_interblock_bonds.size(0) == n_block_types);

  assert(block_type_atoms_forming_chemical_bonds.size(0) == n_block_types);

  assert(block_type_inter_repr_path_distance.size(0) == n_block_types);
  assert(block_type_inter_repr_path_distance.size(1) == max_n_block_atoms);
  assert(block_type_inter_repr_path_distance.size(2) == max_n_block_atoms);
  assert(block_type_intra_repr_path_distance.size(0) == n_block_types);
  assert(block_type_intra_repr_path_distance.size(1) == max_n_block_atoms);
  assert(block_type_intra_repr_path_distance.size(2) == max_n_block_atoms);

  auto dV_dcoords_t = TPack<Vec<Real, 3>, 2, D>::zeros({1, n_atoms});
  auto dV_dcoords = dV_dcoords_t.view;

  auto scratch_rot_spheres_t =
      TPack<Real, 3, D>::zeros({n_poses, max_n_rots_per_pose, 4});
  auto scratch_rot_spheres = scratch_rot_spheres_t.view;

  auto scratch_rot_neighbors_t = TPack<Int, 3, D>::zeros(
      {n_poses, max_n_rots_per_pose, max_n_rots_per_pose});
  auto scratch_rot_neighbors = scratch_rot_neighbors_t.view;

  // printf("starting!\n");

  score::common::sphere_overlap::
      compute_block_spheres<DeviceDispatch, D, Real, Int>::f(
          rot_coords,
          rot_coord_offset,
          block_ind_for_rot,
          pose_ind_for_rot,
          block_type_ind_for_rot,
          block_type_n_atoms,
          scratch_rot_spheres);
  // printf("computed block spheres\n");

  score::common::sphere_overlap::
      detect_block_neighbors<DeviceDispatch, D, Real, Int>::f(
          first_rot_block_type,
          scratch_rot_spheres,
          scratch_rot_neighbors,
          Real(5.5));  // 5.5A hard coded here. Please fix! TEMP!
  // printf("computed block neighbors\n");

  // auto output_t = TPack<Real, 2, D>::zeros({1, n_poses});
  //  auto output_t =
  //      TPack<Real, 4, D>::zeros({3, n_poses, max_n_blocks, max_n_blocks});
  TPack<Real, 4, D> output_t;
  if (output_block_pair_energies) {
    output_t =
        TPack<Real, 4, D>::zeros({1, n_poses, max_n_blocks, max_n_blocks});
  } else {
    output_t = TPack<Real, 4, D>::zeros({1, n_poses, 1, 1});
  }
  auto output = output_t.view;

  // Optimal launch box on v100 and a100 is nt=32, vt=1
  LAUNCH_BOX_32;
  // Define nt and reduce_t
  CTA_REAL_REDUCE_T_TYPEDEF;

  // The total number of unique block pairs (including self-pairs)
  int const max_n_upper_triangle_inds = (max_n_blocks * (max_n_blocks + 1)) / 2;

  auto eval_energies_by_block = ([=] TMOL_DEVICE_FUNC(int cta) {
    auto elec_atom_energy_and_derivs =
        ([=] TMOL_DEVICE_FUNC(
             int atom_tile_ind1,
             int atom_tile_ind2,
             int,
             int,
             ElecScoringData<Real> const& score_dat,
             int cp_separation) {
          return elec_atom_energy(
              atom_tile_ind1, atom_tile_ind2, score_dat, cp_separation);
        });

    auto score_inter_elec_atom_pair = ([=] SCORE_INTER_ELEC_ATOM_PAIR);

    auto score_intra_elec_atom_pair = ([=] SCORE_INTRA_ELEC_ATOM_PAIR);

    auto load_block_coords_and_params_into_shared =
        ([=] LOAD_BLOCK_COORDS_AND_PARAMS_INTO_SHARED);

    auto load_block_into_shared = ([=] LOAD_BLOCK_INTO_SHARED);

    SHARED_MEMORY union shared_mem_union {
      shared_mem_union() {}
      ElecBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN> m;
      CTA_REAL_REDUCE_T_VARIABLE;

    } shared;

    int const max_important_bond_separation = 4;

    int const pose_ind = cta / (max_n_upper_triangle_inds);
    int const block_ind_pair = cta % (max_n_upper_triangle_inds);

    auto upper_triangle_ind = common::upper_triangle_inds_from_linear_index(
        block_ind_pair, max_n_blocks + 1);
    int const block_ind1 = common::get<0>(upper_triangle_ind);
    int const block_ind2 = common::get<1>(upper_triangle_ind) - 1;
    // printf("eval_energies_by_block cta %d pose_ind %d block_ind1 %d
    // block_ind2 %d\n", cta, pose_ind, block_ind1, block_ind2);

    // We do not have to kill half of our thread blocks simply because they
    // represent the lower triangle now that we're using upper-triangle indices
    // if (block_ind1 > block_ind2) {
    //   return;
    // }

    // But we still kill everything that involves non-neighboring blocks
    // and that can be a lot!
    if (scratch_rot_neighbors[pose_ind][block_ind1][block_ind2] == 0) {
      return;
    }

    int const rot_ind1 = rot_offset_for_block[pose_ind][block_ind1];
    int const rot_ind2 = rot_offset_for_block[pose_ind][block_ind2];

    if (rot_ind1 < 0 || rot_ind2 < 0) {
      return;
    }

    int const block_type1 = block_type_ind_for_rot[rot_ind1];
    int const block_type2 = block_type_ind_for_rot[rot_ind2];

    if (block_type1 < 0 || block_type2 < 0) {
      // I think this cannot happen
      return;
    }
    int const n_atoms1 = block_type_n_atoms[block_type1];
    int const n_atoms2 = block_type_n_atoms[block_type2];

    auto load_tile_invariant_interres_data =
        ([=] LOAD_TILE_INVARIANT_INTERRES_DATA);

    auto load_interres1_tile_data_to_shared =
        ([=] LOAD_INTERRES1_TILE_DATA_TO_SHARED);

    auto load_interres2_tile_data_to_shared =
        ([=] LOAD_INTERRES2_TILE_DATA_TO_SHARED);

    auto load_interres_data_from_shared = ([=] LOAD_INTERRES_DATA_FROM_SHARED);

    auto eval_interres_atom_pair_scores = ([=] EVAL_INTERRES_ATOM_PAIR_SCORES);

    auto store_calculated_energies = ([=] STORE_CALCULATED_POSE_ENERGIES);

    auto load_tile_invariant_intrares_data =
        ([=] LOAD_TILE_INVARIANT_INTRARES_DATA);

    auto load_intrares1_tile_data_to_shared =
        ([=] LOAD_INTRARES1_TILE_DATA_TO_SHARED);

    auto load_intrares2_tile_data_to_shared =
        ([=] LOAD_INTRARES2_TILE_DATA_TO_SHARED);

    auto load_intrares_data_from_shared = ([=] LOAD_INTRARES_DATA_FROM_SHARED);

    auto eval_intrares_atom_pair_scores = ([=] EVAL_INTRARES_ATOM_PAIR_SCORES);

    tmol::score::common::tile_evaluate_rot_pair<
        DeviceDispatch,
        D,
        ElecScoringData<Real>,
        ElecScoringData<Real>,
        Real,
        TILE_SIZE>(
        shared,
        pose_ind,
        rot_ind1,
        rot_ind2,
        block_ind1,
        block_ind2,
        block_type1,
        block_type2,
        n_atoms1,
        n_atoms2,
        load_tile_invariant_interres_data,
        load_interres1_tile_data_to_shared,
        load_interres2_tile_data_to_shared,
        load_interres_data_from_shared,
        eval_interres_atom_pair_scores,
        store_calculated_energies,
        load_tile_invariant_intrares_data,
        load_intrares1_tile_data_to_shared,
        load_intrares2_tile_data_to_shared,
        load_intrares_data_from_shared,
        eval_intrares_atom_pair_scores,
        store_calculated_energies);
  });

  auto eval_energies = ([=] TMOL_DEVICE_FUNC(int cta) {
    auto elec_atom_energy_and_derivs =
        ([=] TMOL_DEVICE_FUNC(
             int atom_tile_ind1,
             int atom_tile_ind2,
             int start_atom1,
             int start_atom2,
             ElecScoringData<Real> const& score_dat,
             int cp_separation) {
          if (compute_derivs) {
            auto val = elec_atom_energy_and_derivs_full(
                atom_tile_ind1,
                atom_tile_ind2,
                start_atom1,
                start_atom2,
                score_dat,
                cp_separation,
                dV_dcoords);
            return val;
          } else {
            return elec_atom_energy(
                atom_tile_ind1, atom_tile_ind2, score_dat, cp_separation);
          }
        });

    auto score_inter_elec_atom_pair = ([=] SCORE_INTER_ELEC_ATOM_PAIR);

    auto score_intra_elec_atom_pair = ([=] SCORE_INTRA_ELEC_ATOM_PAIR);

    auto load_block_coords_and_params_into_shared =
        ([=] LOAD_BLOCK_COORDS_AND_PARAMS_INTO_SHARED);

    auto load_block_into_shared = ([=] LOAD_BLOCK_INTO_SHARED);

    SHARED_MEMORY union shared_mem_union {
      shared_mem_union() {}
      ElecBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN> m;
      CTA_REAL_REDUCE_T_VARIABLE;

    } shared;

    int const max_important_bond_separation = 4;

    int const pose_ind = cta / (max_n_upper_triangle_inds);
    int const block_ind_pair = cta % (max_n_upper_triangle_inds);

    auto upper_triangle_ind = common::upper_triangle_inds_from_linear_index(
        block_ind_pair, max_n_blocks + 1);
    int const block_ind1 = common::get<0>(upper_triangle_ind);
    int const block_ind2 = common::get<1>(upper_triangle_ind) - 1;
    // printf("eval_energies cta %d pose_ind %d block_ind1 %d block_ind2 %d\n",
    // cta, pose_ind, block_ind1, block_ind2);

    // We do not have to kill half of our thread blocks simply because they
    // represent the lower triangle now that we're using upper-triangle indices
    // if (block_ind1 > block_ind2) {
    //   return;
    // }

    // But we still kill everything that involves non-neighboring blocks
    // and that can be a lot!
    if (scratch_rot_neighbors[pose_ind][block_ind1][block_ind2] == 0) {
      return;
    }

    int const rot_ind1 = rot_offset_for_block[pose_ind][block_ind1];
    int const rot_ind2 = rot_offset_for_block[pose_ind][block_ind2];

    if (rot_ind1 < 0 || rot_ind2 < 0) {
      return;
    }

    int const block_type1 = block_type_ind_for_rot[rot_ind1];
    int const block_type2 = block_type_ind_for_rot[rot_ind2];

    if (block_type1 < 0 || block_type2 < 0) {
      // I think this cannot happen
      return;
    }
    int const n_atoms1 = block_type_n_atoms[block_type1];
    int const n_atoms2 = block_type_n_atoms[block_type2];

    auto load_tile_invariant_interres_data =
        ([=] LOAD_TILE_INVARIANT_INTERRES_DATA);

    auto load_interres1_tile_data_to_shared =
        ([=] LOAD_INTERRES1_TILE_DATA_TO_SHARED);

    auto load_interres2_tile_data_to_shared =
        ([=] LOAD_INTERRES2_TILE_DATA_TO_SHARED);

    auto load_interres_data_from_shared = ([=] LOAD_INTERRES_DATA_FROM_SHARED);

    auto eval_interres_atom_pair_scores = ([=] EVAL_INTERRES_ATOM_PAIR_SCORES);

    auto store_calculated_energies = ([=] STORE_CALCULATED_POSE_ENERGIES);

    auto load_tile_invariant_intrares_data =
        ([=] LOAD_TILE_INVARIANT_INTRARES_DATA);

    auto load_intrares1_tile_data_to_shared =
        ([=] LOAD_INTRARES1_TILE_DATA_TO_SHARED);

    auto load_intrares2_tile_data_to_shared =
        ([=] LOAD_INTRARES2_TILE_DATA_TO_SHARED);

    auto load_intrares_data_from_shared = ([=] LOAD_INTRARES_DATA_FROM_SHARED);

    auto eval_intrares_atom_pair_scores = ([=] EVAL_INTRARES_ATOM_PAIR_SCORES);

    tmol::score::common::tile_evaluate_rot_pair<
        DeviceDispatch,
        D,
        ElecScoringData<Real>,
        ElecScoringData<Real>,
        Real,
        TILE_SIZE>(
        shared,
        pose_ind,
        rot_ind1,
        rot_ind2,
        block_ind1,
        block_ind2,
        block_type1,
        block_type2,
        n_atoms1,
        n_atoms2,
        load_tile_invariant_interres_data,
        load_interres1_tile_data_to_shared,
        load_interres2_tile_data_to_shared,
        load_interres_data_from_shared,
        eval_interres_atom_pair_scores,
        store_calculated_energies,
        load_tile_invariant_intrares_data,
        load_intrares1_tile_data_to_shared,
        load_intrares2_tile_data_to_shared,
        load_intrares_data_from_shared,
        eval_intrares_atom_pair_scores,
        store_calculated_energies);
  });

  ///////////////////////////////////////////////////////////////////////

  // Three steps
  // 0: setup
  // 1: launch a kernel to find a small bounding sphere surrounding the blocks
  // 2: launch a kernel to look for spheres that are within striking distance of
  // each other
  // 3: launch a kernel to evaluate lj/lk between pairs of blocks
  // within striking distance

  // 0
  // TO DO: let DeviceDispatch hold a cuda stream (??)
  // at::cuda::CUDAStream wrapped_stream = at::cuda::getDefaultCUDAStream();
  // mgpu::standard_context_t context(wrapped_stream.stream());

  // 3
  if (output_block_pair_energies) {
    DeviceDispatch<D>::template foreach_workgroup<launch_t>(
        n_poses * max_n_upper_triangle_inds, eval_energies_by_block);
  } else {
    DeviceDispatch<D>::template foreach_workgroup<launch_t>(
        n_poses * max_n_upper_triangle_inds, eval_energies);
  }
  // printf("computed energies\n");

  // DeviceDispatch<D>::synchronize_device();
  return {output_t, dV_dcoords_t, scratch_rot_neighbors_t};
}  // namespace potentials

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto ElecPoseScoreDispatch<DeviceDispatch, D, Real, Int>::backward(
    // common params
    TView<Vec<Real, 3>, 1, D> rot_coords,
    TView<Int, 1, D> rot_coord_offset,
    TView<Int, 1, D> pose_ind_for_atom,
    TView<Int, 2, D> first_rot_for_block,
    TView<Int, 2, D> first_rot_block_type,
    TView<Int, 1, D> block_ind_for_rot,
    TView<Int, 1, D> pose_ind_for_rot,
    TView<Int, 1, D> block_type_ind_for_rot,
    TView<Int, 1, D> n_rots_for_pose,
    TView<Int, 1, D> rot_offset_for_pose,
    TView<Int, 2, D> n_rots_for_block,
    TView<Int, 2, D> rot_offset_for_block,
    Int max_n_rots_per_pose,

    // dims: n-poses x max-n-blocks x max-n-blocks
    // Quick lookup: given the inds of two blocks, ask: what is the minimum
    // number of chemical bonds that separate any pair of atoms in those
    // blocks? If this minimum is greater than the crossover, then no further
    // logic for deciding whether two atoms in those blocks should have their
    // interaction energies calculated: all should. intentionally small to
    // (possibly) fit in constant cache
    TView<Int, 3, D> pose_stack_min_bond_separation,

    // dims: n-poses x max-n-blocks x max-n-blocks x
    // max-n-interblock-connections x max-n-interblock-connections
    TView<Int, 5, D> pose_stack_inter_block_bondsep,

    //////////////////////
    // Chemical properties
    // how many atoms for a given block
    // Dimsize n_block_types
    TView<Int, 1, D> block_type_n_atoms,

    TView<Real, 2, D> block_type_partial_charge,

    // how many inter-block chemical bonds are there
    // Dimsize: n_block_types
    TView<Int, 1, D> block_type_n_interblock_bonds,

    // what atoms form the inter-block chemical bonds
    // Dimsize: n_block_types x max_n_interblock_bonds
    TView<Int, 2, D> block_type_atoms_forming_chemical_bonds,

    // what is the path distance between pairs of atoms in the block
    // denormalized by their count-pair representative; used for
    // inter-block chemical-bond separation determination. Entry
    // i, j stores path_dist[i, rep(j)]
    // Dimsize: n_block_types x max_n_atoms x max_n_atoms
    TView<Int, 3, D> block_type_inter_repr_path_distance,

    // what is the path distance between pairs of atoms in the block
    // denormalized (twice!) by their count-pair representative;
    // used for intra-block chemical-bond separation determination.
    // Entry i, j stores path_dist[rep(i), rep(j)]
    // Dimsize: n_block_types x max_n_atoms x max_n_atoms
    TView<Int, 3, D> block_type_intra_repr_path_distance,
    //////////////////////

    // LJ parameters
    TView<ElecGlobalParams<Real>, 1, D> global_params,
    TView<Int, 3, D> scratch_rot_neighbors,

    TView<Real, 2, D> dTdV  // nterms x nposes x len x len
    ) -> TPack<Vec<Real, 3>, 2, D> {
  using tmol::score::common::accumulate;
  using Real3 = Vec<Real, 3>;

  int const n_atoms = rot_coords.size(0);
  int const n_rots = rot_coord_offset.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);

  int const n_block_types = block_type_n_atoms.size(0);
  int const max_n_block_atoms = block_type_partial_charge.size(1);
  int const max_n_interblock_bonds =
      block_type_atoms_forming_chemical_bonds.size(1);

  assert(first_rot_block_type.size(0) == n_poses);
  assert(first_rot_block_type.size(1) == max_n_blocks);

  assert(block_ind_for_rot.size(0) == n_rots);
  assert(pose_ind_for_rot.size(0) == n_rots);
  assert(block_type_ind_for_rot.size(0) == n_rots);

  assert(n_rots_for_pose.size(0) == n_poses);
  assert(rot_offset_for_pose.size(0) == n_poses);

  assert(n_rots_for_block.size(0) == n_poses);
  assert(n_rots_for_block.size(1) == max_n_blocks);

  assert(rot_offset_for_block.size(0) == n_poses);
  assert(rot_offset_for_block.size(1) == max_n_blocks);

  assert(max_n_interblock_bonds <= MAX_N_CONN);

  assert(pose_stack_min_bond_separation.size(0) == n_poses);
  assert(pose_stack_min_bond_separation.size(1) == max_n_blocks);
  assert(pose_stack_min_bond_separation.size(2) == max_n_blocks);

  assert(pose_stack_inter_block_bondsep.size(0) == n_poses);
  assert(pose_stack_inter_block_bondsep.size(1) == max_n_blocks);
  assert(pose_stack_inter_block_bondsep.size(2) == max_n_blocks);
  assert(pose_stack_inter_block_bondsep.size(3) == max_n_interblock_bonds);
  assert(pose_stack_inter_block_bondsep.size(4) == max_n_interblock_bonds);

  assert(block_type_n_interblock_bonds.size(0) == n_block_types);

  assert(block_type_atoms_forming_chemical_bonds.size(0) == n_block_types);

  assert(block_type_inter_repr_path_distance.size(0) == n_block_types);
  assert(block_type_inter_repr_path_distance.size(1) == max_n_block_atoms);
  assert(block_type_inter_repr_path_distance.size(2) == max_n_block_atoms);
  assert(block_type_intra_repr_path_distance.size(0) == n_block_types);
  assert(block_type_intra_repr_path_distance.size(1) == max_n_block_atoms);
  assert(block_type_intra_repr_path_distance.size(2) == max_n_block_atoms);

  auto dV_dcoords_t = TPack<Vec<Real, 3>, 2, D>::zeros({1, n_atoms});
  auto dV_dcoords = dV_dcoords_t.view;

  // Optimal launch box on v100 and a100 is nt=32, vt=1
  LAUNCH_BOX_32;
  // Define nt and reduce_t
  CTA_REAL_REDUCE_T_TYPEDEF;

  // The total number of unique block pairs (including self-pairs)
  int const max_n_upper_triangle_inds = (max_n_blocks * (max_n_blocks + 1)) / 2;

  auto eval_derivs = ([=] TMOL_DEVICE_FUNC(int cta) {
    auto elec_atom_energy_and_derivs =
        ([=] TMOL_DEVICE_FUNC(
             int atom_tile_ind1,
             int atom_tile_ind2,
             int start_atom1,
             int start_atom2,
             ElecScoringData<Real> const& score_dat,
             int cp_separation) {
          elec_atom_derivs(
              atom_tile_ind1,
              atom_tile_ind2,
              start_atom1,
              start_atom2,
              score_dat,
              cp_separation,
              dTdV[0][cta],
              dV_dcoords);
          return 0.0;
        });

    // TEST!
    auto score_inter_elec_atom_pair = ([=] SCORE_INTER_ELEC_ATOM_PAIR);

    auto score_intra_elec_atom_pair = ([=] SCORE_INTRA_ELEC_ATOM_PAIR);

    auto load_block_coords_and_params_into_shared =
        ([=] LOAD_BLOCK_COORDS_AND_PARAMS_INTO_SHARED);

    auto load_block_into_shared = ([=] LOAD_BLOCK_INTO_SHARED);

    SHARED_MEMORY union shared_mem_union {
      shared_mem_union() {}
      ElecBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN> m;
      CTA_REAL_REDUCE_T_VARIABLE;

    } shared;

    int const max_important_bond_separation = 4;
    int const pose_ind = cta / (max_n_upper_triangle_inds);
    int const block_ind_pair = cta % (max_n_upper_triangle_inds);

    auto upper_triangle_ind = common::upper_triangle_inds_from_linear_index(
        block_ind_pair, max_n_blocks + 1);
    int const block_ind1 = common::get<0>(upper_triangle_ind);
    int const block_ind2 = common::get<1>(upper_triangle_ind) - 1;

    int const rot_ind1 = rot_offset_for_block[pose_ind][block_ind1];
    int const rot_ind2 = rot_offset_for_block[pose_ind][block_ind2];

    if (scratch_rot_neighbors[pose_ind][block_ind1][block_ind2] == 0) {
      return;
    }

    int const block_type1 = block_type_ind_for_rot[rot_ind1];
    int const block_type2 = block_type_ind_for_rot[rot_ind2];

    if (block_type1 < 0 || block_type2 < 0) {
      return;
    }

    int const n_atoms1 = block_type_n_atoms[block_type1];
    int const n_atoms2 = block_type_n_atoms[block_type2];

    auto load_tile_invariant_interres_data =
        ([=] LOAD_TILE_INVARIANT_INTERRES_DATA);

    auto load_interres1_tile_data_to_shared =
        ([=] LOAD_INTERRES1_TILE_DATA_TO_SHARED);

    auto load_interres2_tile_data_to_shared =
        ([=] LOAD_INTERRES2_TILE_DATA_TO_SHARED);

    auto load_interres_data_from_shared = ([=] LOAD_INTERRES_DATA_FROM_SHARED);

    auto eval_interres_atom_pair_scores = ([=] EVAL_INTERRES_ATOM_PAIR_SCORES);

    auto store_calculated_energies =
        ([=](ElecScoringData<Real>& score_dat, shared_mem_union& shared) {
          ;  // no op for gradients ()
        });

    auto load_tile_invariant_intrares_data =
        ([=] LOAD_TILE_INVARIANT_INTRARES_DATA);

    auto load_intrares1_tile_data_to_shared =
        ([=] LOAD_INTRARES1_TILE_DATA_TO_SHARED);

    auto load_intrares2_tile_data_to_shared =
        ([=] LOAD_INTRARES2_TILE_DATA_TO_SHARED);

    auto load_intrares_data_from_shared = ([=] LOAD_INTRARES_DATA_FROM_SHARED);

    auto eval_intrares_atom_pair_scores = ([=] EVAL_INTRARES_ATOM_PAIR_SCORES);

    tmol::score::common::tile_evaluate_rot_pair<
        DeviceDispatch,
        D,
        ElecScoringData<Real>,
        ElecScoringData<Real>,
        Real,
        TILE_SIZE>(
        shared,
        pose_ind,
        rot_ind1,
        rot_ind2,
        block_ind1,
        block_ind2,
        block_type1,
        block_type2,
        n_atoms1,
        n_atoms2,
        load_tile_invariant_interres_data,
        load_interres1_tile_data_to_shared,
        load_interres2_tile_data_to_shared,
        load_interres_data_from_shared,
        eval_interres_atom_pair_scores,
        store_calculated_energies,
        load_tile_invariant_intrares_data,
        load_intrares1_tile_data_to_shared,
        load_intrares2_tile_data_to_shared,
        load_intrares_data_from_shared,
        eval_intrares_atom_pair_scores,
        store_calculated_energies);
  });

  // Since we have the sphere overlap results from the forward pass,
  // there's only a single kernel launch here
  DeviceDispatch<D>::template foreach_workgroup<launch_t>(
      n_poses * max_n_upper_triangle_inds, eval_derivs);

  return dV_dcoords_t;
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto ElecRotamerScoreDispatch<DeviceDispatch, D, Real, Int>::forward(
    // common params
    TView<Vec<Real, 3>, 1, D> rot_coords,
    TView<Int, 1, D> rot_coord_offset,
    TView<Int, 1, D> pose_ind_for_atom,
    TView<Int, 2, D> first_rot_for_block,
    TView<Int, 2, D> first_rot_block_type,
    TView<Int, 1, D> block_ind_for_rot,
    TView<Int, 1, D> pose_ind_for_rot,
    TView<Int, 1, D> block_type_ind_for_rot,
    TView<Int, 1, D> n_rots_for_pose,
    TView<Int, 1, D> rot_offset_for_pose,
    TView<Int, 2, D> n_rots_for_block,
    TView<Int, 2, D> rot_offset_for_block,
    Int max_n_rots_per_pose,

    // dims: n-poses x max-n-blocks x max-n-blocks
    // Quick lookup: given the inds of two blocks, ask: what is the minimum
    // number of chemical bonds that separate any pair of atoms in those
    // blocks? If this minimum is greater than the crossover, then no further
    // logic for deciding whether two atoms in those blocks should have their
    // interaction energies calculated: all should. intentionally small to
    // (possibly) fit in constant cache
    TView<Int, 3, D> pose_stack_min_bond_separation,

    // dims: n-poses x max-n-blocks x max-n-blocks x
    // max-n-interblock-connections x max-n-interblock-connections
    TView<Int, 5, D> pose_stack_inter_block_bondsep,

    //////////////////////
    // Chemical properties
    // how many atoms for a given block
    // Dimsize n_block_types
    TView<Int, 1, D> block_type_n_atoms,

    TView<Real, 2, D> block_type_partial_charge,

    // how many inter-block chemical bonds are there
    // Dimsize: n_block_types
    TView<Int, 1, D> block_type_n_interblock_bonds,

    // what atoms form the inter-block chemical bonds
    // Dimsize: n_block_types x max_n_interblock_bonds
    TView<Int, 2, D> block_type_atoms_forming_chemical_bonds,

    // what is the path distance between pairs of atoms in the block
    // denormalized by their count-pair representative; used for
    // inter-block chemical-bond separation determination. Entry
    // i, j stores path_dist[i, rep(j)]
    // Dimsize: n_block_types x max_n_atoms x max_n_atoms
    TView<Int, 3, D> block_type_inter_repr_path_distance,

    // what is the path distance between pairs of atoms in the block
    // denormalized (twice!) by their count-pair representative;
    // used for intra-block chemical-bond separation determination.
    // Entry i, j stores path_dist[rep(i), rep(j)]
    // Dimsize: n_block_types x max_n_atoms x max_n_atoms
    TView<Int, 3, D> block_type_intra_repr_path_distance,
    //////////////////////

    // LJ parameters
    TView<ElecGlobalParams<Real>, 1, D> global_params,
    bool output_block_pair_energies,
    bool compute_derivs) -> std::
    tuple<TPack<Real, 2, D>, TPack<Vec<Real, 3>, 2, D>, TPack<Int, 2, D> > {
  using tmol::score::common::accumulate;
  using Real3 = Vec<Real, 3>;

  int const n_atoms = rot_coords.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const n_rots = rot_coord_offset.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);

  // int const max_n_pose_atoms = coords.size(1);
  int const max_n_block_atoms = block_type_partial_charge.size(1);
  int const n_block_types = block_type_n_atoms.size(0);
  int const max_n_interblock_bonds =
      block_type_atoms_forming_chemical_bonds.size(1);

  assert(max_n_interblock_bonds <= MAX_N_CONN);

  assert(first_rot_block_type.size(0) == n_poses);
  assert(first_rot_block_type.size(1) == max_n_blocks);

  assert(block_ind_for_rot.size(0) == n_rots);
  assert(pose_ind_for_rot.size(0) == n_rots);
  assert(block_type_ind_for_rot.size(0) == n_rots);

  assert(n_rots_for_pose.size(0) == n_poses);
  assert(rot_offset_for_pose.size(0) == n_poses);

  assert(n_rots_for_block.size(0) == n_poses);
  assert(n_rots_for_block.size(1) == max_n_blocks);

  assert(rot_offset_for_block.size(0) == n_poses);
  assert(rot_offset_for_block.size(1) == max_n_blocks);

  assert(pose_stack_min_bond_separation.size(0) == n_poses);
  assert(pose_stack_min_bond_separation.size(1) == max_n_blocks);
  assert(pose_stack_min_bond_separation.size(2) == max_n_blocks);

  assert(pose_stack_inter_block_bondsep.size(0) == n_poses);
  assert(pose_stack_inter_block_bondsep.size(1) == max_n_blocks);
  assert(pose_stack_inter_block_bondsep.size(2) == max_n_blocks);
  assert(pose_stack_inter_block_bondsep.size(3) == max_n_interblock_bonds);
  assert(pose_stack_inter_block_bondsep.size(4) == max_n_interblock_bonds);

  assert(block_type_n_interblock_bonds.size(0) == n_block_types);

  assert(block_type_atoms_forming_chemical_bonds.size(0) == n_block_types);

  assert(block_type_inter_repr_path_distance.size(0) == n_block_types);
  assert(block_type_inter_repr_path_distance.size(1) == max_n_block_atoms);
  assert(block_type_inter_repr_path_distance.size(2) == max_n_block_atoms);
  assert(block_type_intra_repr_path_distance.size(0) == n_block_types);
  assert(block_type_intra_repr_path_distance.size(1) == max_n_block_atoms);
  assert(block_type_intra_repr_path_distance.size(2) == max_n_block_atoms);

  auto dV_dcoords_t = TPack<Vec<Real, 3>, 2, D>::zeros({1, n_atoms});
  auto dV_dcoords = dV_dcoords_t.view;

  auto scratch_rot_spheres_t = TPack<Real, 2, D>::zeros({n_rots, 4});
  auto scratch_rot_spheres = scratch_rot_spheres_t.view;

  auto scratch_rot_neighbors_t = TPack<Int, 3, D>::zeros(
      {n_poses, max_n_rots_per_pose, max_n_rots_per_pose});
  auto scratch_rot_neighbors = scratch_rot_neighbors_t.view;

  score::common::sphere_overlap::
      compute_rot_spheres<DeviceDispatch, D, Real, Int>::f(
          rot_coords,
          rot_coord_offset,
          block_type_ind_for_rot,
          block_type_n_atoms,
          scratch_rot_spheres);

  score::common::sphere_overlap::
      detect_rot_neighbors<DeviceDispatch, D, Real, Int>::f(
          max_n_rots_per_pose,
          block_ind_for_rot,
          block_type_ind_for_rot,
          block_type_n_atoms,
          n_rots_for_pose,
          rot_offset_for_pose,
          n_rots_for_block,
          scratch_rot_spheres,
          scratch_rot_neighbors,
          Real(5.5));  // 5.5A hard coded here. Please fix! TEMP!

  auto dispatch_indices_t = score::common::sphere_overlap::
      rot_neighbor_indices<DeviceDispatch, D, Int>::f(
          scratch_rot_neighbors, rot_offset_for_pose);

  auto dispatch_indices = dispatch_indices_t.view;

  // auto output_t = TPack<Real, 2, D>::zeros({1, n_poses});
  //  auto output_t =
  //      TPack<Real, 4, D>::zeros({3, n_poses, max_n_blocks, max_n_blocks});
  TPack<Real, 2, D> output_t;
  if (output_block_pair_energies) {
    output_t = TPack<Real, 2, D>::zeros({1, dispatch_indices.size(1)});
  } else {
    output_t = TPack<Real, 2, D>::zeros({1, n_poses});
  }
  auto output = output_t.view;

  // Optimal launch box on v100 and a100 is nt=32, vt=1
  LAUNCH_BOX_32;
  // Define nt and reduce_t
  CTA_REAL_REDUCE_T_TYPEDEF;

  auto eval_energies_by_block = ([=] TMOL_DEVICE_FUNC(int cta) {
    auto elec_atom_energy_and_derivs =
        ([=] TMOL_DEVICE_FUNC(
             int atom_tile_ind1,
             int atom_tile_ind2,
             int,
             int,
             ElecScoringData<Real> const& score_dat,
             int cp_separation) {
          return elec_atom_energy(
              atom_tile_ind1, atom_tile_ind2, score_dat, cp_separation);
        });

    auto score_inter_elec_atom_pair = ([=] SCORE_INTER_ELEC_ATOM_PAIR);

    auto score_intra_elec_atom_pair = ([=] SCORE_INTRA_ELEC_ATOM_PAIR);

    auto load_block_coords_and_params_into_shared =
        ([=] LOAD_BLOCK_COORDS_AND_PARAMS_INTO_SHARED);

    auto load_block_into_shared = ([=] LOAD_BLOCK_INTO_SHARED);

    SHARED_MEMORY union shared_mem_union {
      shared_mem_union() {}
      ElecBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN> m;
      CTA_REAL_REDUCE_T_VARIABLE;

    } shared;

    int const max_important_bond_separation = 4;

    int const pose_ind = dispatch_indices[0][cta];

    int const rot_ind1 = dispatch_indices[1][cta];
    int const rot_ind2 = dispatch_indices[2][cta];

    int const block_ind1 = block_ind_for_rot[rot_ind1];
    int const block_ind2 = block_ind_for_rot[rot_ind2];

    int const block_type1 = block_type_ind_for_rot[rot_ind1];
    int const block_type2 = block_type_ind_for_rot[rot_ind2];

    if (block_type1 < 0 || block_type2 < 0) {
      return;
    }

    int const n_atoms1 = block_type_n_atoms[block_type1];
    int const n_atoms2 = block_type_n_atoms[block_type2];

    // int const pose_ind = cta / (max_n_blocks * max_n_blocks);
    // int const block_ind_pair = cta % (max_n_blocks * max_n_blocks);
    // int const block_ind1 = block_ind_pair / max_n_blocks;
    // int const block_ind2 = block_ind_pair % max_n_blocks;
    // if (block_ind1 > block_ind2) {
    //   return;
    // }

    // if (scratch_block_neighbors[pose_ind][block_ind1][block_ind2] == 0) {
    //   return;
    // }

    // int const block_type1 = pose_stack_block_type[pose_ind][block_ind1];
    // int const block_type2 = pose_stack_block_type[pose_ind][block_ind2];

    // if (block_type1 < 0 || block_type2 < 0) {
    //   return;
    // }

    auto load_tile_invariant_interres_data =
        ([=] LOAD_TILE_INVARIANT_INTERRES_DATA);

    auto load_interres1_tile_data_to_shared =
        ([=] LOAD_INTERRES1_TILE_DATA_TO_SHARED);

    auto load_interres2_tile_data_to_shared =
        ([=] LOAD_INTERRES2_TILE_DATA_TO_SHARED);

    auto load_interres_data_from_shared = ([=] LOAD_INTERRES_DATA_FROM_SHARED);

    auto eval_interres_atom_pair_scores = ([=] EVAL_INTERRES_ATOM_PAIR_SCORES);

    auto store_calculated_energies = ([=] STORE_CALCULATED_ROTAMER_ENERGIES);

    auto load_tile_invariant_intrares_data =
        ([=] LOAD_TILE_INVARIANT_INTRARES_DATA);

    auto load_intrares1_tile_data_to_shared =
        ([=] LOAD_INTRARES1_TILE_DATA_TO_SHARED);

    auto load_intrares2_tile_data_to_shared =
        ([=] LOAD_INTRARES2_TILE_DATA_TO_SHARED);

    auto load_intrares_data_from_shared = ([=] LOAD_INTRARES_DATA_FROM_SHARED);

    auto eval_intrares_atom_pair_scores = ([=] EVAL_INTRARES_ATOM_PAIR_SCORES);

    tmol::score::common::tile_evaluate_rot_pair<
        DeviceDispatch,
        D,
        ElecScoringData<Real>,
        ElecScoringData<Real>,
        Real,
        TILE_SIZE>(
        shared,
        pose_ind,
        rot_ind1,
        rot_ind2,
        block_ind1,
        block_ind2,
        block_type1,
        block_type2,
        n_atoms1,
        n_atoms2,
        load_tile_invariant_interres_data,
        load_interres1_tile_data_to_shared,
        load_interres2_tile_data_to_shared,
        load_interres_data_from_shared,
        eval_interres_atom_pair_scores,
        store_calculated_energies,
        load_tile_invariant_intrares_data,
        load_intrares1_tile_data_to_shared,
        load_intrares2_tile_data_to_shared,
        load_intrares_data_from_shared,
        eval_intrares_atom_pair_scores,
        store_calculated_energies);
  });

  // auto eval_energies = ([=] TMOL_DEVICE_FUNC(int cta) {
  //   auto elec_atom_energy_and_derivs =
  //       ([=] TMOL_DEVICE_FUNC(
  //            int atom_tile_ind1,
  //            int atom_tile_ind2,
  //            int start_atom1,
  //            int start_atom2,
  //            ElecScoringData<Real> const &score_dat,
  //            int cp_separation) {
  //         if (compute_derivs) {
  //           auto val = elec_atom_energy_and_derivs_full(
  //               atom_tile_ind1,
  //               atom_tile_ind2,
  //               start_atom1,
  //               start_atom2,
  //               score_dat,
  //               cp_separation,
  //               dV_dcoords);
  //           return val;
  //         } else {
  //           return elec_atom_energy(
  //               atom_tile_ind1, atom_tile_ind2, score_dat, cp_separation);
  //         }
  //       });

  //   auto score_inter_elec_atom_pair = ([=] SCORE_INTER_ELEC_ATOM_PAIR);

  //   auto score_intra_elec_atom_pair = ([=] SCORE_INTRA_ELEC_ATOM_PAIR);

  //   auto load_block_coords_and_params_into_shared =
  //       ([=] LOAD_BLOCK_COORDS_AND_PARAMS_INTO_SHARED);

  //   auto load_block_into_shared = ([=] LOAD_BLOCK_INTO_SHARED);

  //   SHARED_MEMORY union shared_mem_union {
  //     shared_mem_union() {}
  //     ElecBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN> m;
  //     CTA_REAL_REDUCE_T_VARIABLE;

  //   } shared;

  //   int const max_important_bond_separation = 4;

  //   int const pose_ind = dispatch_indices[0][cta];

  //   int const rot_ind1 = dispatch_indices[1][cta];
  //   int const rot_ind2 = dispatch_indices[2][cta];

  //   int const block_ind1 = block_ind_for_rot[rot_ind1];
  //   int const block_ind2 = block_ind_for_rot[rot_ind2];

  //   int const block_type1 = block_type_ind_for_rot[rot_ind1];
  //   int const block_type2 = block_type_ind_for_rot[rot_ind2];

  //   if (block_type1 < 0 || block_type2 < 0) {
  //     return;
  //   }

  //   int const n_atoms1 = block_type_n_atoms[block_type1];
  //   int const n_atoms2 = block_type_n_atoms[block_type2];

  //   auto load_tile_invariant_interres_data =
  //       ([=] LOAD_TILE_INVARIANT_INTERRES_DATA);

  //   auto load_interres1_tile_data_to_shared =
  //       ([=] LOAD_INTERRES1_TILE_DATA_TO_SHARED);

  //   auto load_interres2_tile_data_to_shared =
  //       ([=] LOAD_INTERRES2_TILE_DATA_TO_SHARED);

  //   auto load_interres_data_from_shared = ([=]
  //   LOAD_INTERRES_DATA_FROM_SHARED);

  //   auto eval_interres_atom_pair_scores = ([=]
  //   EVAL_INTERRES_ATOM_PAIR_SCORES);

  //   auto store_calculated_energies = ([=] STORE_CALCULATED_ENERGIES);

  //   auto load_tile_invariant_intrares_data =
  //       ([=] LOAD_TILE_INVARIANT_INTRARES_DATA);

  //   auto load_intrares1_tile_data_to_shared =
  //       ([=] LOAD_INTRARES1_TILE_DATA_TO_SHARED);

  //   auto load_intrares2_tile_data_to_shared =
  //       ([=] LOAD_INTRARES2_TILE_DATA_TO_SHARED);

  //   auto load_intrares_data_from_shared = ([=]
  //   LOAD_INTRARES_DATA_FROM_SHARED);

  //   auto eval_intrares_atom_pair_scores = ([=]
  //   EVAL_INTRARES_ATOM_PAIR_SCORES);

  //   tmol::score::common::tile_evaluate_rot_pair<
  //       DeviceDispatch,
  //       D,
  //       ElecScoringData<Real>,
  //       ElecScoringData<Real>,
  //       Real,
  //       TILE_SIZE>(
  //       shared,
  //       pose_ind,
  //       rot_ind1,
  //       rot_ind2,
  //       block_ind1,
  //       block_ind2,
  //       block_type1,
  //       block_type2,
  //       n_atoms1,
  //       n_atoms2,
  //       load_tile_invariant_interres_data,
  //       load_interres1_tile_data_to_shared,
  //       load_interres2_tile_data_to_shared,
  //       load_interres_data_from_shared,
  //       eval_interres_atom_pair_scores,
  //       store_calculated_energies,
  //       load_tile_invariant_intrares_data,
  //       load_intrares1_tile_data_to_shared,
  //       load_intrares2_tile_data_to_shared,
  //       load_intrares_data_from_shared,
  //       eval_intrares_atom_pair_scores,
  //       store_calculated_energies);
  // });

  ///////////////////////////////////////////////////////////////////////

  // Three steps
  // 0: setup
  // 1: launch a kernel to find a small bounding sphere surrounding the blocks
  // 2: launch a kernel to look for spheres that are within striking distance of
  // each other
  // 3: launch a kernel to evaluate lj/lk between pairs of blocks
  // within striking distance

  // 0
  // TO DO: let DeviceDispatch hold a cuda stream (??)
  // at::cuda::CUDAStream wrapped_stream = at::cuda::getDefaultCUDAStream();
  // mgpu::standard_context_t context(wrapped_stream.stream());

  // 3
  // if (output_block_pair_energies) {
  // Rotamer energy evaluation is only done "by block"
  DeviceDispatch<D>::template foreach_workgroup<launch_t>(
      dispatch_indices.size(1), eval_energies_by_block);
  // } else {
  //   DeviceDispatch<D>::template foreach_workgroup<launch_t>(
  //       dispatch_indices.size(1), eval_energies);
  // }

  // DeviceDispatch<D>::synchronize_device();
  return {output_t, dV_dcoords_t, dispatch_indices_t};
}  // namespace potentials

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto ElecRotamerScoreDispatch<DeviceDispatch, D, Real, Int>::backward(
    // common params
    TView<Vec<Real, 3>, 1, D> rot_coords,
    TView<Int, 1, D> rot_coord_offset,
    TView<Int, 1, D> pose_ind_for_atom,
    TView<Int, 2, D> first_rot_for_block,
    TView<Int, 2, D> first_rot_block_type,
    TView<Int, 1, D> block_ind_for_rot,
    TView<Int, 1, D> pose_ind_for_rot,
    TView<Int, 1, D> block_type_ind_for_rot,
    TView<Int, 1, D> n_rots_for_pose,
    TView<Int, 1, D> rot_offset_for_pose,
    TView<Int, 2, D> n_rots_for_block,
    TView<Int, 2, D> rot_offset_for_block,
    Int max_n_rots_per_pose,

    // dims: n-poses x max-n-blocks x max-n-blocks
    // Quick lookup: given the inds of two blocks, ask: what is the minimum
    // number of chemical bonds that separate any pair of atoms in those
    // blocks? If this minimum is greater than the crossover, then no further
    // logic for deciding whether two atoms in those blocks should have their
    // interaction energies calculated: all should. intentionally small to
    // (possibly) fit in constant cache
    TView<Int, 3, D> pose_stack_min_bond_separation,

    // dims: n-poses x max-n-blocks x max-n-blocks x
    // max-n-interblock-connections x max-n-interblock-connections
    TView<Int, 5, D> pose_stack_inter_block_bondsep,

    //////////////////////
    // Chemical properties
    // how many atoms for a given block
    // Dimsize n_block_types
    TView<Int, 1, D> block_type_n_atoms,

    TView<Real, 2, D> block_type_partial_charge,

    // how many inter-block chemical bonds are there
    // Dimsize: n_block_types
    TView<Int, 1, D> block_type_n_interblock_bonds,

    // what atoms form the inter-block chemical bonds
    // Dimsize: n_block_types x max_n_interblock_bonds
    TView<Int, 2, D> block_type_atoms_forming_chemical_bonds,

    // what is the path distance between pairs of atoms in the block
    // denormalized by their count-pair representative; used for
    // inter-block chemical-bond separation determination. Entry
    // i, j stores path_dist[i, rep(j)]
    // Dimsize: n_block_types x max_n_atoms x max_n_atoms
    TView<Int, 3, D> block_type_inter_repr_path_distance,

    // what is the path distance between pairs of atoms in the block
    // denormalized (twice!) by their count-pair representative;
    // used for intra-block chemical-bond separation determination.
    // Entry i, j stores path_dist[rep(i), rep(j)]
    // Dimsize: n_block_types x max_n_atoms x max_n_atoms
    TView<Int, 3, D> block_type_intra_repr_path_distance,
    //////////////////////

    // LJ parameters
    TView<ElecGlobalParams<Real>, 1, D> global_params,

    TView<Int, 2, D> dispatch_indices,  // from forward pass
    TView<Real, 2, D> dTdV              // nterms x nposes x len x len
    ) -> TPack<Vec<Real, 3>, 2, D> {
  using tmol::score::common::accumulate;
  using Real3 = Vec<Real, 3>;

  int const n_atoms = rot_coords.size(0);
  int const n_rots = rot_coord_offset.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);

  int const n_block_types = block_type_n_atoms.size(0);
  int const max_n_block_atoms = block_type_partial_charge.size(1);
  int const max_n_interblock_bonds =
      block_type_atoms_forming_chemical_bonds.size(1);

  assert(first_rot_block_type.size(0) == n_poses);
  assert(first_rot_block_type.size(1) == max_n_blocks);

  assert(block_ind_for_rot.size(0) == n_rots);
  assert(pose_ind_for_rot.size(0) == n_rots);
  assert(block_type_ind_for_rot.size(0) == n_rots);

  assert(n_rots_for_pose.size(0) == n_poses);
  assert(rot_offset_for_pose.size(0) == n_poses);

  assert(n_rots_for_block.size(0) == n_poses);
  assert(n_rots_for_block.size(1) == max_n_blocks);

  assert(rot_offset_for_block.size(0) == n_poses);
  assert(rot_offset_for_block.size(1) == max_n_blocks);

  assert(max_n_interblock_bonds <= MAX_N_CONN);

  assert(pose_stack_min_bond_separation.size(0) == n_poses);
  assert(pose_stack_min_bond_separation.size(1) == max_n_blocks);
  assert(pose_stack_min_bond_separation.size(2) == max_n_blocks);

  assert(pose_stack_inter_block_bondsep.size(0) == n_poses);
  assert(pose_stack_inter_block_bondsep.size(1) == max_n_blocks);
  assert(pose_stack_inter_block_bondsep.size(2) == max_n_blocks);
  assert(pose_stack_inter_block_bondsep.size(3) == max_n_interblock_bonds);
  assert(pose_stack_inter_block_bondsep.size(4) == max_n_interblock_bonds);

  assert(block_type_n_interblock_bonds.size(0) == n_block_types);

  assert(block_type_atoms_forming_chemical_bonds.size(0) == n_block_types);

  assert(block_type_inter_repr_path_distance.size(0) == n_block_types);
  assert(block_type_inter_repr_path_distance.size(1) == max_n_block_atoms);
  assert(block_type_inter_repr_path_distance.size(2) == max_n_block_atoms);
  assert(block_type_intra_repr_path_distance.size(0) == n_block_types);
  assert(block_type_intra_repr_path_distance.size(1) == max_n_block_atoms);
  assert(block_type_intra_repr_path_distance.size(2) == max_n_block_atoms);

  auto dV_dcoords_t = TPack<Vec<Real, 3>, 2, D>::zeros({1, n_atoms});
  auto dV_dcoords = dV_dcoords_t.view;

  // Optimal launch box on v100 and a100 is nt=32, vt=1
  LAUNCH_BOX_32;
  // Define nt and reduce_t
  CTA_REAL_REDUCE_T_TYPEDEF;

  auto eval_derivs = ([=] TMOL_DEVICE_FUNC(int cta) {
    auto elec_atom_energy_and_derivs =
        ([=] TMOL_DEVICE_FUNC(
             int atom_tile_ind1,
             int atom_tile_ind2,
             int start_atom1,
             int start_atom2,
             ElecScoringData<Real> const& score_dat,
             int cp_separation) {
          elec_atom_derivs(
              atom_tile_ind1,
              atom_tile_ind2,
              start_atom1,
              start_atom2,
              score_dat,
              cp_separation,
              dTdV[0][cta],
              dV_dcoords);
          return 0.0;
        });

    // TEST!
    auto score_inter_elec_atom_pair = ([=] SCORE_INTER_ELEC_ATOM_PAIR);

    auto score_intra_elec_atom_pair = ([=] SCORE_INTRA_ELEC_ATOM_PAIR);

    auto load_block_coords_and_params_into_shared =
        ([=] LOAD_BLOCK_COORDS_AND_PARAMS_INTO_SHARED);

    auto load_block_into_shared = ([=] LOAD_BLOCK_INTO_SHARED);

    SHARED_MEMORY union shared_mem_union {
      shared_mem_union() {}
      ElecBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN> m;
      CTA_REAL_REDUCE_T_VARIABLE;

    } shared;

    int const max_important_bond_separation = 4;
    int const pose_ind = dispatch_indices[0][cta];

    int const rot_ind1 = dispatch_indices[1][cta];
    int const rot_ind2 = dispatch_indices[2][cta];

    int const block_ind1 = block_ind_for_rot[rot_ind1];
    int const block_ind2 = block_ind_for_rot[rot_ind2];

    int const block_type1 = block_type_ind_for_rot[rot_ind1];
    int const block_type2 = block_type_ind_for_rot[rot_ind2];

    if (block_type1 < 0 || block_type2 < 0) {
      return;
    }

    int const n_atoms1 = block_type_n_atoms[block_type1];
    int const n_atoms2 = block_type_n_atoms[block_type2];

    auto load_tile_invariant_interres_data =
        ([=] LOAD_TILE_INVARIANT_INTERRES_DATA);

    auto load_interres1_tile_data_to_shared =
        ([=] LOAD_INTERRES1_TILE_DATA_TO_SHARED);

    auto load_interres2_tile_data_to_shared =
        ([=] LOAD_INTERRES2_TILE_DATA_TO_SHARED);

    auto load_interres_data_from_shared = ([=] LOAD_INTERRES_DATA_FROM_SHARED);

    auto eval_interres_atom_pair_scores = ([=] EVAL_INTERRES_ATOM_PAIR_SCORES);

    auto store_calculated_energies =
        ([=](ElecScoringData<Real>& score_dat, shared_mem_union& shared) {
          ;  // no op for gradients ()
        });

    auto load_tile_invariant_intrares_data =
        ([=] LOAD_TILE_INVARIANT_INTRARES_DATA);

    auto load_intrares1_tile_data_to_shared =
        ([=] LOAD_INTRARES1_TILE_DATA_TO_SHARED);

    auto load_intrares2_tile_data_to_shared =
        ([=] LOAD_INTRARES2_TILE_DATA_TO_SHARED);

    auto load_intrares_data_from_shared = ([=] LOAD_INTRARES_DATA_FROM_SHARED);

    auto eval_intrares_atom_pair_scores = ([=] EVAL_INTRARES_ATOM_PAIR_SCORES);

    tmol::score::common::tile_evaluate_rot_pair<
        DeviceDispatch,
        D,
        ElecScoringData<Real>,
        ElecScoringData<Real>,
        Real,
        TILE_SIZE>(
        shared,
        pose_ind,
        rot_ind1,
        rot_ind2,
        block_ind1,
        block_ind2,
        block_type1,
        block_type2,
        n_atoms1,
        n_atoms2,
        load_tile_invariant_interres_data,
        load_interres1_tile_data_to_shared,
        load_interres2_tile_data_to_shared,
        load_interres_data_from_shared,
        eval_interres_atom_pair_scores,
        store_calculated_energies,
        load_tile_invariant_intrares_data,
        load_intrares1_tile_data_to_shared,
        load_intrares2_tile_data_to_shared,
        load_intrares_data_from_shared,
        eval_intrares_atom_pair_scores,
        store_calculated_energies);
  });

  // Since we have the sphere overlap results from the forward pass,
  // there's only a single kernel launch here
  DeviceDispatch<D>::template foreach_workgroup<launch_t>(
      dispatch_indices.size(1), eval_derivs);

  return dV_dcoords_t;
}

}  // namespace potentials
}  // namespace elec
}  // namespace score
}  // namespace tmol
