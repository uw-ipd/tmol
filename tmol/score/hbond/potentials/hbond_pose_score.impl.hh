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
#include <tmol/score/common/warp_segreduce.hh>
#include <tmol/score/common/warp_stride_reduce.hh>

#include <tmol/score/hbond/potentials/hbond.hh>
#include <tmol/score/hbond/potentials/params.hh>
#include <tmol/score/hbond/potentials/hbond_pose_score.hh>

// Operator definitions; safe for CPU compilation
#include <moderngpu/operators.hxx>

#include <chrono>

// The maximum number of inter-residue chemical bonds
#define MAX_N_CONN 4
#define TILE_SIZE 32

namespace tmol {
namespace score {
namespace hbond {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <int TILE, typename InterEnergyData>
EIGEN_DEVICE_FUNC int interres_count_pair_separation(
    InterEnergyData const &inter_dat, int atom_tile_ind1, int atom_tile_ind2) {
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

// ROT CENTRIC
// ROT CENTRIC
// ROT CENTRIC
// ROT CENTRIC
// ROT CENTRIC
// ROT CENTRIC
// ROT CENTRIC
// ROT CENTRIC
// ROT CENTRIC

// done
#define HBOND_ATOM_ENERGY_2                                      \
  TMOL_DEVICE_FUNC(                                              \
      int don_ind,                                               \
      int acc_ind,                                               \
      int don_tile_ind,                                          \
      int acc_tile_ind,                                          \
      int don_start,                                             \
      int acc_start,                                             \
      HBondSingleResData<Real> const &don_dat,                   \
      HBondSingleResData<Real> const &acc_dat,                   \
      HBondRotPairData<Dev, Real, Int> const &respair_dat,       \
      int cp_separation) {                                       \
    if (cp_separation < 5) {                                     \
      return Real(0.0);                                          \
    }                                                            \
    if (compute_derivs) {                                        \
      Real val = hbond_atom_energy_and_derivs_full_2<TILE_SIZE>( \
          don_ind,                                               \
          acc_ind,                                               \
          don_tile_ind,                                          \
          acc_tile_ind,                                          \
          don_start,                                             \
          acc_start,                                             \
          don_dat,                                               \
          acc_dat,                                               \
          respair_dat,                                           \
          cp_separation,                                         \
          dV_dcoords);                                           \
      return val;                                                \
    } else {                                                     \
      Real val = hbond_atom_energy_full_2<TILE_SIZE>(            \
          don_ind,                                               \
          acc_ind,                                               \
          don_tile_ind,                                          \
          acc_tile_ind,                                          \
          don_start,                                             \
          acc_start,                                             \
          don_dat,                                               \
          acc_dat,                                               \
          respair_dat,                                           \
          cp_separation);                                        \
      return val;                                                \
    }                                                            \
  }

// done
#define SCORE_INTER_HBOND_ATOM_PAIR_2                                        \
  TMOL_DEVICE_FUNC(                                                          \
      int don_start,                                                         \
      int acc_start,                                                         \
      int don_ind,                                                           \
      int acc_ind,                                                           \
      HBondScoringData2<Dev, Real, Int> const &inter_dat,                    \
      bool donor_first)                                                      \
      ->std::array<Real, 1> {                                                \
    int don_tile_ind =                                                       \
        (donor_first ? inter_dat.r1 : inter_dat.r2).donH_tile_inds[don_ind]; \
    int acc_tile_ind =                                                       \
        (donor_first ? inter_dat.r2 : inter_dat.r1).acc_tile_inds[acc_ind];  \
    int separation = interres_count_pair_separation<TILE_SIZE>(              \
        inter_dat,                                                           \
        (donor_first ? don_ind : acc_ind),                                   \
        (donor_first ? acc_ind : don_ind));                                  \
    Real hbond = hbond_atom_energy(                                          \
        don_ind,                                                             \
        acc_ind,                                                             \
        don_tile_ind,                                                        \
        acc_tile_ind,                                                        \
        don_start,                                                           \
        acc_start,                                                           \
        donor_first ? inter_dat.r1 : inter_dat.r2,                           \
        donor_first ? inter_dat.r2 : inter_dat.r1,                           \
        inter_dat.pair_data,                                                 \
        separation);                                                         \
    return {hbond};                                                          \
  }

// done
#define SCORE_INTRA_HBOND_ATOM_PAIR_2                                        \
  TMOL_DEVICE_FUNC(                                                          \
      int don_start,                                                         \
      int acc_start,                                                         \
      int don_ind,                                                           \
      int acc_ind,                                                           \
      HBondScoringData2<Dev, Real, Int> const &intra_dat,                    \
      bool donor_first)                                                      \
      ->std::array<Real, 1> {                                                \
    int don_tile_ind =                                                       \
        (donor_first ? intra_dat.r1 : intra_dat.r2).donH_tile_inds[don_ind]; \
    int acc_tile_ind =                                                       \
        (donor_first ? intra_dat.r2 : intra_dat.r1).acc_tile_inds[acc_ind];  \
    int const don_atom_ind = don_start + don_tile_ind;                       \
    int const acc_atom_ind = acc_start + acc_tile_ind;                       \
    int const separation =                                                   \
        block_type_path_distance[intra_dat.r1.block_type][don_atom_ind]      \
                                [acc_atom_ind];                              \
    Real hbond = hbond_atom_energy(                                          \
        don_ind,                                                             \
        acc_ind,                                                             \
        don_tile_ind,                                                        \
        acc_tile_ind,                                                        \
        don_start,                                                           \
        acc_start,                                                           \
        donor_first ? intra_dat.r1 : intra_dat.r2,                           \
        donor_first ? intra_dat.r2 : intra_dat.r1,                           \
        intra_dat.pair_data,                                                 \
        separation);                                                         \
    return {hbond};                                                          \
  }

// done
#define LOAD_TILE_INVARIANT_INTERRES_DATA_2                             \
  TMOL_DEVICE_FUNC(                                                     \
      int pose_ind,                                                     \
      int rot_ind1,                                                     \
      int rot_ind2,                                                     \
      int block_ind1,                                                   \
      int block_ind2,                                                   \
      int block_type1,                                                  \
      int block_type2,                                                  \
      int n_atoms1,                                                     \
      int n_atoms2,                                                     \
      HBondScoringData2<Dev, Real, Int> &inter_dat,                     \
      shared_mem_union &shared) {                                       \
    hbond_load_tile_invariant_interres_data_2<DeviceDispatch, Dev, nt>( \
        rot_coords,                                                     \
        first_rot_for_block,                                            \
        first_rot_block_type,                                           \
        rot_coord_offset,                                               \
        block_type_ind_for_rot,                                         \
        pose_stack_inter_residue_connections,                           \
        pose_stack_min_bond_separation,                                 \
        pose_stack_inter_block_bondsep,                                 \
        block_type_n_all_bonds,                                         \
        block_type_all_bonds,                                           \
        block_type_atom_all_bond_ranges,                                \
        block_type_n_interblock_bonds,                                  \
        block_type_atoms_forming_chemical_bonds,                        \
        block_type_atom_is_hydrogen,                                    \
        pair_params,                                                    \
        pair_polynomials,                                               \
        global_params,                                                  \
        max_important_bond_separation,                                  \
        pose_ind,                                                       \
        rot_ind1,                                                       \
        rot_ind2,                                                       \
        block_ind1,                                                     \
        block_ind2,                                                     \
        block_type1,                                                    \
        block_type2,                                                    \
        n_atoms1,                                                       \
        n_atoms2,                                                       \
        inter_dat,                                                      \
        shared.m);                                                      \
  }

// done
#define LOAD_INTERRES1_TILE_DATA_TO_SHARED_2                             \
  TMOL_DEVICE_FUNC(                                                      \
      int tile_ind,                                                      \
      int start_atom1,                                                   \
      int n_atoms_to_load1,                                              \
      HBondScoringData2<Dev, Real, Int> &inter_dat,                      \
      shared_mem_union &shared) {                                        \
    hbond_load_interres1_tile_data_to_shared_2<DeviceDispatch, Dev, nt>( \
        rot_coords,                                                      \
        block_type_tile_n_donH,                                          \
        block_type_tile_n_acc,                                           \
        block_type_tile_donH_inds,                                       \
        block_type_tile_acc_inds,                                        \
        block_type_tile_donor_type,                                      \
        block_type_tile_acceptor_type,                                   \
        block_type_tile_hybridization,                                   \
        block_type_path_distance,                                        \
        tile_ind,                                                        \
        start_atom1,                                                     \
        n_atoms_to_load1,                                                \
        inter_dat,                                                       \
        shared.m);                                                       \
  }

// done
#define LOAD_INTERRES2_TILE_DATA_TO_SHARED_2                             \
  TMOL_DEVICE_FUNC(                                                      \
      int tile_ind,                                                      \
      int start_atom2,                                                   \
      int n_atoms_to_load2,                                              \
      HBondScoringData2<Dev, Real, Int> &inter_dat,                      \
      shared_mem_union &shared) {                                        \
    hbond_load_interres2_tile_data_to_shared_2<DeviceDispatch, Dev, nt>( \
        rot_coords,                                                      \
        block_type_tile_n_donH,                                          \
        block_type_tile_n_acc,                                           \
        block_type_tile_donH_inds,                                       \
        block_type_tile_acc_inds,                                        \
        block_type_tile_donor_type,                                      \
        block_type_tile_acceptor_type,                                   \
        block_type_tile_hybridization,                                   \
        block_type_path_distance,                                        \
        tile_ind,                                                        \
        start_atom2,                                                     \
        n_atoms_to_load2,                                                \
        inter_dat,                                                       \
        shared.m);                                                       \
  }

// done
#define LOAD_INTERRES_DATA_FROM_SHARED_2 \
  TMOL_DEVICE_FUNC(                      \
      int, int, shared_mem_union &, HBondScoringData2<Dev, Real, Int> &) {}

// done
#define EVAL_INTERRES_ATOM_PAIR_SCORES_2                                   \
  TMOL_DEVICE_FUNC(                                                        \
      HBondScoringData2<Dev, Real, Int> &inter_dat,                        \
      int start_atom1,                                                     \
      int start_atom2) {                                                   \
    eval_interres_don_acc_pair_energies_2<DeviceDispatch, Dev, nt>(        \
        inter_dat, start_atom1, start_atom2, score_inter_hbond_atom_pair); \
  }

// done
#define STORE_CALCULATED_ENERGIES_2                                           \
  TMOL_DEVICE_FUNC(                                                           \
      HBondScoringData2<Dev, Real, Int> &score_dat,                           \
      shared_mem_union &shared) {                                             \
    auto reduce_energies = ([&](int tid) {                                    \
      Real const cta_total_hbond =                                            \
          DeviceDispatch<Dev>::template reduce_in_workgroup<nt>(              \
              score_dat.pair_data.total_hbond, shared, mgpu::plus_t<Real>()); \
      if (tid == 0) {                                                         \
        if (!output_block_pair_energies) {                                    \
          accumulate<Dev, Real>::add(output[cta], cta_total_hbond);           \
        } else {                                                              \
          output[cta] = cta_total_hbond;                                      \
        }                                                                     \
      }                                                                       \
    });                                                                       \
    DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(reduce_energies); \
  }

// done
#define LOAD_TILE_INVARIANT_INTRARES_DATA_2                             \
  TMOL_DEVICE_FUNC(                                                     \
      int pose_ind,                                                     \
      int rot_ind1,                                                     \
      int block_ind1,                                                   \
      int block_type1,                                                  \
      int n_atoms1,                                                     \
      HBondScoringData2<Dev, Real, Int> &intra_dat,                     \
      shared_mem_union &shared) {                                       \
    hbond_load_tile_invariant_intrares_data_2<DeviceDispatch, Dev, nt>( \
        rot_coords,                                                     \
        first_rot_for_block,                                            \
        first_rot_block_type,                                           \
        rot_coord_offset,                                               \
        block_type_ind_for_rot,                                         \
        pose_stack_inter_residue_connections,                           \
        block_type_n_all_bonds,                                         \
        block_type_all_bonds,                                           \
        block_type_atom_all_bond_ranges,                                \
        block_type_atoms_forming_chemical_bonds,                        \
        block_type_atom_is_hydrogen,                                    \
        pair_params,                                                    \
        pair_polynomials,                                               \
        global_params,                                                  \
        max_important_bond_separation,                                  \
        pose_ind,                                                       \
        rot_ind1,                                                       \
        block_ind1,                                                     \
        block_type1,                                                    \
        n_atoms1,                                                       \
        intra_dat,                                                      \
        shared.m);                                                      \
  }

// done
#define LOAD_INTRARES1_TILE_DATA_TO_SHARED_2                             \
  TMOL_DEVICE_FUNC(                                                      \
      int tile_ind,                                                      \
      int start_atom1,                                                   \
      int n_atoms_to_load1,                                              \
      HBondScoringData2<Dev, Real, Int> &intra_dat,                      \
      shared_mem_union &shared) {                                        \
    hbond_load_intrares1_tile_data_to_shared_2<DeviceDispatch, Dev, nt>( \
        rot_coords,                                                      \
        block_type_tile_n_donH,                                          \
        block_type_tile_n_acc,                                           \
        block_type_tile_donH_inds,                                       \
        block_type_tile_acc_inds,                                        \
        block_type_tile_donor_type,                                      \
        block_type_tile_acceptor_type,                                   \
        block_type_tile_hybridization,                                   \
        tile_ind,                                                        \
        start_atom1,                                                     \
        n_atoms_to_load1,                                                \
        intra_dat,                                                       \
        shared.m);                                                       \
  }

// done
#define LOAD_INTRARES2_TILE_DATA_TO_SHARED_2                             \
  TMOL_DEVICE_FUNC(                                                      \
      int tile_ind,                                                      \
      int start_atom2,                                                   \
      int n_atoms_to_load2,                                              \
      HBondScoringData2<Dev, Real, Int> &intra_dat,                      \
      shared_mem_union &shared) {                                        \
    hbond_load_intrares2_tile_data_to_shared_2<DeviceDispatch, Dev, nt>( \
        rot_coords,                                                      \
        block_type_tile_n_donH,                                          \
        block_type_tile_n_acc,                                           \
        block_type_tile_donH_inds,                                       \
        block_type_tile_acc_inds,                                        \
        block_type_tile_donor_type,                                      \
        block_type_tile_acceptor_type,                                   \
        block_type_tile_hybridization,                                   \
        tile_ind,                                                        \
        start_atom2,                                                     \
        n_atoms_to_load2,                                                \
        intra_dat,                                                       \
        shared.m);                                                       \
  }

// done
#define LOAD_INTRARES_DATA_FROM_SHARED_2              \
  TMOL_DEVICE_FUNC(                                   \
      int tile_ind1,                                  \
      int tile_ind2,                                  \
      shared_mem_union &shared,                       \
      HBondScoringData2<Dev, Real, Int> &intra_dat) { \
    hbond_load_intrares_data_from_shared_2(           \
        tile_ind1, tile_ind2, shared.m, intra_dat);   \
  }

// done
#define EVAL_INTRARES_ATOM_PAIR_SCORES_2                                   \
  TMOL_DEVICE_FUNC(                                                        \
      HBondScoringData2<Dev, Real, Int> &intra_dat,                        \
      int start_atom1,                                                     \
      int start_atom2) {                                                   \
    eval_intrares_don_acc_pair_energies_2<DeviceDispatch, Dev, nt>(        \
        intra_dat, start_atom1, start_atom2, score_intra_hbond_atom_pair); \
  }

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    typename Real,
    typename Int>
auto HBondPoseScoreDispatch2<DeviceDispatch, Dev, Real, Int>::forward(
    TView<Vec<Real, 3>, 1, Dev> rot_coords,
    TView<Int, 1, Dev> rot_coord_offset,

    TView<Int, 2, Dev> first_rot_for_block,
    TView<Int, 2, Dev> first_rot_block_type,
    TView<Int, 1, Dev> block_ind_for_rot,
    TView<Int, 1, Dev> pose_ind_for_rot,

    TView<Int, 1, Dev> block_type_ind_for_rot,

    TView<Int, 1, Dev> n_rots_for_pose,
    TView<Int, 1, Dev> rot_offset_for_pose,
    TView<Int, 2, Dev> n_rots_for_block,
    TView<Int, 2, Dev> rot_offset_for_block,
    Int max_n_rots_per_pose,

    // For determining which atoms to retrieve from neighboring
    // residues we have to know how the blocks in the Pose
    // are connected
    TView<Vec<Int, 2>, 3, Dev> pose_stack_inter_residue_connections,
    // void*, //Unsure how to proceed - for hbond i guess we can just rely on
    // dispatch to take care of this?

    // dims: n-poses x max-n-blocks x max-n-blocks
    // Quick lookup: given the inds of two blocks, ask: what is the minimum
    // number of chemical bonds that separate any pair of atoms in those
    // blocks? If this minimum is greater than the crossover, then no further
    // logic for deciding whether two atoms in those blocks should have their
    // interaction energies calculated: all should. intentionally small to
    // (possibly) fit in constant cache
    TView<Int, 3, Dev> pose_stack_min_bond_separation,

    // dims: n-poses x max-n-blocks x max-n-blocks x
    // max-n-interblock-connections x max-n-interblock-connections
    TView<Int, 5, Dev> pose_stack_inter_block_bondsep,

    //////////////////////
    // Chemical properties
    // how many atoms for a given block
    // Dimsize n_block_types
    TView<Int, 1, Dev> block_type_n_atoms,

    // how many inter-block chemical bonds are there
    // Dimsize: n_block_types
    TView<Int, 1, Dev> block_type_n_interblock_bonds,

    // what atoms form the inter-block chemical bonds
    // Dimsize: n_block_types x max_n_interblock_bonds
    TView<Int, 2, Dev> block_type_atoms_forming_chemical_bonds,

    TView<Int, 1, Dev> block_type_n_all_bonds,
    TView<Vec<Int, 3>, 2, Dev> block_type_all_bonds,
    TView<Vec<Int, 2>, 2, Dev> block_type_atom_all_bond_ranges,

    TView<Int, 2, Dev> block_type_tile_n_donH,
    TView<Int, 2, Dev> block_type_tile_n_acc,
    TView<Int, 3, Dev> block_type_tile_donH_inds,
    TView<Int, 3, Dev> block_type_tile_acc_inds,
    TView<Int, 3, Dev> block_type_tile_donor_type,
    TView<Int, 3, Dev> block_type_tile_acceptor_type,
    TView<Int, 3, Dev> block_type_tile_hybridization,
    TView<Int, 2, Dev> block_type_atom_is_hydrogen,

    // How many chemical bonds separate all pairs of atoms
    // within each block type?
    // Dimsize: n_block_types x max_n_atoms x max_n_atoms
    TView<Int, 3, Dev> block_type_path_distance,

    //////////////////////

    // HBond potential parameters
    TView<HBondPairParams<Real>, 2, Dev> pair_params,
    TView<HBondPolynomials<double>, 2, Dev> pair_polynomials,
    TView<HBondGlobalParams<Real>, 1, Dev> global_params,

    bool output_block_pair_energies,
    bool compute_derivs

    )
    -> std::tuple<
        TPack<Real, 1, Dev>,
        TPack<Vec<Real, 3>, 2, Dev>,
        TPack<Int, 2, Dev> > {
  using tmol::score::common::accumulate;
  using Real3 = Vec<Real, 3>;

  printf("CHECK0\n");

  int const n_poses = n_rots_for_pose.size(0);
  // int const max_n_pose_atoms = rot_coords.size(1);
  // int const max_n_blocks = pose_stack_block_type.size(1);
  int const max_n_conn = pose_stack_inter_residue_connections.size(2);
  int const n_block_types = block_type_n_atoms.size(0);
  int const max_n_block_atoms = block_type_atom_is_hydrogen.size(1);
  int const max_n_interblock_bonds =
      block_type_atoms_forming_chemical_bonds.size(1);
  int const max_n_all_bonds = block_type_all_bonds.size(1);
  int const max_n_tiles = block_type_tile_donH_inds.size(1);
  int const max_n_donH_per_tile = block_type_tile_donH_inds.size(2);
  int const max_n_acc_per_tile = block_type_tile_acc_inds.size(2);

  assert(max_n_interblock_bonds <= MAX_N_CONN);
  printf("CHECK1\n");

  /*assert(pose_stack_block_coord_offset.size(0) == n_poses);
  assert(pose_stack_block_coord_offset.size(1) == max_n_blocks);

  assert(pose_stack_block_type.size(0) == n_poses);

  assert(pose_stack_inter_residue_connections.size(0) == n_poses);
  assert(pose_stack_inter_residue_connections.size(1) == max_n_blocks);

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

  assert(block_type_n_all_bonds.size(0) == n_block_types);
  assert(block_type_all_bonds.size(0) == n_block_types);
  assert(block_type_atom_all_bond_ranges.size(0) == n_block_types);
  assert(block_type_atom_all_bond_ranges.size(1) == max_n_block_atoms);

  assert(block_type_tile_n_donH.size(0) == n_block_types);
  assert(block_type_tile_n_donH.size(1) == max_n_tiles);
  assert(block_type_tile_n_acc.size(0) == n_block_types);
  assert(block_type_tile_n_acc.size(1) == max_n_tiles);
  assert(block_type_tile_donH_inds.size(0) == n_block_types);
  assert(block_type_tile_donH_inds.size(1) == max_n_tiles);
  assert(block_type_tile_acc_inds.size(0) == n_block_types);
  assert(block_type_tile_acc_inds.size(1) == max_n_tiles);
  assert(block_type_tile_donor_type.size(0) == n_block_types);
  assert(block_type_tile_donor_type.size(1) == max_n_tiles);
  assert(block_type_tile_donor_type.size(2) == max_n_donH_per_tile);
  assert(block_type_tile_acceptor_type.size(0) == n_block_types);
  assert(block_type_tile_acceptor_type.size(1) == max_n_tiles);
  assert(block_type_tile_acceptor_type.size(2) == max_n_acc_per_tile);
  assert(block_type_tile_hybridization.size(0) == n_block_types);
  assert(block_type_tile_hybridization.size(1) == max_n_tiles);
  assert(block_type_tile_hybridization.size(2) == max_n_acc_per_tile);
  assert(block_type_atom_is_hydrogen.size(0) == n_block_types);
  assert(block_type_atom_is_hydrogen.size(1) == max_n_block_atoms);

  assert(block_type_path_distance.size(0) == n_block_types);
  assert(block_type_path_distance.size(1) == max_n_block_atoms);
  assert(block_type_path_distance.size(2) == max_n_block_atoms);*/

  /*TPack<Real, 4, Dev> output_t;
  if (output_block_pair_energies) {
    output_t =
        TPack<Real, 4, Dev>::zeros({1, n_poses, max_n_blocks, max_n_blocks});
  } else {
    output_t = TPack<Real, 4, Dev>::zeros({1, n_poses, 1, 1});
  }*/

  // auto accum_output_t = TPack<double, 2, Dev>::zeros({1, n_poses});
  // auto accum_output = accum_output_t.view;

  auto n_rots = rot_coord_offset.size(0);
  // auto rot_max_n_atoms = rot_coords.size(1);
  auto n_atoms = rot_coords.size(0);

  auto dV_dcoords_t = TPack<Vec<Real, 3>, 2, Dev>::zeros({1, n_atoms});
  auto dV_dcoords = dV_dcoords_t.view;

  auto scratch_rot_spheres_t = TPack<Real, 2, Dev>::zeros({n_rots, 4});
  auto scratch_rot_spheres = scratch_rot_spheres_t.view;

  auto scratch_rot_neighbors_t = TPack<Int, 3, Dev>::zeros(
      {n_poses, max_n_rots_per_pose, max_n_rots_per_pose});
  auto scratch_rot_neighbors = scratch_rot_neighbors_t.view;
  printf("CHECK2\n");

  // TPack<Int, 2, Dev> dispatch_indices_t;

  score::common::sphere_overlap::
      compute_rot_spheres<DeviceDispatch, Dev, Real, Int>::f(
          rot_coords,
          rot_coord_offset,
          block_type_ind_for_rot,
          block_type_n_atoms,
          scratch_rot_spheres);
  printf("CHECK3\n");

  score::common::sphere_overlap::
      detect_rot_neighbors<DeviceDispatch, Dev, Real, Int>::f(
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
  printf("CHECK4\n");

  auto dispatch_indices_t = score::common::sphere_overlap::
      rot_neighbor_indices<DeviceDispatch, Dev, Int>::f(
          scratch_rot_neighbors, rot_offset_for_pose);
  auto dispatch_indices = dispatch_indices_t.view;

  TPack<Real, 1, Dev> output_t;
  output_t = TPack<Real, 1, Dev>::zeros({dispatch_indices.size(1)});
  auto output = output_t.view;

  /*for(int i = 0; i < dispatch_indices.size(1); i++){
    printf("DISPATCH %i, %i %i %i", i,
      dispatch_indices[0][i],
      dispatch_indices[1][i],
      dispatch_indices[2][i]);
  }*/

  // Optimal launch box on v100 and a100 is nt=32, vt=1
  LAUNCH_BOX_32;
  // Define nt and reduce_t
  CTA_REAL_REDUCE_T_TYPEDEF;

  auto eval_energies = ([=] TMOL_DEVICE_FUNC(int cta) {
    auto hbond_atom_energy = ([=] HBOND_ATOM_ENERGY_2);

    auto score_inter_hbond_atom_pair = ([=] SCORE_INTER_HBOND_ATOM_PAIR_2);

    auto score_intra_hbond_atom_pair = ([=] SCORE_INTRA_HBOND_ATOM_PAIR_2);

    SHARED_MEMORY union shared_mem_union {
      shared_mem_union() {}
      HBondBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN> m;
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

    int const n_atoms1 = block_type_n_atoms[block_type1];
    int const n_atoms2 = block_type_n_atoms[block_type2];

    // printf("B: %i %i %i\n", t_pose_ind, t_block_ind1, t_block_ind2);

    // if (block_type1 < 0 || block_type2 < 0) { // this should be entirely
    // unnecessary because we handle this while computing dispatch return;
    //}

    auto load_tile_invariant_interres_data =
        ([=] LOAD_TILE_INVARIANT_INTERRES_DATA_2);

    auto load_interres1_tile_data_to_shared =
        ([=] LOAD_INTERRES1_TILE_DATA_TO_SHARED_2);

    auto load_interres2_tile_data_to_shared =
        ([=] LOAD_INTERRES2_TILE_DATA_TO_SHARED_2);

    auto load_interres_data_from_shared =
        ([=] LOAD_INTERRES_DATA_FROM_SHARED_2);

    auto eval_interres_atom_pair_scores =
        ([=] EVAL_INTERRES_ATOM_PAIR_SCORES_2);

    auto store_calculated_energies = ([=] STORE_CALCULATED_ENERGIES_2);

    auto load_tile_invariant_intrares_data =
        ([=] LOAD_TILE_INVARIANT_INTRARES_DATA_2);

    auto load_intrares1_tile_data_to_shared =
        ([=] LOAD_INTRARES1_TILE_DATA_TO_SHARED_2);

    auto load_intrares2_tile_data_to_shared =
        ([=] LOAD_INTRARES2_TILE_DATA_TO_SHARED_2);

    auto load_intrares_data_from_shared =
        ([=] LOAD_INTRARES_DATA_FROM_SHARED_2);

    auto eval_intrares_atom_pair_scores =
        ([=] EVAL_INTRARES_ATOM_PAIR_SCORES_2);

    tmol::score::common::tile_evaluate_rot_pair<
        DeviceDispatch,
        Dev,
        HBondScoringData2<Dev, Real, Int>,
        HBondScoringData2<Dev, Real, Int>,
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
  // 1: launch a kernel to find a small bounding sphere surrounding the
  // blocks 2: launch a kernel to look for spheres that are within
  // striking distance of each other 3: launch a kernel to evaluate
  // lj/lk between pairs of blocks within striking distance

  // 0
  // TO DO: let DeviceDispatch hold a cuda stream (??)
  // at::cuda::CUDAStream wrapped_stream =
  // at::cuda::getDefaultCUDAStream(); mgpu::standard_context_t
  // context(wrapped_stream.stream());

  // 3
  DeviceDispatch<Dev>::template foreach_workgroup<launch_t>(
      dispatch_indices.size(1), eval_energies);

  return {output_t, dV_dcoords_t, dispatch_indices_t};
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    typename Real,
    typename Int>
auto HBondPoseScoreDispatch2<DeviceDispatch, Dev, Real, Int>::backward(
    TView<Vec<Real, 3>, 1, Dev> rot_coords,
    TView<Int, 1, Dev> rot_coord_offset,

    TView<Int, 2, Dev> first_rot_for_block,
    TView<Int, 2, Dev> first_rot_block_type,

    TView<Int, 1, Dev> block_ind_for_rot,
    TView<Int, 1, Dev> pose_ind_for_rot,

    TView<Int, 1, Dev> block_type_ind_for_rot,

    TView<Int, 1, Dev> n_rots_for_pose,
    TView<Int, 1, Dev> rot_offset_for_pose,
    TView<Int, 2, Dev> n_rots_for_block,
    TView<Int, 2, Dev> rot_offset_for_block,
    Int max_n_rots_per_pose,

    // For determining which atoms to retrieve from neighboring
    // residues we have to know how the blocks in the Pose
    // are connected
    TView<Vec<Int, 2>, 3, Dev> pose_stack_inter_residue_connections,

    // dims: n-poses x max-n-blocks x max-n-blocks
    // Quick lookup: given the inds of two blocks, ask: what is the minimum
    // number of chemical bonds that separate any pair of atoms in those
    // blocks? If this minimum is greater than the crossover, then no further
    // logic for deciding whether two atoms in those blocks should have their
    // interaction energies calculated: all should. intentionally small to
    // (possibly) fit in constant cache
    TView<Int, 3, Dev> pose_stack_min_bond_separation,

    // dims: n-poses x max-n-blocks x max-n-blocks x
    // max-n-interblock-connections x max-n-interblock-connections
    TView<Int, 5, Dev> pose_stack_inter_block_bondsep,

    //////////////////////
    // Chemical properties
    // how many atoms for a given block
    // Dimsize n_block_types
    TView<Int, 1, Dev> block_type_n_atoms,

    // how many inter-block chemical bonds are there
    // Dimsize: n_block_types
    TView<Int, 1, Dev> block_type_n_interblock_bonds,

    // what atoms form the inter-block chemical bonds
    // Dimsize: n_block_types x max_n_interblock_bonds
    TView<Int, 2, Dev> block_type_atoms_forming_chemical_bonds,

    TView<Int, 1, Dev> block_type_n_all_bonds,
    TView<Vec<Int, 3>, 2, Dev> block_type_all_bonds,
    TView<Vec<Int, 2>, 2, Dev> block_type_atom_all_bond_ranges,

    TView<Int, 2, Dev> block_type_tile_n_donH,
    TView<Int, 2, Dev> block_type_tile_n_acc,
    TView<Int, 3, Dev> block_type_tile_donH_inds,
    TView<Int, 3, Dev> block_type_tile_acc_inds,
    TView<Int, 3, Dev> block_type_tile_donor_type,
    TView<Int, 3, Dev> block_type_tile_acceptor_type,
    TView<Int, 3, Dev> block_type_tile_hybridization,
    TView<Int, 2, Dev> block_type_atom_is_hydrogen,

    // How many chemical bonds separate all pairs of atoms
    // within each block type?
    // Dimsize: n_block_types x max_n_atoms x max_n_atoms
    TView<Int, 3, Dev> block_type_path_distance,

    //////////////////////

    // HBond potential parameters
    TView<HBondPairParams<Real>, 2, Dev> pair_params,
    TView<HBondPolynomials<double>, 2, Dev> pair_polynomials,
    TView<HBondGlobalParams<Real>, 1, Dev> global_params,

    TView<Int, 2, Dev> dispatch_indices,  // from forward pass
    TView<Real, 1, Dev> dTdV              // nterms x nposes x len x len
    ) -> TPack<Vec<Real, 3>, 2, Dev>      // TODO: add extra dimension for terms
{
  using tmol::score::common::accumulate;
  using Real3 = Vec<Real, 3>;

  printf("CHECK0\n");

  int const n_poses = n_rots_for_pose.size(0);
  // int const max_n_pose_atoms = rot_coords.size(1);
  // int const max_n_blocks = pose_stack_block_type.size(1);
  int const max_n_conn = pose_stack_inter_residue_connections.size(2);
  int const n_block_types = block_type_n_atoms.size(0);
  int const max_n_block_atoms = block_type_atom_is_hydrogen.size(1);
  int const max_n_interblock_bonds =
      block_type_atoms_forming_chemical_bonds.size(1);
  int const max_n_all_bonds = block_type_all_bonds.size(1);
  int const max_n_tiles = block_type_tile_donH_inds.size(1);
  int const max_n_donH_per_tile = block_type_tile_donH_inds.size(2);
  int const max_n_acc_per_tile = block_type_tile_acc_inds.size(2);

  assert(max_n_interblock_bonds <= MAX_N_CONN);
  printf("CHECK1\n");

  auto n_rots = rot_coord_offset.size(0);
  // auto rot_max_n_atoms = rot_coords.size(1);
  auto n_atoms = rot_coords.size(0);

  auto dV_dcoords_t = TPack<Vec<Real, 3>, 2, Dev>::zeros({1, n_atoms});  // TODO
  auto dV_dcoords = dV_dcoords_t.view;

  auto scratch_rot_spheres_t = TPack<Real, 2, Dev>::zeros({n_rots, 4});
  auto scratch_rot_spheres = scratch_rot_spheres_t.view;

  auto scratch_rot_neighbors_t = TPack<Int, 3, Dev>::zeros(
      {n_poses, max_n_rots_per_pose, max_n_rots_per_pose});
  auto scratch_rot_neighbors = scratch_rot_neighbors_t.view;
  printf("CHECK2\n");

  // Optimal launch box on v100 and a100 is nt=32, vt=1
  LAUNCH_BOX_32;
  // Define nt and reduce_t
  CTA_REAL_REDUCE_T_TYPEDEF;

  auto eval_derivs = ([=] TMOL_DEVICE_FUNC(int cta) {
    auto hbond_atom_energy =
        ([=] TMOL_DEVICE_FUNC(
             int don_ind,
             int acc_ind,
             int don_tile_ind,
             int acc_tile_ind,
             int don_start,
             int acc_start,
             HBondSingleResData<Real> const &don_dat,
             HBondSingleResData<Real> const &acc_dat,
             HBondRotPairData<Dev, Real, Int> const &respair_dat,
             int cp_separation) {
          if (cp_separation < 5) {
            return Real(0.0);
          }
          Real val = hbond_atom_derivs_2<TILE_SIZE>(
              don_ind,
              acc_ind,
              don_tile_ind,
              acc_tile_ind,
              don_start,
              acc_start,
              don_dat,
              acc_dat,
              respair_dat,
              cp_separation,
              dTdV[cta],
              dV_dcoords);
          return Real(0.0);
        });

    auto score_inter_hbond_atom_pair = ([=] SCORE_INTER_HBOND_ATOM_PAIR_2);

    auto score_intra_hbond_atom_pair = ([=] SCORE_INTRA_HBOND_ATOM_PAIR_2);

    SHARED_MEMORY union shared_mem_union {
      shared_mem_union() {}
      HBondBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN> m;
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

    int const n_atoms1 = block_type_n_atoms[block_type1];
    int const n_atoms2 = block_type_n_atoms[block_type2];

    auto load_tile_invariant_interres_data =
        ([=] LOAD_TILE_INVARIANT_INTERRES_DATA_2);

    auto load_interres1_tile_data_to_shared =
        ([=] LOAD_INTERRES1_TILE_DATA_TO_SHARED_2);

    auto load_interres2_tile_data_to_shared =
        ([=] LOAD_INTERRES2_TILE_DATA_TO_SHARED_2);

    auto load_interres_data_from_shared =
        ([=] LOAD_INTERRES_DATA_FROM_SHARED_2);

    auto eval_interres_atom_pair_scores =
        ([=] EVAL_INTERRES_ATOM_PAIR_SCORES_2);

    auto store_calculated_energies =
        ([=] TMOL_DEVICE_FUNC(
             HBondScoringData2<Dev, Real, Int> & score_dat,
             shared_mem_union & shared) { ; });

    auto load_tile_invariant_intrares_data =
        ([=] LOAD_TILE_INVARIANT_INTRARES_DATA_2);

    auto load_intrares1_tile_data_to_shared =
        ([=] LOAD_INTRARES1_TILE_DATA_TO_SHARED_2);

    auto load_intrares2_tile_data_to_shared =
        ([=] LOAD_INTRARES2_TILE_DATA_TO_SHARED_2);

    auto load_intrares_data_from_shared =
        ([=] LOAD_INTRARES_DATA_FROM_SHARED_2);

    auto eval_intrares_atom_pair_scores =
        ([=] EVAL_INTRARES_ATOM_PAIR_SCORES_2);

    tmol::score::common::tile_evaluate_rot_pair<
        DeviceDispatch,
        Dev,
        HBondScoringData2<Dev, Real, Int>,
        HBondScoringData2<Dev, Real, Int>,
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
  // 1: launch a kernel to find a small bounding sphere surrounding the
  // blocks 2: launch a kernel to look for spheres that are within
  // striking distance of each other 3: launch a kernel to evaluate
  // lj/lk between pairs of blocks within striking distance

  // 0
  // TO DO: let DeviceDispatch hold a cuda stream (??)
  // at::cuda::CUDAStream wrapped_stream =
  // at::cuda::getDefaultCUDAStream(); mgpu::standard_context_t
  // context(wrapped_stream.stream());

  // 3
  DeviceDispatch<Dev>::template foreach_workgroup<launch_t>(
      dispatch_indices.size(1), eval_derivs);

  // for(int i = 0; i < dV_dcoords.size(1); i++) {
  // for(int j = 0; j < 3; j++)
  // printf("%f ", dV_dcoords[0][i][j]);
  // printf("\n");
  //}

  return dV_dcoords_t;
}

}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol
