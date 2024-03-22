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

#include <tmol/score/lk_ball/potentials/lk_ball.hh>
#include <tmol/score/lk_ball/potentials/params.hh>

// Operator definitions; safe for CPU compilation
#include <moderngpu/operators.hxx>

#include <chrono>

// Definitions for TILE_SIZE and MAX_N_WATERS
// Shared between this file and gen_pose_waters.impl.hh
#include <tmol/score/lk_ball/potentials/constants.hh>

namespace tmol {
namespace score {
namespace lk_ball {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

// TO DO: standardize tiled inter-block count pair
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

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device Dev,
    typename Real,
    typename Int>
class LKBallPoseScoreDispatch {
 public:
  static auto forward(
      TView<Vec<Real, 3>, 2, Dev> pose_coords,
      TView<Vec<Real, 3>, 3, Dev> water_coords,
      TView<Int, 2, Dev> pose_stack_block_coord_offset,
      TView<Int, 2, Dev> pose_stack_block_type,

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
      // TO DO: Rename since lots of atoms form chemical bonds
      TView<Int, 2, Dev> block_type_atoms_forming_chemical_bonds,

      TView<Int, 2, Dev> block_type_tile_n_polar_atoms,
      TView<Int, 2, Dev> block_type_tile_n_occluder_atoms,
      TView<Int, 3, Dev> block_type_tile_pol_occ_inds,
      TView<LKBallTypeParams<Real>, 3, Dev> block_type_tile_lk_ball_params,

      // How many chemical bonds separate all pairs of atoms
      // within each block type?
      // Dimsize: n_block_types x max_n_atoms x max_n_atoms
      TView<Int, 3, Dev> block_type_path_distance,

      //////////////////////

      // LKBall potential parameters
      TView<LKBallGlobalParams<Real>, 1, Dev> global_params,
      bool output_block_pair_energies)
      -> std::tuple<TPack<Real, 4, Dev>, TPack<Int, 3, Dev>> {
    using tmol::score::common::accumulate;
    using Real3 = Vec<Real, 3>;

    int const n_poses = pose_coords.size(0);
    int const max_n_pose_atoms = water_coords.size(1);
    int const max_n_blocks = pose_stack_block_type.size(1);
    int const max_n_conn = pose_stack_inter_residue_connections.size(2);
    int const n_block_types = block_type_n_atoms.size(0);
    int const max_n_block_atoms = block_type_path_distance.size(1);
    int const max_n_interblock_bonds =
        block_type_atoms_forming_chemical_bonds.size(1);
    int const max_n_tiles = block_type_tile_pol_occ_inds.size(1);

    assert(max_n_interblock_bonds <= MAX_N_CONN);

    assert(water_coords.size(0) == n_poses);
    assert(water_coords.size(1) == max_n_pose_atoms);
    assert(water_coords.size(2) == MAX_N_WATER);

    assert(pose_stack_block_coord_offset.size(0) == n_poses);
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

    assert(block_type_tile_n_polar_atoms.size(0) == n_block_types);
    assert(block_type_tile_n_polar_atoms.size(1) == max_n_tiles);
    assert(block_type_tile_n_occluder_atoms.size(0) == n_block_types);
    assert(block_type_tile_n_occluder_atoms.size(1) == max_n_tiles);
    assert(block_type_tile_pol_occ_inds.size(0) == n_block_types);
    assert(block_type_tile_pol_occ_inds.size(1) == max_n_tiles);
    assert(block_type_tile_pol_occ_inds.size(2) == TILE_SIZE);
    assert(block_type_tile_lk_ball_params.size(0) == n_block_types);
    assert(block_type_tile_lk_ball_params.size(1) == max_n_tiles);
    assert(block_type_tile_lk_ball_params.size(2) == TILE_SIZE);

    assert(block_type_path_distance.size(0) == n_block_types);
    assert(block_type_path_distance.size(1) == max_n_block_atoms);
    assert(block_type_path_distance.size(2) == max_n_block_atoms);

    TPack<Real, 4, Dev> output_t;
    if (output_block_pair_energies) {
      output_t = TPack<Real, 4, Dev>::zeros(
          {n_lk_ball_score_types, n_poses, max_n_blocks, max_n_blocks});
    } else {
      output_t =
          TPack<Real, 4, Dev>::zeros({n_lk_ball_score_types, n_poses, 1, 1});
    }
    auto output = output_t.view;

    auto scratch_block_spheres_t =
        TPack<Real, 3, Dev>::zeros({n_poses, max_n_blocks, 4});
    auto scratch_block_spheres = scratch_block_spheres_t.view;

    auto scratch_block_neighbors_t =
        TPack<Int, 3, Dev>::zeros({n_poses, max_n_blocks, max_n_blocks});
    auto scratch_block_neighbors = scratch_block_neighbors_t.view;

    // Optimal launch box on v100 and a100 is nt=32, vt=1
    LAUNCH_BOX_32;
    // Define nt and reduce_t
    CTA_REAL_REDUCE_T_TYPEDEF;

    auto eval_energies_by_block = ([=] TMOL_DEVICE_FUNC(int cta) {
      auto score_inter_lk_ball_atom_pair =
          ([=] TMOL_DEVICE_FUNC(
               int pol_start,
               int occ_start,
               int pol_ind,
               int occ_ind,
               LKBallScoringData<Real> &inter_dat,
               bool polar_first) {
            int pol_tile_ind = (polar_first ? inter_dat.r1 : inter_dat.r2)
                                   .pol_occ_tile_inds[pol_ind];
            int occ_tile_ind = (polar_first ? inter_dat.r2 : inter_dat.r1)
                                   .pol_occ_tile_inds[occ_ind];
            int separation = interres_count_pair_separation<TILE_SIZE>(
                inter_dat,
                (polar_first ? pol_tile_ind : occ_tile_ind),
                (polar_first ? occ_tile_ind : pol_tile_ind));
            lk_ball_Vt<Real> E = lk_ball_atom_energy_full<MAX_N_WATER>(
                pol_ind,
                occ_ind,
                pol_tile_ind,
                occ_tile_ind,
                pol_start,
                occ_start,
                polar_first ? inter_dat.r1 : inter_dat.r2,
                polar_first ? inter_dat.r2 : inter_dat.r1,
                inter_dat.pair_data,
                separation);
            inter_dat.pair_data.total_lk_ball_iso += E.lkball_iso;
            inter_dat.pair_data.total_lk_ball += E.lkball;
            inter_dat.pair_data.total_lk_bridge += E.lkbridge;
            inter_dat.pair_data.total_lk_bridge_uncpl += E.lkbridge_uncpl;
          });

      auto score_intra_lk_ball_atom_pair =
          ([=] TMOL_DEVICE_FUNC(
               int pol_start,
               int occ_start,
               int pol_ind,
               int occ_ind,
               LKBallScoringData<Real> &intra_dat,
               bool polar_first) {
            int pol_tile_ind = (polar_first ? intra_dat.r1 : intra_dat.r2)
                                   .pol_occ_tile_inds[pol_ind];
            int occ_tile_ind = (polar_first ? intra_dat.r2 : intra_dat.r1)
                                   .pol_occ_tile_inds[occ_ind];
            int const pol_atom_ind = pol_start + pol_tile_ind;
            int const occ_atom_ind = occ_start + occ_tile_ind;

            int const separation =
                block_type_path_distance[intra_dat.r1.block_type][pol_atom_ind]
                                        [occ_atom_ind];
            lk_ball_Vt<Real> E = lk_ball_atom_energy_full<MAX_N_WATER>(
                pol_ind,
                occ_ind,
                pol_tile_ind,
                occ_tile_ind,
                pol_start,
                occ_start,
                polar_first ? intra_dat.r1 : intra_dat.r2,
                polar_first ? intra_dat.r2 : intra_dat.r1,
                intra_dat.pair_data,
                separation);
            intra_dat.pair_data.total_lk_ball_iso += E.lkball_iso;
            intra_dat.pair_data.total_lk_ball += E.lkball;
            intra_dat.pair_data.total_lk_bridge += E.lkbridge;
            intra_dat.pair_data.total_lk_bridge_uncpl += E.lkbridge_uncpl;
          });

      SHARED_MEMORY union shared_mem_union {
        shared_mem_union() {}
        LKBallBlockPairSharedData<Real, TILE_SIZE, MAX_N_WATER, MAX_N_CONN> m;
        CTA_REAL_REDUCE_T_VARIABLE;

      } shared;

      int const pose_ind = cta / (max_n_blocks * max_n_blocks);
      int const block_ind_pair = cta % (max_n_blocks * max_n_blocks);
      int const block_ind1 = block_ind_pair / max_n_blocks;
      int const block_ind2 = block_ind_pair % max_n_blocks;
      if (block_ind1 > block_ind2) {
        return;
      }

      if (scratch_block_neighbors[pose_ind][block_ind1][block_ind2] == 0) {
        return;
      }

      int const max_important_bond_separation = 4;

      int const block_type1 = pose_stack_block_type[pose_ind][block_ind1];
      int const block_type2 = pose_stack_block_type[pose_ind][block_ind2];

      if (block_type1 < 0 || block_type2 < 0) {
        return;
      }

      int const n_atoms1 = block_type_n_atoms[block_type1];
      int const n_atoms2 = block_type_n_atoms[block_type2];

      auto load_tile_invariant_interres_data =
          ([=](int pose_ind,
               int block_ind1,
               int block_ind2,
               int block_type1,
               int block_type2,
               int n_atoms1,
               int n_atoms2,
               LKBallScoringData<Real> &inter_dat,
               shared_mem_union &shared) {
            lk_ball_load_tile_invariant_interres_data<DeviceDispatch, Dev, nt>(
                pose_stack_block_coord_offset,
                pose_stack_block_type,
                pose_stack_inter_residue_connections,
                pose_stack_min_bond_separation,
                pose_stack_inter_block_bondsep,

                block_type_n_interblock_bonds,
                block_type_atoms_forming_chemical_bonds,
                global_params,
                max_important_bond_separation,
                pose_ind,

                block_ind1,
                block_ind2,
                block_type1,
                block_type2,
                n_atoms1,

                n_atoms2,
                inter_dat,
                shared.m);
          });

      auto load_interres1_tile_data_to_shared =
          ([=](int tile_ind,
               int start_atom1,
               int n_atoms_to_load1,
               LKBallScoringData<Real> &inter_dat,
               shared_mem_union &shared) {
            lk_ball_load_interres1_tile_data_to_shared<DeviceDispatch, Dev, nt>(
                pose_coords,
                water_coords,
                block_type_tile_n_polar_atoms,
                block_type_tile_n_occluder_atoms,
                block_type_tile_pol_occ_inds,
                block_type_tile_lk_ball_params,
                block_type_path_distance,
                tile_ind,
                start_atom1,
                n_atoms_to_load1,
                inter_dat,
                shared.m);
          });

      auto load_interres2_tile_data_to_shared =
          ([=](int tile_ind,
               int start_atom2,
               int n_atoms_to_load2,
               LKBallScoringData<Real> &inter_dat,
               shared_mem_union &shared) {
            lk_ball_load_interres2_tile_data_to_shared<DeviceDispatch, Dev, nt>(
                pose_coords,
                water_coords,
                block_type_tile_n_polar_atoms,
                block_type_tile_n_occluder_atoms,
                block_type_tile_pol_occ_inds,
                block_type_tile_lk_ball_params,
                block_type_path_distance,
                tile_ind,
                start_atom2,
                n_atoms_to_load2,
                inter_dat,
                shared.m);
          });

      auto load_interres_data_from_shared =
          ([=](int, int, shared_mem_union &, LKBallScoringData<Real> &) {});

      auto eval_interres_atom_pair_scores =
          ([=](LKBallScoringData<Real> &inter_dat,
               int start_atom1,
               int start_atom2) {
            eval_interres_pol_occ_pair_energies<DeviceDispatch, Dev, nt>(
                inter_dat,
                start_atom1,
                start_atom2,
                score_inter_lk_ball_atom_pair);
          });

      auto store_calculated_energies = ([=](LKBallScoringData<Real> &score_dat,
                                            shared_mem_union &shared) {
        auto reduce_energies = ([&](int tid) {
          Real const cta_total_lk_ball_iso =
              DeviceDispatch<Dev>::template reduce_in_workgroup<nt>(
                  score_dat.pair_data.total_lk_ball_iso,
                  shared,
                  mgpu::plus_t<Real>());
          Real const cta_total_lk_ball =
              DeviceDispatch<Dev>::template reduce_in_workgroup<nt>(
                  score_dat.pair_data.total_lk_ball,
                  shared,
                  mgpu::plus_t<Real>());
          Real const cta_total_lk_bridge =
              DeviceDispatch<Dev>::template reduce_in_workgroup<nt>(
                  score_dat.pair_data.total_lk_bridge,
                  shared,
                  mgpu::plus_t<Real>());
          Real const cta_total_lk_bridge_uncpl =
              DeviceDispatch<Dev>::template reduce_in_workgroup<nt>(
                  score_dat.pair_data.total_lk_bridge_uncpl,
                  shared,
                  mgpu::plus_t<Real>());

          if (tid == 0) {
            if (!output_block_pair_energies) {
              accumulate<Dev, Real>::add(
                  output[w_lk_ball_iso][score_dat.pair_data.pose_ind][0][0],
                  cta_total_lk_ball_iso);
              accumulate<Dev, Real>::add(
                  output[w_lk_ball][score_dat.pair_data.pose_ind][0][0],
                  cta_total_lk_ball);
              accumulate<Dev, Real>::add(
                  output[w_lk_bridge][score_dat.pair_data.pose_ind][0][0],
                  cta_total_lk_bridge);
              accumulate<Dev, Real>::add(
                  output[w_lk_bridge_uncpl][score_dat.pair_data.pose_ind][0][0],
                  cta_total_lk_bridge_uncpl);
            } else {
              if (score_dat.r1.block_ind == score_dat.r2.block_ind) {
                output[w_lk_ball_iso][score_dat.pair_data.pose_ind]
                      [score_dat.r1.block_ind][score_dat.r1.block_ind] =
                          cta_total_lk_ball_iso;
                output[w_lk_ball][score_dat.pair_data.pose_ind]
                      [score_dat.r1.block_ind][score_dat.r1.block_ind] =
                          cta_total_lk_ball;
                output[w_lk_bridge][score_dat.pair_data.pose_ind]
                      [score_dat.r1.block_ind][score_dat.r1.block_ind] =
                          cta_total_lk_bridge;
                output[w_lk_bridge_uncpl][score_dat.pair_data.pose_ind]
                      [score_dat.r1.block_ind][score_dat.r1.block_ind] =
                          cta_total_lk_bridge_uncpl;
              } else {
                output[w_lk_ball_iso][score_dat.pair_data.pose_ind]
                      [score_dat.r1.block_ind][score_dat.r2.block_ind] =
                          0.5 * cta_total_lk_ball_iso;
                output[w_lk_ball_iso][score_dat.pair_data.pose_ind]
                      [score_dat.r2.block_ind][score_dat.r1.block_ind] =
                          0.5 * cta_total_lk_ball_iso;

                output[w_lk_ball][score_dat.pair_data.pose_ind]
                      [score_dat.r1.block_ind][score_dat.r2.block_ind] =
                          0.5 * cta_total_lk_ball;
                output[w_lk_ball][score_dat.pair_data.pose_ind]
                      [score_dat.r2.block_ind][score_dat.r1.block_ind] =
                          0.5 * cta_total_lk_ball;

                output[w_lk_bridge][score_dat.pair_data.pose_ind]
                      [score_dat.r1.block_ind][score_dat.r2.block_ind] =
                          0.5 * cta_total_lk_bridge;
                output[w_lk_bridge][score_dat.pair_data.pose_ind]
                      [score_dat.r2.block_ind][score_dat.r1.block_ind] =
                          0.5 * cta_total_lk_bridge;

                output[w_lk_bridge_uncpl][score_dat.pair_data.pose_ind]
                      [score_dat.r1.block_ind][score_dat.r2.block_ind] =
                          0.5 * cta_total_lk_bridge_uncpl;
                output[w_lk_bridge_uncpl][score_dat.pair_data.pose_ind]
                      [score_dat.r2.block_ind][score_dat.r1.block_ind] =
                          0.5 * cta_total_lk_bridge_uncpl;
              }
            }
          }
        });
        DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(
            reduce_energies);
      });

      auto load_tile_invariant_intrares_data =
          ([=](int pose_ind,
               int block_ind1,
               int block_type1,
               int n_atoms1,
               LKBallScoringData<Real> &intra_dat,
               shared_mem_union &shared) {
            lk_ball_load_tile_invariant_intrares_data<DeviceDispatch, Dev, nt>(
                pose_stack_block_coord_offset,
                pose_stack_block_type,
                global_params,
                max_important_bond_separation,
                pose_ind,
                block_ind1,
                block_type1,
                n_atoms1,
                intra_dat,
                shared.m);
          });

      auto load_intrares1_tile_data_to_shared =
          ([=](int tile_ind,
               int start_atom1,
               int n_atoms_to_load1,
               LKBallScoringData<Real> &intra_dat,
               shared_mem_union &shared) {
            lk_ball_load_intrares1_tile_data_to_shared<DeviceDispatch, Dev, nt>(
                pose_coords,
                water_coords,
                block_type_tile_n_polar_atoms,
                block_type_tile_n_occluder_atoms,
                block_type_tile_pol_occ_inds,
                block_type_tile_lk_ball_params,

                tile_ind,
                start_atom1,
                n_atoms_to_load1,
                intra_dat,
                shared.m);
          });

      auto load_intrares2_tile_data_to_shared =
          ([=](int tile_ind,
               int start_atom2,
               int n_atoms_to_load2,
               LKBallScoringData<Real> &intra_dat,
               shared_mem_union &shared) {
            lk_ball_load_intrares2_tile_data_to_shared<DeviceDispatch, Dev, nt>(
                pose_coords,
                water_coords,
                block_type_tile_n_polar_atoms,
                block_type_tile_n_occluder_atoms,
                block_type_tile_pol_occ_inds,
                block_type_tile_lk_ball_params,
                tile_ind,
                start_atom2,
                n_atoms_to_load2,
                intra_dat,
                shared.m);
          });

      auto load_intrares_data_from_shared =
          ([=](int tile_ind1,
               int tile_ind2,
               shared_mem_union &shared,
               LKBallScoringData<Real> &intra_dat) {
            lk_ball_load_intrares_data_from_shared(
                tile_ind1, tile_ind2, shared.m, intra_dat);
          });

      auto eval_intrares_atom_pair_scores =
          ([=](LKBallScoringData<Real> &intra_dat,
               int start_atom1,
               int start_atom2) {
            eval_intrares_pol_occ_pair_energies<DeviceDispatch, Dev, nt>(
                intra_dat,
                start_atom1,
                start_atom2,
                score_intra_lk_ball_atom_pair);
          });

      tmol::score::common::tile_evaluate_block_pair<
          DeviceDispatch,
          Dev,
          LKBallScoringData<Real>,
          LKBallScoringData<Real>,
          Real,
          TILE_SIZE>(
          shared,
          pose_ind,
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
    // lk-ball desolvation between pairs of blocks within striking distance

    // 0
    // TO DO: let DeviceDispatch hold a cuda stream (??)
    // at::cuda::CUDAStream wrapped_stream =
    // at::cuda::getDefaultCUDAStream(); mgpu::standard_context_t
    // context(wrapped_stream.stream());
    int const n_block_pairs = n_poses * max_n_blocks * max_n_blocks;

    score::common::sphere_overlap::
        compute_block_spheres<DeviceDispatch, Dev, Real, Int>::f(
            pose_coords,
            pose_stack_block_coord_offset,
            pose_stack_block_type,
            block_type_n_atoms,
            scratch_block_spheres);

    score::common::sphere_overlap::
        detect_block_neighbors<DeviceDispatch, Dev, Real, Int>::f(
            pose_coords,
            pose_stack_block_coord_offset,
            pose_stack_block_type,
            block_type_n_atoms,
            scratch_block_spheres,
            scratch_block_neighbors,
            Real(6.0));  // 6.0 hard coded here. Please fix! TEMP!

    // 3 Only the forward pass in this calculation
    DeviceDispatch<Dev>::template foreach_workgroup<launch_t>(
        n_block_pairs, eval_energies_by_block);

    return {output_t, scratch_block_neighbors_t};
  }

  static auto backward(
      TView<Vec<Real, 3>, 2, Dev> pose_coords,
      TView<Vec<Real, 3>, 3, Dev> water_coords,
      TView<Int, 2, Dev> pose_stack_block_coord_offset,
      TView<Int, 2, Dev> pose_stack_block_type,

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

      TView<Int, 2, Dev> block_type_tile_n_polar_atoms,
      TView<Int, 2, Dev> block_type_tile_n_occluder_atoms,
      TView<Int, 3, Dev> block_type_tile_pol_occ_inds,
      TView<LKBallTypeParams<Real>, 3, Dev> block_type_tile_lk_ball_params,

      // How many chemical bonds separate all pairs of atoms
      // within each block type?
      // Dimsize: n_block_types x max_n_atoms x max_n_atoms
      TView<Int, 3, Dev> block_type_path_distance,

      //////////////////////

      // LKBall potential parameters
      TView<LKBallGlobalParams<Real>, 1, Dev> global_params,
      TView<Int, 3, Dev> scratch_block_neighbors,  // from forward pass
      TView<Real, 4, Dev> dTdV,
      bool block_pair_scoring)
      -> std::tuple<TPack<Vec<Real, 3>, 2, Dev>, TPack<Vec<Real, 3>, 3, Dev>> {
    // std::cout << "d lkball start" << std::endl;
    using tmol::score::common::accumulate;
    using Real3 = Vec<Real, 3>;

    int const n_poses = pose_coords.size(0);
    int const max_n_pose_atoms = water_coords.size(1);
    int const max_n_blocks = pose_stack_block_type.size(1);
    int const max_n_conn = pose_stack_inter_residue_connections.size(2);
    int const n_block_types = block_type_n_atoms.size(0);
    int const max_n_block_atoms = block_type_path_distance.size(1);
    int const max_n_interblock_bonds =
        block_type_atoms_forming_chemical_bonds.size(1);
    int const max_n_tiles = block_type_tile_pol_occ_inds.size(1);

    assert(max_n_interblock_bonds <= MAX_N_CONN);

    assert(water_coords.size(0) == n_poses);
    assert(water_coords.size(1) == max_n_pose_atoms);
    assert(water_coords.size(2) == MAX_N_WATER);

    assert(pose_stack_block_coord_offset.size(0) == n_poses);
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

    assert(block_type_tile_n_polar_atoms.size(0) == n_block_types);
    assert(block_type_tile_n_polar_atoms.size(1) == max_n_tiles);
    assert(block_type_tile_n_occluder_atoms.size(0) == n_block_types);
    assert(block_type_tile_n_occluder_atoms.size(1) == max_n_tiles);
    assert(block_type_tile_pol_occ_inds.size(0) == n_block_types);
    assert(block_type_tile_pol_occ_inds.size(1) == max_n_tiles);
    assert(block_type_tile_pol_occ_inds.size(2) == TILE_SIZE);
    assert(block_type_tile_lk_ball_params.size(0) == n_block_types);
    assert(block_type_tile_lk_ball_params.size(1) == max_n_tiles);
    assert(block_type_tile_lk_ball_params.size(2) == TILE_SIZE);

    assert(block_type_path_distance.size(0) == n_block_types);
    assert(block_type_path_distance.size(1) == max_n_block_atoms);
    assert(block_type_path_distance.size(2) == max_n_block_atoms);

    assert(scratch_block_neighbors.size(0) == n_poses);
    assert(scratch_block_neighbors.size(1) == max_n_blocks);
    assert(scratch_block_neighbors.size(2) == max_n_blocks);

    assert(dTdV.size(0) == 4);
    assert(dTdV.size(1) == n_poses);

    auto dV_d_pose_coords_t =
        TPack<Vec<Real, 3>, 2, Dev>::zeros({n_poses, max_n_pose_atoms});
    auto dV_d_pose_coords = dV_d_pose_coords_t.view;

    auto dV_d_water_coords_t = TPack<Vec<Real, 3>, 3, Dev>::zeros(
        {n_poses, max_n_pose_atoms, MAX_N_WATER});
    auto dV_d_water_coords = dV_d_water_coords_t.view;

    // Optimal launch box on v100 and a100 is nt=32, vt=1
    LAUNCH_BOX_32;
    // Define nt and reduce_t
    CTA_REAL_REDUCE_T_TYPEDEF;

    auto eval_derivs = ([=] TMOL_DEVICE_FUNC(int cta) {
      auto lk_ball_atom_derivs =
          ([=] TMOL_DEVICE_FUNC(
               int pol_ind,
               int occ_ind,
               int pol_tile_ind,
               int occ_tile_ind,
               int pol_start,
               int occ_start,
               LKBallSingleResData<Real> const &pol_dat,
               LKBallSingleResData<Real> const &occ_dat,
               LKBallResPairData<Real> const &respair_dat,
               int cp_separation) {
            // capture dTdV, dV_d_pose_coords, & dV_d_water_coords

            lk_ball_atom_derivs_full<TILE_SIZE, MAX_N_WATER>(
                pol_ind,
                occ_ind,
                pol_tile_ind,
                occ_tile_ind,
                pol_start,
                occ_start,
                pol_dat,
                occ_dat,
                respair_dat,
                cp_separation,
                dTdV,
                dV_d_pose_coords,
                dV_d_water_coords,
                block_pair_scoring);
          });

      auto dscore_inter_lk_ball_atom_pair =
          ([=] TMOL_DEVICE_FUNC(
               int pol_start,
               int occ_start,
               int pol_ind,
               int occ_ind,
               LKBallScoringData<Real> const &inter_dat,
               bool polar_first) {
            int pol_tile_ind = (polar_first ? inter_dat.r1 : inter_dat.r2)
                                   .pol_occ_tile_inds[pol_ind];
            int occ_tile_ind = (polar_first ? inter_dat.r2 : inter_dat.r1)
                                   .pol_occ_tile_inds[occ_ind];
            int separation = interres_count_pair_separation<TILE_SIZE>(
                inter_dat,
                (polar_first ? pol_tile_ind : occ_tile_ind),
                (polar_first ? occ_tile_ind : pol_tile_ind));
            lk_ball_atom_derivs(
                pol_ind,
                occ_ind,
                pol_tile_ind,
                occ_tile_ind,
                pol_start,
                occ_start,
                polar_first ? inter_dat.r1 : inter_dat.r2,
                polar_first ? inter_dat.r2 : inter_dat.r1,
                inter_dat.pair_data,
                separation);
          });

      auto dscore_intra_lk_ball_atom_pair =
          ([=] TMOL_DEVICE_FUNC(
               int pol_start,
               int occ_start,
               int pol_ind,
               int occ_ind,
               LKBallScoringData<Real> const &intra_dat,
               bool polar_first) {
            int pol_tile_ind = (polar_first ? intra_dat.r1 : intra_dat.r2)
                                   .pol_occ_tile_inds[pol_ind];
            int occ_tile_ind = (polar_first ? intra_dat.r2 : intra_dat.r1)
                                   .pol_occ_tile_inds[occ_ind];
            int const pol_atom_ind = pol_start + pol_tile_ind;
            int const occ_atom_ind = occ_start + occ_tile_ind;

            int const separation =
                block_type_path_distance[intra_dat.r1.block_type][pol_atom_ind]
                                        [occ_atom_ind];
            return lk_ball_atom_derivs(
                pol_ind,
                occ_ind,
                pol_tile_ind,
                occ_tile_ind,
                pol_start,
                occ_start,
                polar_first ? intra_dat.r1 : intra_dat.r2,
                polar_first ? intra_dat.r2 : intra_dat.r1,
                intra_dat.pair_data,
                separation);
          });

      SHARED_MEMORY union shared_mem_union {
        shared_mem_union() {}
        LKBallBlockPairSharedData<Real, TILE_SIZE, MAX_N_WATER, MAX_N_CONN> m;
        CTA_REAL_REDUCE_T_VARIABLE;

      } shared;

      int const pose_ind = cta / (max_n_blocks * max_n_blocks);
      int const block_ind_pair = cta % (max_n_blocks * max_n_blocks);
      int const block_ind1 = block_ind_pair / max_n_blocks;
      int const block_ind2 = block_ind_pair % max_n_blocks;
      if (block_ind1 > block_ind2) {
        return;
      }

      if (scratch_block_neighbors[pose_ind][block_ind1][block_ind2] == 0) {
        return;
      }

      int const max_important_bond_separation = 4;

      int const block_type1 = pose_stack_block_type[pose_ind][block_ind1];
      int const block_type2 = pose_stack_block_type[pose_ind][block_ind2];

      if (block_type1 < 0 || block_type2 < 0) {
        return;
      }

      int const n_atoms1 = block_type_n_atoms[block_type1];
      int const n_atoms2 = block_type_n_atoms[block_type2];

      auto load_tile_invariant_interres_data =
          ([=](int pose_ind,
               int block_ind1,
               int block_ind2,
               int block_type1,
               int block_type2,
               int n_atoms1,
               int n_atoms2,
               LKBallScoringData<Real> &inter_dat,
               shared_mem_union &shared) {
            lk_ball_load_tile_invariant_interres_data<DeviceDispatch, Dev, nt>(
                pose_stack_block_coord_offset,
                pose_stack_block_type,
                pose_stack_inter_residue_connections,
                pose_stack_min_bond_separation,
                pose_stack_inter_block_bondsep,

                block_type_n_interblock_bonds,
                block_type_atoms_forming_chemical_bonds,
                global_params,
                max_important_bond_separation,
                pose_ind,

                block_ind1,
                block_ind2,
                block_type1,
                block_type2,
                n_atoms1,

                n_atoms2,
                inter_dat,
                shared.m);
          });

      auto load_interres1_tile_data_to_shared =
          ([=](int tile_ind,
               int start_atom1,
               int n_atoms_to_load1,
               LKBallScoringData<Real> &inter_dat,
               shared_mem_union &shared) {
            lk_ball_load_interres1_tile_data_to_shared<DeviceDispatch, Dev, nt>(
                pose_coords,
                water_coords,
                block_type_tile_n_polar_atoms,
                block_type_tile_n_occluder_atoms,
                block_type_tile_pol_occ_inds,
                block_type_tile_lk_ball_params,
                block_type_path_distance,
                tile_ind,
                start_atom1,
                n_atoms_to_load1,
                inter_dat,
                shared.m);
          });

      auto load_interres2_tile_data_to_shared =
          ([=](int tile_ind,
               int start_atom2,
               int n_atoms_to_load2,
               LKBallScoringData<Real> &inter_dat,
               shared_mem_union &shared) {
            lk_ball_load_interres2_tile_data_to_shared<DeviceDispatch, Dev, nt>(
                pose_coords,
                water_coords,
                block_type_tile_n_polar_atoms,
                block_type_tile_n_occluder_atoms,
                block_type_tile_pol_occ_inds,
                block_type_tile_lk_ball_params,
                block_type_path_distance,
                tile_ind,
                start_atom2,
                n_atoms_to_load2,
                inter_dat,
                shared.m);
          });

      auto load_interres_data_from_shared =
          ([=](int, int, shared_mem_union &, LKBallScoringData<Real> &) {});

      auto eval_interres_atom_pair_scores =
          ([=](LKBallScoringData<Real> &inter_dat,
               int start_atom1,
               int start_atom2) {
            eval_interres_pol_occ_pair_energies<DeviceDispatch, Dev, nt>(
                inter_dat,
                start_atom1,
                start_atom2,
                dscore_inter_lk_ball_atom_pair);
          });

      auto store_calculated_energies =
          ([=](LKBallScoringData<Real> &score_dat, shared_mem_union &shared) {
            // no op; only derivs, no scoring
          });

      auto load_tile_invariant_intrares_data =
          ([=](int pose_ind,
               int block_ind1,
               int block_type1,
               int n_atoms1,
               LKBallScoringData<Real> &intra_dat,
               shared_mem_union &shared) {
            lk_ball_load_tile_invariant_intrares_data<DeviceDispatch, Dev, nt>(
                pose_stack_block_coord_offset,
                pose_stack_block_type,
                global_params,
                max_important_bond_separation,
                pose_ind,
                block_ind1,
                block_type1,
                n_atoms1,
                intra_dat,
                shared.m);
          });

      auto load_intrares1_tile_data_to_shared =
          ([=](int tile_ind,
               int start_atom1,
               int n_atoms_to_load1,
               LKBallScoringData<Real> &intra_dat,
               shared_mem_union &shared) {
            lk_ball_load_intrares1_tile_data_to_shared<DeviceDispatch, Dev, nt>(
                pose_coords,
                water_coords,
                block_type_tile_n_polar_atoms,
                block_type_tile_n_occluder_atoms,
                block_type_tile_pol_occ_inds,
                block_type_tile_lk_ball_params,

                tile_ind,
                start_atom1,
                n_atoms_to_load1,
                intra_dat,
                shared.m);
          });

      auto load_intrares2_tile_data_to_shared =
          ([=](int tile_ind,
               int start_atom2,
               int n_atoms_to_load2,
               LKBallScoringData<Real> &intra_dat,
               shared_mem_union &shared) {
            lk_ball_load_intrares2_tile_data_to_shared<DeviceDispatch, Dev, nt>(
                pose_coords,
                water_coords,
                block_type_tile_n_polar_atoms,
                block_type_tile_n_occluder_atoms,
                block_type_tile_pol_occ_inds,
                block_type_tile_lk_ball_params,
                tile_ind,
                start_atom2,
                n_atoms_to_load2,
                intra_dat,
                shared.m);
          });

      auto load_intrares_data_from_shared =
          ([=](int tile_ind1,
               int tile_ind2,
               shared_mem_union &shared,
               LKBallScoringData<Real> &intra_dat) {
            lk_ball_load_intrares_data_from_shared(
                tile_ind1, tile_ind2, shared.m, intra_dat);
          });

      auto eval_intrares_atom_pair_scores =
          ([=](LKBallScoringData<Real> &intra_dat,
               int start_atom1,
               int start_atom2) {
            eval_intrares_pol_occ_pair_energies<DeviceDispatch, Dev, nt>(
                intra_dat,
                start_atom1,
                start_atom2,
                dscore_intra_lk_ball_atom_pair);
          });

      tmol::score::common::tile_evaluate_block_pair<
          DeviceDispatch,
          Dev,
          LKBallScoringData<Real>,
          LKBallScoringData<Real>,
          Real,
          TILE_SIZE>(
          shared,
          pose_ind,
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
    int const n_block_pairs = n_poses * max_n_blocks * max_n_blocks;
    DeviceDispatch<Dev>::template foreach_workgroup<launch_t>(
        n_block_pairs, eval_derivs);
    // std::cout << "d lkball end" << std::endl;

    return {dV_d_pose_coords_t, dV_d_water_coords_t};
  }
};

}  // namespace potentials
}  // namespace lk_ball
}  // namespace score
}  // namespace tmol
