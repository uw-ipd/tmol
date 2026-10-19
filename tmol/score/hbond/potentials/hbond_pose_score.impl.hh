#pragma once

#include <tmol/utility/portable/array.hh>
#include <tmol/utility/portable/reduce_ops.hh>
#include <tmol/utility/portable/math.hh>
#include <tmol/utility/portable/matrix.hh>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/count_pair.hh>
#include <tmol/score/common/counting.hh>
#include <tmol/score/common/data_loading.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/geom.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/common/pose_total_reduce.hh>
#include <tmol/score/common/sphere_overlap.impl.hh>
#include <tmol/score/common/tile_atom_pair_evaluation.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/upper_triangle_indices.hh>
#include <tmol/score/common/warp_segreduce.hh>
#include <tmol/score/common/warp_stride_reduce.hh>

#include <tmol/score/hbond/potentials/hbond.hh>
#include <tmol/score/hbond/potentials/hbond_ctx_tensor.hh>
#include <tmol/score/hbond/potentials/hbond_kernels.hh>
#include <tmol/score/hbond/potentials/hbond_mps_launch.hh>
#include <tmol/score/hbond/potentials/params.hh>
#include <tmol/score/hbond/potentials/hbond_pose_score.hh>

// Operator definitions; safe for CPU compilation

#include <chrono>
#include <type_traits>


namespace tmol {
namespace score {
namespace hbond {
namespace potentials {

using tmol::portable::Vec;


template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    typename Real,
    typename Int>
auto HBondPoseScoreDispatch<DeviceDispatch, Dev, Real, Int>::forward(
    ContextManager& mgr,
    // common params
    TView<Vec<Real, 3>, 1, Dev> rot_coords,
    TView<Int, 1, Dev> rot_coord_offset,
    TView<Int, 1, Dev> pose_ind_for_atom,
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

    // InterBlockBondsep: [pose, block1, slot, (block2, min separation)]
    // and [pose, block1, slot, conn1, conn2]
    TView<Int, 4, Dev> pose_stack_near_blocks,
    TView<int8_t, 5, Dev> pose_stack_inter_block_bondsep,

    //////////////////////
    // Chemical properties
    // how many atoms for a given block
    // Dimsize n_block_types
    TView<Int, 1, Dev> block_type_n_atoms,  // ?? needed ?? I think so

    // how many inter-block chemical bonds are there
    // Dimsize: n_block_types
    TView<Int, 1, Dev> block_type_n_interblock_bonds,

    // what atoms form the inter-block chemical bonds
    // Dimsize: n_block_types x max_n_interblock_bonds
    TView<Int, 2, Dev> block_type_atoms_forming_chemical_bonds,

    TView<Int, 1, Dev> block_type_n_all_bonds,
    TView<Vec<Int, 3>, 2, Dev> block_type_all_bonds,
    TView<Vec<Int, 2>, 2, Dev> block_type_atom_all_bond_ranges,
    // How many chemical bonds separate all pairs of atoms
    // within each block type?
    // Dimsize: n_block_types x max_n_atoms x max_n_atoms
    TView<Int, 3, Dev> block_type_path_distance,

    TView<Int, 2, Dev> block_type_tile_n_donH,
    TView<Int, 2, Dev> block_type_tile_n_acc,
    TView<Int, 3, Dev> block_type_tile_donH_inds,
    TView<Int, 3, Dev> block_type_tile_acc_inds,
    TView<Int, 3, Dev> block_type_tile_donor_type,
    TView<Int, 3, Dev> block_type_tile_acceptor_type,
    TView<Int, 3, Dev> block_type_tile_hybridization,
    TView<Int, 2, Dev> block_type_atom_is_hydrogen,

    // HBond potential parameters
    TView<HBondPairParams<Real>, 2, Dev> pair_params,
    TView<HBondPolysFor<Dev>, 2, Dev> pair_polynomials,
    TView<HBondGlobalParams<Real>, 1, Dev> global_params,

    // Derived-atom coords + source-atom indices produced by the
    // hbond pre-pass kernel (GenerateHBondBases).
    TView<Vec<Real, 3>, 2, Dev> derived_coords,
    TView<Int, 2, Dev> derived_atom_inds,

    TView<Int, 1, Dev> shared_compact_block_neighbors,
    bool output_block_pair_energies,
    bool compute_derivs,
    bool allow_split_pairs

    )
    -> std::tuple<
        TPack<Real, 4, Dev>,
        TPack<Vec<Real, 3>, 2, Dev>,
        TPack<Int, 3, Dev>> {
  using tmol::score::common::accumulate;
  using Real3 = Vec<Real, 3>;

  int const n_atoms = rot_coords.size(0);
  int const n_rots = rot_coord_offset.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);

  int const max_n_conn = pose_stack_inter_residue_connections.size(2);
  int const n_block_types = block_type_n_atoms.size(0);
  int const max_n_block_atoms = block_type_atom_is_hydrogen.size(1);
  int const max_n_interblock_bonds =
      block_type_atoms_forming_chemical_bonds.size(1);
  int const max_n_all_bonds = block_type_all_bonds.size(1);
  int const max_n_tiles = block_type_tile_donH_inds.size(1);
  int const max_n_donH_per_tile = block_type_tile_donH_inds.size(2);
  int const max_n_acc_per_tile = block_type_tile_acc_inds.size(2);

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

  assert(pose_stack_inter_residue_connections.size(0) == n_poses);
  assert(pose_stack_inter_residue_connections.size(1) == max_n_blocks);

  assert(pose_stack_near_blocks.size(0) == n_poses);
  assert(pose_stack_near_blocks.size(1) == max_n_blocks);
  assert(pose_stack_near_blocks.size(3) == 2);

  assert(pose_stack_inter_block_bondsep.size(0) == n_poses);
  assert(pose_stack_inter_block_bondsep.size(1) == max_n_blocks);
  assert(
      pose_stack_inter_block_bondsep.size(2) == pose_stack_near_blocks.size(2));
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
  assert(block_type_path_distance.size(2) == max_n_block_atoms);

  assert(max_n_interblock_bonds <= MAX_N_CONN);

  // Block-pair autograd recomputes coordinate derivatives in backward.
  bool const accumulate_derivs = compute_derivs && !output_block_pair_energies;
  auto dV_dcoords_t = accumulate_derivs
                          ? TPack<Vec<Real, 3>, 2, Dev>::zeros({1, n_atoms})
                          : TPack<Vec<Real, 3>, 2, Dev>::empty({1, 0});
  auto dV_dcoords = dV_dcoords_t.view;

  bool const use_shared_compact_block_neighbors =
      shared_compact_block_neighbors.size(0) != 0;
  TPack<Real, 3, Dev> scratch_rot_spheres_t;
  TPack<Int, 3, Dev> scratch_rot_neighbors_t;
  TView<Real, 3, Dev> scratch_rot_spheres;
  TView<Int, 3, Dev> scratch_rot_neighbors;
  if (!use_shared_compact_block_neighbors) {
    scratch_rot_spheres_t =
        Dev == Device::CPU
            ? TPack<Real, 3, Dev>::zeros({n_poses, max_n_blocks, 4})
            : TPack<Real, 3, Dev>::empty({n_poses, max_n_blocks, 4});
    scratch_rot_neighbors_t =
        Dev == Device::CPU
            ? TPack<Int, 3, Dev>::zeros({n_poses, max_n_blocks, max_n_blocks})
            : TPack<Int, 3, Dev>::empty({n_poses, max_n_blocks, max_n_blocks});
    scratch_rot_spheres = scratch_rot_spheres_t.view;
    scratch_rot_neighbors = scratch_rot_neighbors_t.view;

    score::common::sphere_overlap::
        compute_block_spheres<DeviceDispatch, Dev, Real, Int>::f(
            mgr,
            rot_coords,
            rot_coord_offset,
            block_ind_for_rot,
            pose_ind_for_rot,
            block_type_ind_for_rot,
            block_type_n_atoms,
            scratch_rot_spheres);

    score::common::sphere_overlap::
        detect_block_neighbors<DeviceDispatch, Dev, Real, Int>::f(
            mgr,
            first_rot_block_type,
            scratch_rot_spheres,
            scratch_rot_neighbors,
            Real(5.5));
  }

  // Metal accumulates whole-pose totals into per-block-pair cells and reduces
  // them afterwards, rather than atomically adding into one cell.
  //
  // One atomic per threadgroup is still thousands of them per pose arriving in
  // an order the GPU is free to vary, and float addition is not associative.
  // The resulting jitter is about one float32 ULP of the pose total: small,
  // but a line search that validates its analytic derivative against a finite
  // difference cannot tolerate a non-reproducible energy. Writing each
  // threadgroup's partial to its own cell removes the atomic entirely, and
  // summing the cells is a fixed-order reduction. The cells are the ones
  // block-pair scoring already writes, so this reuses the store that is there.
  // The CPU and CUDA paths are unchanged. See elec_pose_score.impl.hh, which
  // solved this first.
  bool const mps_cell_reduction =
      Dev == tmol::Device::MPS && !output_block_pair_energies;
  TPack<Real, 4, Dev> output_t;
  TPack<Real, 4, Dev> cells_t;
  if (output_block_pair_energies) {
    output_t =
        TPack<Real, 4, Dev>::zeros({1, n_poses, max_n_blocks, max_n_blocks});
  } else {
    output_t = TPack<Real, 4, Dev>::zeros({1, n_poses, 1, 1});
    if (mps_cell_reduction) {
      cells_t =
          TPack<Real, 4, Dev>::zeros({1, n_poses, max_n_blocks, max_n_blocks});
    }
  }
  auto output = mps_cell_reduction ? cells_t.view : output_t.view;

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

  // The total number of unique block pairs (including self-pairs)
  int const max_n_upper_triangle_inds = score::common::checked_triangular_size(
      max_n_blocks, true, "hydrogen-bond block-pair dispatch");

  // State the kernel bodies read, named explicitly rather than captured
  // implicitly by [=]; the Metal backend has nowhere for a closure to live, so
  // the tile machinery in hbond_kernels.hh reads this struct instead. One
  // context serves both kernel families: whole-pose scoring writes
  // [term, pose, block, block] and rotamer scoring [term, rot_pair].
  HBondCtx<Real, Int, Dev, 4, 3, 2> ctx;
  ctx.rot_coords = rot_coords;
  ctx.rot_coord_offset = rot_coord_offset;
  ctx.block_ind_for_rot = block_ind_for_rot;
  ctx.block_type_ind_for_rot = block_type_ind_for_rot;
  ctx.rot_offset_for_block = rot_offset_for_block;
  ctx.pose_stack_near_blocks = pose_stack_near_blocks;
  ctx.pose_stack_inter_block_bondsep = pose_stack_inter_block_bondsep;
  ctx.block_type_n_atoms = block_type_n_atoms;
  ctx.block_type_n_interblock_bonds = block_type_n_interblock_bonds;
  ctx.block_type_atoms_forming_chemical_bonds =
      block_type_atoms_forming_chemical_bonds;
  ctx.block_type_path_distance = block_type_path_distance;
  ctx.block_type_tile_n_donH = block_type_tile_n_donH;
  ctx.block_type_tile_n_acc = block_type_tile_n_acc;
  ctx.block_type_tile_donH_inds = block_type_tile_donH_inds;
  ctx.block_type_tile_acc_inds = block_type_tile_acc_inds;
  ctx.block_type_tile_donor_type = block_type_tile_donor_type;
  ctx.block_type_tile_acceptor_type = block_type_tile_acceptor_type;
  ctx.block_type_tile_hybridization = block_type_tile_hybridization;
  ctx.pair_params = pair_params;
  ctx.pair_polynomials = pair_polynomials;
  ctx.global_params = global_params;
  ctx.derived_coords = derived_coords;
  ctx.derived_atom_inds = derived_atom_inds;
  ctx.dispatch_indices = scratch_rot_neighbors;
  ctx.dV_dcoords = dV_dcoords;
  ctx.max_n_blocks = max_n_blocks;
  ctx.output_block_pair_energies = output_block_pair_energies;
  ctx.accumulate_derivs = accumulate_derivs;
  ctx.output = output;
  ctx.dTdV = output;
  ctx.max_n_upper_triangle_inds = max_n_upper_triangle_inds;
  ctx.use_shared_compact_block_neighbors = use_shared_compact_block_neighbors;
  // Empty on CPU and CUDA; on Metal it carries the thread index and the
  // reduction scratch the workgroup primitives need. See workgroup_lane.hh.
  WorkgroupLane<Dev> lane;

  using PairMode = common::TilePairMode;
#ifdef __NVCC__
  auto eval_energies_impl =
      ([=] TMOL_DEVICE_FUNC(
           int cta, auto derivative_tag, auto pair_mode_tag) {
        constexpr auto Mode = decltype(pair_mode_tag)::value;
        typedef HBondSharedMem<Real, TILE_SIZE, MAX_N_CONN, launch_t>
            shared_mem_union;
        SHARED_MEMORY shared_mem_union shared;
        hbond_pose_eval_energies<
            DeviceDispatch,
            Dev,
            Real,
            Int,
            launch_t,
            Mode>(ctx, shared, cta, derivative_tag);
      });
#else
  auto eval_energies = ([=] TMOL_DEVICE_FUNC(int cta) {
    typedef HBondSharedMem<Real, TILE_SIZE, MAX_N_CONN, launch_t>
        shared_mem_union;
    SHARED_MEMORY shared_mem_union shared;
    hbond_pose_eval_energies<
        DeviceDispatch,
        Dev,
        Real,
        Int,
        launch_t,
        PairMode::InterAndIntra>(ctx, shared, cta, accumulate_derivs);
  });
#endif

#ifdef __NVCC__
  auto launch_energies = [&](auto eval_energies) {
#endif
    if (Dev == tmol::Device::MPS) {
      // Metal cannot launch a device lambda -- there is nowhere for a closure
      // to live, and entry points are compiled by a separate toolchain -- so
      // the same kernel body is reached through a named entry instead. The
      // context the lambda closes over is the same object either way.
      //
      // Tested ahead of the compact-block-neighbor path below because that one
      // launches through DeviceDispatch, which on this backend would run the
      // kernel on the CPU.
      hbond_mps_launch_if<Dev>(
          accumulate_derivs ? "hbond_pose_eval_energies_derivs_f32"
                            : "hbond_pose_eval_energies_f32",
          ctx,
          int64_t(n_poses) * int64_t(max_n_upper_triangle_inds));
      if (mps_cell_reduction) {
        // Fixed-order reduction over the cells, so the pose total is
        // bit-reproducible across calls with the same shape.
        score::common::pose_total_reduce<DeviceDispatch, Dev, Real>::f(
            mgr, cells_t.view, output_t.view);
      }
    } else if (
        !output_block_pair_energies && use_shared_compact_block_neighbors) {
      score::common::sphere_overlap::launch_precomputed_block_neighbors<
          DeviceDispatch,
          Dev,
          launch_t,
          Int>(
          mgr,
          shared_compact_block_neighbors,
          n_poses,
          max_n_blocks,
          eval_energies);
    }
#ifdef __NVCC__
    else if (
        !output_block_pair_energies
        && score::common::sphere_overlap::should_compact_block_neighbors(
            n_poses, max_n_blocks, compute_derivs)) {
      score::common::sphere_overlap::
          launch_compact_block_neighbors<DeviceDispatch, Dev, launch_t, Int>(
              mgr, scratch_rot_neighbors, eval_energies);
    }
#endif
    else {
      DeviceDispatch<Dev>::template foreach_pose_workgroup<launch_t>(
          mgr, n_poses, max_n_upper_triangle_inds, eval_energies);
    }

#ifdef __NVCC__
  };
  // Split derivative evaluation only when the stack is large enough to
  // amortize the second pass. Both passes preserve the original candidate IDs.
  bool const split_pairs =
      allow_split_pairs && !output_block_pair_energies
      && use_shared_compact_block_neighbors && max_n_blocks > 1
      && int64_t(n_poses) * max_n_blocks >= 4096
      && shared_compact_block_neighbors.size(0) - 1 >= (1 << 15);
  // Keep the combined latency path for small stacks. Specializing only the
  // HBond score kernel can slow subsequent terms in this workload range.
  if (accumulate_derivs || n_rots < 512) {
    if (accumulate_derivs && split_pairs) {
      auto eval_inter = ([=] TMOL_DEVICE_FUNC(int cta) {
        eval_energies_impl(
            cta,
            std::true_type{},
            std::integral_constant<PairMode, PairMode::Inter>{});
      });
      launch_energies(eval_inter);
      auto eval_intra = ([=] TMOL_DEVICE_FUNC(int cta) {
        eval_energies_impl(
            cta,
            std::true_type{},
            std::integral_constant<PairMode, PairMode::Intra>{});
      });
      launch_energies(eval_intra);
    } else {
      auto eval = ([=] TMOL_DEVICE_FUNC(int cta) {
        eval_energies_impl(
            cta,
            accumulate_derivs,
            std::integral_constant<PairMode, PairMode::InterAndIntra>{});
      });
      launch_energies(eval);
    }
  } else {
    auto eval = ([=] TMOL_DEVICE_FUNC(int cta) {
      eval_energies_impl(
          cta,
          std::false_type{},
          std::integral_constant<PairMode, PairMode::InterAndIntra>{});
    });
    launch_energies(eval);
  }
#endif

  // DeviceDispatch<Dev>::synchronize_device();
  return {output_t, dV_dcoords_t, scratch_rot_neighbors_t};
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    typename Real,
    typename Int>
auto HBondPoseScoreDispatch<DeviceDispatch, Dev, Real, Int>::backward(
    ContextManager& mgr,
    // common params
    TView<Vec<Real, 3>, 1, Dev> rot_coords,
    TView<Int, 1, Dev> rot_coord_offset,
    TView<Int, 1, Dev> pose_ind_for_atom,
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

    // InterBlockBondsep: [pose, block1, slot, (block2, min separation)]
    // and [pose, block1, slot, conn1, conn2]
    TView<Int, 4, Dev> pose_stack_near_blocks,
    TView<int8_t, 5, Dev> pose_stack_inter_block_bondsep,

    //////////////////////
    // Chemical properties
    // how many atoms for a given block
    // Dimsize n_block_types
    TView<Int, 1, Dev> block_type_n_atoms,  // ?? needed ?? I think so

    // how many inter-block chemical bonds are there
    // Dimsize: n_block_types
    TView<Int, 1, Dev> block_type_n_interblock_bonds,

    // what atoms form the inter-block chemical bonds
    // Dimsize: n_block_types x max_n_interblock_bonds
    TView<Int, 2, Dev> block_type_atoms_forming_chemical_bonds,

    TView<Int, 1, Dev> block_type_n_all_bonds,
    TView<Vec<Int, 3>, 2, Dev> block_type_all_bonds,
    TView<Vec<Int, 2>, 2, Dev> block_type_atom_all_bond_ranges,
    // How many chemical bonds separate all pairs of atoms
    // within each block type?
    // Dimsize: n_block_types x max_n_atoms x max_n_atoms
    TView<Int, 3, Dev> block_type_path_distance,

    TView<Int, 2, Dev> block_type_tile_n_donH,
    TView<Int, 2, Dev> block_type_tile_n_acc,
    TView<Int, 3, Dev> block_type_tile_donH_inds,
    TView<Int, 3, Dev> block_type_tile_acc_inds,
    TView<Int, 3, Dev> block_type_tile_donor_type,
    TView<Int, 3, Dev> block_type_tile_acceptor_type,
    TView<Int, 3, Dev> block_type_tile_hybridization,
    TView<Int, 2, Dev> block_type_atom_is_hydrogen,

    //////////////////////

    // HBond potential parameters
    TView<HBondPairParams<Real>, 2, Dev> pair_params,
    TView<HBondPolysFor<Dev>, 2, Dev> pair_polynomials,
    TView<HBondGlobalParams<Real>, 1, Dev> global_params,

    // Derived-atom coords + source-atom indices produced by the
    // hbond pre-pass kernel (GenerateHBondBases).
    TView<Vec<Real, 3>, 2, Dev> derived_coords,
    TView<Int, 2, Dev> derived_atom_inds,

    TView<Int, 3, Dev> scratch_rot_neighbors,  // from forward pass
    TView<Real, 4, Dev> dTdV                   // nterms x nposes x len x len
    ) -> TPack<Vec<Real, 3>, 1, Dev>  // TODO: add extra dimension for terms
{
  using tmol::score::common::accumulate;
  using Real3 = Vec<Real, 3>;

  int const n_poses = n_rots_for_pose.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);
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

  auto n_rots = rot_coord_offset.size(0);
  // auto rot_max_n_atoms = rot_coords.size(1);
  auto n_atoms = rot_coords.size(0);

  auto dV_dcoords_t = TPack<Vec<Real, 3>, 1, Dev>::zeros({n_atoms});  // TODO
  auto dV_dcoords = dV_dcoords_t.view;

  // Optimal launch box on v100 and a100 is nt=32, vt=1
  LAUNCH_BOX_32;
  // Define nt and reduce_t
  CTA_REAL_REDUCE_T_TYPEDEF;
  // The total number of unique block pairs (including self-pairs)
  int const max_n_upper_triangle_inds = score::common::checked_triangular_size(
      max_n_blocks, true, "hydrogen-bond derivative block-pair dispatch");

  // State the kernel bodies read, named explicitly rather than captured
  // implicitly by [=]; see the forward pass. The last context parameter is
  // the derivative rank: a backward pass accumulates into a rank-1 tensor
  // where a forward pass writes the rank-2 one.
  HBondCtx<Real, Int, Dev, 4, 3, 1> ctx;
  ctx.rot_coords = rot_coords;
  ctx.rot_coord_offset = rot_coord_offset;
  ctx.block_ind_for_rot = block_ind_for_rot;
  ctx.block_type_ind_for_rot = block_type_ind_for_rot;
  ctx.rot_offset_for_block = rot_offset_for_block;
  ctx.pose_stack_near_blocks = pose_stack_near_blocks;
  ctx.pose_stack_inter_block_bondsep = pose_stack_inter_block_bondsep;
  ctx.block_type_n_atoms = block_type_n_atoms;
  ctx.block_type_n_interblock_bonds = block_type_n_interblock_bonds;
  ctx.block_type_atoms_forming_chemical_bonds =
      block_type_atoms_forming_chemical_bonds;
  ctx.block_type_path_distance = block_type_path_distance;
  ctx.block_type_tile_n_donH = block_type_tile_n_donH;
  ctx.block_type_tile_n_acc = block_type_tile_n_acc;
  ctx.block_type_tile_donH_inds = block_type_tile_donH_inds;
  ctx.block_type_tile_acc_inds = block_type_tile_acc_inds;
  ctx.block_type_tile_donor_type = block_type_tile_donor_type;
  ctx.block_type_tile_acceptor_type = block_type_tile_acceptor_type;
  ctx.block_type_tile_hybridization = block_type_tile_hybridization;
  ctx.pair_params = pair_params;
  ctx.pair_polynomials = pair_polynomials;
  ctx.global_params = global_params;
  ctx.derived_coords = derived_coords;
  ctx.derived_atom_inds = derived_atom_inds;
  ctx.dispatch_indices = scratch_rot_neighbors;
  ctx.output = dTdV;
  ctx.dTdV = dTdV;
  ctx.dV_dcoords = dV_dcoords;
  ctx.max_n_blocks = max_n_blocks;
  ctx.max_n_upper_triangle_inds = max_n_upper_triangle_inds;
  ctx.use_shared_compact_block_neighbors = false;
  ctx.output_block_pair_energies = true;
  ctx.accumulate_derivs = true;
  // Empty on CPU and CUDA; on Metal it carries the thread index and the
  // reduction scratch the workgroup primitives need. See workgroup_lane.hh.
  WorkgroupLane<Dev> lane;

  auto eval_derivs = ([=] TMOL_DEVICE_FUNC(int cta) {
    typedef HBondSharedMem<Real, TILE_SIZE, MAX_N_CONN, launch_t>
        shared_mem_union;
    SHARED_MEMORY shared_mem_union shared;
    hbond_pose_eval_derivs<DeviceDispatch, Dev, Real, Int, launch_t>(
        ctx, shared, cta);
  });

  if (Dev == tmol::Device::MPS) {
    // As in the forward pass: Metal has nowhere for a closure to live, so the
    // identical body is reached through a named entry point instead.
    //
    // Derivative accumulation here is order-independent by construction --
    // accumulate_grad snaps each contribution to a 2^-13 grid before adding,
    // so the atomics commute. See accumulate.hh.
    hbond_mps_launch_if<Dev>(
        "hbond_pose_eval_derivs_f32",
        ctx,
        int64_t(n_poses) * int64_t(max_n_upper_triangle_inds));
  } else {
    DeviceDispatch<Dev>::template foreach_pose_workgroup<launch_t>(
        mgr, n_poses, max_n_upper_triangle_inds, eval_derivs);
  }

  return dV_dcoords_t;
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    typename Real,
    typename Int>
auto HBondRotamerScoreDispatch<DeviceDispatch, Dev, Real, Int>::rotamer_spheres(
    ContextManager& mgr,
    TView<Vec<Real, 3>, 1, Dev> rot_coords,
    TView<Int, 1, Dev> rot_coord_offset,
    TView<Int, 2, Dev> first_rot_block_type,
    TView<Int, 1, Dev> block_type_ind_for_rot,
    TView<Int, 2, Dev> n_rots_for_block,
    TView<Int, 2, Dev> rot_offset_for_block,
    TView<Int, 1, Dev> block_type_n_atoms)
    -> std::tuple<TPack<Real, 2, Dev>, TPack<Real, 3, Dev>> {
  auto rot_spheres = TPack<Real, 2, Dev>::empty({rot_coord_offset.size(0), 4});
  auto block_spheres = TPack<Real, 3, Dev>::empty(
      {first_rot_block_type.size(0), first_rot_block_type.size(1), 4});
  score::common::sphere_overlap::
      compute_rot_spheres<DeviceDispatch, Dev, Real, Int>::f(
          mgr,
          rot_coords,
          rot_coord_offset,
          block_type_ind_for_rot,
          block_type_n_atoms,
          rot_spheres.view);
  score::common::sphere_overlap::
      compute_block_spheres_from_rot_spheres<DeviceDispatch, Dev, Real, Int>::f(
          mgr,
          rot_spheres.view,
          n_rots_for_block,
          rot_offset_for_block,
          block_spheres.view);
  return {rot_spheres, block_spheres};
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    typename Real,
    typename Int>
auto HBondRotamerScoreDispatch<DeviceDispatch, Dev, Real, Int>::
    rotamer_dispatch_page(
        ContextManager& mgr,
        TView<Int, 2, Dev> first_rot_block_type,
        TView<Real, 3, Dev> block_spheres,
        TView<Int, 2, Dev> n_rots_for_block,
        TView<Int, 2, Dev> rot_offset_for_block,
        TView<Real, 2, Dev> rot_spheres,
        TView<Int, 2, Dev> lockstep_group_for_block,
        Real reach,
        Int candidate_begin,
        Int candidate_end) -> TPack<Int, 2, Dev> {
  return score::common::sphere_overlap::
      rot_neighbor_indices_from_block_neighbors<
          DeviceDispatch,
          Dev,
          Real,
          Int>::
          f(mgr,
            first_rot_block_type,
            block_spheres,
            n_rots_for_block,
            rot_offset_for_block,
            rot_spheres,
            lockstep_group_for_block,
            reach,
            int64_t(candidate_begin),
            // The shared window API is count-based; -1 means "to the end".
            candidate_end < 0
                ? int64_t(-1)
                : int64_t(candidate_end) - int64_t(candidate_begin));
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    typename Real,
    typename Int>
auto HBondRotamerScoreDispatch<DeviceDispatch, Dev, Real, Int>::forward(
    ContextManager& mgr,
    // common params
    TView<Vec<Real, 3>, 1, Dev> rot_coords,
    TView<Int, 1, Dev> rot_coord_offset,
    TView<Int, 1, Dev> pose_ind_for_atom,
    TView<Int, 2, Dev> first_rot_for_block,
    TView<Int, 2, Dev> first_rot_block_type,
    TView<Int, 1, Dev> block_ind_for_rot,
    TView<Int, 1, Dev> pose_ind_for_rot,
    TView<Int, 1, Dev> block_type_ind_for_rot,
    TView<Int, 1, Dev> n_rots_for_pose,
    TView<Int, 1, Dev> rot_offset_for_pose,
    TView<Int, 2, Dev> n_rots_for_block,
    TView<Int, 2, Dev> rot_offset_for_block,
    // [n_poses, max_n_blocks]; blocks sharing an id >= 0 move in
    // lockstep, so only matching rotamer indices ever coexist
    TView<Int, 2, Dev> lockstep_group_for_block,
    Int max_n_rots_per_pose,

    // For determining which atoms to retrieve from neighboring
    // residues we have to know how the blocks in the Pose
    // are connected
    TView<Vec<Int, 2>, 3, Dev> pose_stack_inter_residue_connections,

    // InterBlockBondsep: [pose, block1, slot, (block2, min separation)]
    // and [pose, block1, slot, conn1, conn2]
    TView<Int, 4, Dev> pose_stack_near_blocks,
    TView<int8_t, 5, Dev> pose_stack_inter_block_bondsep,

    //////////////////////
    // Chemical properties
    // how many atoms for a given block
    // Dimsize n_block_types
    TView<Int, 1, Dev> block_type_n_atoms,  // ?? needed ?? I think so

    // how many inter-block chemical bonds are there
    // Dimsize: n_block_types
    TView<Int, 1, Dev> block_type_n_interblock_bonds,

    // what atoms form the inter-block chemical bonds
    // Dimsize: n_block_types x max_n_interblock_bonds
    TView<Int, 2, Dev> block_type_atoms_forming_chemical_bonds,

    TView<Int, 1, Dev> block_type_n_all_bonds,
    TView<Vec<Int, 3>, 2, Dev> block_type_all_bonds,
    TView<Vec<Int, 2>, 2, Dev> block_type_atom_all_bond_ranges,
    // How many chemical bonds separate all pairs of atoms
    // within each block type?
    // Dimsize: n_block_types x max_n_atoms x max_n_atoms
    TView<Int, 3, Dev> block_type_path_distance,

    TView<Int, 2, Dev> block_type_tile_n_donH,
    TView<Int, 2, Dev> block_type_tile_n_acc,
    TView<Int, 3, Dev> block_type_tile_donH_inds,
    TView<Int, 3, Dev> block_type_tile_acc_inds,
    TView<Int, 3, Dev> block_type_tile_donor_type,
    TView<Int, 3, Dev> block_type_tile_acceptor_type,
    TView<Int, 3, Dev> block_type_tile_hybridization,
    TView<Int, 2, Dev> block_type_atom_is_hydrogen,

    // HBond potential parameters
    TView<HBondPairParams<Real>, 2, Dev> pair_params,
    TView<HBondPolysFor<Dev>, 2, Dev> pair_polynomials,
    TView<HBondGlobalParams<Real>, 1, Dev> global_params,

    // Derived-atom coords + source-atom indices produced by the
    // hbond pre-pass kernel (GenerateHBondBases).
    TView<Vec<Real, 3>, 2, Dev> derived_coords,
    TView<Int, 2, Dev> derived_atom_inds,

    bool output_block_pair_energies,
    bool compute_derivs,
    TPack<Int, 2, Dev> shared_dispatch_indices

    )
    -> std::tuple<
        TPack<Real, 2, Dev>,
        TPack<Vec<Real, 3>, 2, Dev>,
        TPack<Int, 2, Dev>> {
  using tmol::score::common::accumulate;
  using Real3 = Vec<Real, 3>;

  int const n_atoms = rot_coords.size(0);
  int const n_rots = rot_coord_offset.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);

  int const max_n_conn = pose_stack_inter_residue_connections.size(2);
  int const n_block_types = block_type_n_atoms.size(0);
  int const max_n_block_atoms = block_type_atom_is_hydrogen.size(1);
  int const max_n_interblock_bonds =
      block_type_atoms_forming_chemical_bonds.size(1);
  int const max_n_all_bonds = block_type_all_bonds.size(1);
  int const max_n_tiles = block_type_tile_donH_inds.size(1);
  int const max_n_donH_per_tile = block_type_tile_donH_inds.size(2);
  int const max_n_acc_per_tile = block_type_tile_acc_inds.size(2);

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

  assert(pose_stack_inter_residue_connections.size(0) == n_poses);
  assert(pose_stack_inter_residue_connections.size(1) == max_n_blocks);

  assert(pose_stack_near_blocks.size(0) == n_poses);
  assert(pose_stack_near_blocks.size(1) == max_n_blocks);
  assert(pose_stack_near_blocks.size(3) == 2);

  assert(pose_stack_inter_block_bondsep.size(0) == n_poses);
  assert(pose_stack_inter_block_bondsep.size(1) == max_n_blocks);
  assert(
      pose_stack_inter_block_bondsep.size(2) == pose_stack_near_blocks.size(2));
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
  assert(block_type_path_distance.size(2) == max_n_block_atoms);

  assert(max_n_interblock_bonds <= MAX_N_CONN);

  bool const accumulate_derivs = compute_derivs;
  auto dV_dcoords_t = accumulate_derivs
                          ? TPack<Vec<Real, 3>, 2, Dev>::zeros({1, n_atoms})
                          : TPack<Vec<Real, 3>, 2, Dev>::empty({1, 0});
  auto dV_dcoords = dV_dcoords_t.view;

  TPack<Int, 2, Dev> dispatch_indices_t;
  if (shared_dispatch_indices.size(0) == 3) {
    dispatch_indices_t = shared_dispatch_indices;
  } else {
    auto scratch_rot_spheres_t = Dev == Device::CPU
                                     ? TPack<Real, 2, Dev>::zeros({n_rots, 4})
                                     : TPack<Real, 2, Dev>::empty({n_rots, 4});
    auto scratch_rot_spheres = scratch_rot_spheres_t.view;

    auto scratch_block_spheres_t =
        Dev == Device::CPU
            ? TPack<Real, 3, Dev>::zeros({n_poses, max_n_blocks, 4})
            : TPack<Real, 3, Dev>::empty({n_poses, max_n_blocks, 4});
    auto scratch_block_spheres = scratch_block_spheres_t.view;

    score::common::sphere_overlap::
        compute_rot_spheres<DeviceDispatch, Dev, Real, Int>::f(
            mgr,
            rot_coords,
            rot_coord_offset,
            block_type_ind_for_rot,
            block_type_n_atoms,
            scratch_rot_spheres);

    score::common::sphere_overlap::
        compute_block_spheres_from_rot_spheres<DeviceDispatch, Dev, Real, Int>::
            f(mgr,
              scratch_rot_spheres,
              n_rots_for_block,
              rot_offset_for_block,
              scratch_block_spheres);

    dispatch_indices_t = score::common::sphere_overlap::
        rot_neighbor_indices_from_block_neighbors<
            DeviceDispatch,
            Dev,
            Real,
            Int>::
            f(mgr,
              first_rot_block_type,
              scratch_block_spheres,
              n_rots_for_block,
              rot_offset_for_block,
              scratch_rot_spheres,
              lockstep_group_for_block,
              Real(5.5));
  }

  auto dispatch_indices = dispatch_indices_t.view;

  TPack<Real, 2, Dev> output_t;
  if (output_block_pair_energies) {
    output_t = TPack<Real, 2, Dev>::zeros({1, dispatch_indices.size(1)});
  } else {
    output_t = TPack<Real, 2, Dev>::zeros({1, n_poses});
  }
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

  // State the kernel body reads, named explicitly rather than captured
  // implicitly by [=]. The rotamer family's output is [term, rot_pair] and its
  // dispatch list is [3, rot_pair]; the whole-pose family uses the same
  // context with ranks 4 and 3.
  HBondCtx<Real, Int, Dev, 2, 2, 2> ctx;
  ctx.rot_coords = rot_coords;
  ctx.rot_coord_offset = rot_coord_offset;
  ctx.block_ind_for_rot = block_ind_for_rot;
  ctx.block_type_ind_for_rot = block_type_ind_for_rot;
  ctx.rot_offset_for_block = rot_offset_for_block;
  ctx.pose_stack_near_blocks = pose_stack_near_blocks;
  ctx.pose_stack_inter_block_bondsep = pose_stack_inter_block_bondsep;
  ctx.block_type_n_atoms = block_type_n_atoms;
  ctx.block_type_n_interblock_bonds = block_type_n_interblock_bonds;
  ctx.block_type_atoms_forming_chemical_bonds =
      block_type_atoms_forming_chemical_bonds;
  ctx.block_type_path_distance = block_type_path_distance;
  ctx.block_type_tile_n_donH = block_type_tile_n_donH;
  ctx.block_type_tile_n_acc = block_type_tile_n_acc;
  ctx.block_type_tile_donH_inds = block_type_tile_donH_inds;
  ctx.block_type_tile_acc_inds = block_type_tile_acc_inds;
  ctx.block_type_tile_donor_type = block_type_tile_donor_type;
  ctx.block_type_tile_acceptor_type = block_type_tile_acceptor_type;
  ctx.block_type_tile_hybridization = block_type_tile_hybridization;
  ctx.pair_params = pair_params;
  ctx.pair_polynomials = pair_polynomials;
  ctx.global_params = global_params;
  ctx.derived_coords = derived_coords;
  ctx.derived_atom_inds = derived_atom_inds;
  ctx.dispatch_indices = dispatch_indices;
  ctx.output = output;
  ctx.dV_dcoords = dV_dcoords;
  ctx.max_n_blocks = max_n_blocks;
  ctx.output_block_pair_energies = output_block_pair_energies;
  ctx.accumulate_derivs = accumulate_derivs;
  // Read only by the backward kernel; the Metal launcher binds every view, so
  // it must not be left default-constructed.
  ctx.dTdV = output;

  auto eval_energies = ([=] TMOL_DEVICE_FUNC(int cta) {
    typedef HBondSharedMem<Real, TILE_SIZE, MAX_N_CONN, launch_t>
        shared_mem_union;
    SHARED_MEMORY shared_mem_union shared;
    if (accumulate_derivs) {
      hbond_rotamer_eval_energies<
          DeviceDispatch,
          Dev,
          Real,
          Int,
          launch_t,
          true>(ctx, shared, cta);
    } else {
      hbond_rotamer_eval_energies<
          DeviceDispatch,
          Dev,
          Real,
          Int,
          launch_t,
          false>(ctx, shared, cta);
    }
  });

  // Evaluate the prepared rotamer-pair dispatch list.

  if (Dev == tmol::Device::MPS) {
    // Metal cannot launch a device lambda -- there is nowhere for a closure to
    // live, and entry points are compiled by a separate toolchain -- so the
    // same kernel body is reached through a named entry instead. The context
    // the lambda closes over is the same object either way.
    //
    // Every pair writes its own cell of `output`, so unlike the whole-pose
    // kernel there are no atomics here and no reduction to make deterministic.
    // Packing never asks for derivatives, so it gets the lean entry; the
    // other exists so a gradient-carrying call stays correct.
    hbond_mps_launch_if<Dev>(
        accumulate_derivs ? "hbond_rotamer_eval_derivs_energies_f32"
                          : "hbond_rotamer_eval_energies_f32",
        ctx,
        dispatch_indices.size(1));
  } else if (compute_derivs) {
    DeviceDispatch<Dev>::template foreach_workgroup<launch_t>(
        mgr, dispatch_indices.size(1), eval_energies);
  } else {
    DeviceDispatch<Dev>::template foreach_independent_workgroup<launch_t>(
        mgr, dispatch_indices.size(1), eval_energies);
  }

  // DeviceDispatch<Dev>::synchronize_device();
  return {output_t, dV_dcoords_t, dispatch_indices_t};
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    typename Real,
    typename Int>
auto HBondRotamerScoreDispatch<DeviceDispatch, Dev, Real, Int>::backward(
    ContextManager& mgr,
    // common params
    TView<Vec<Real, 3>, 1, Dev> rot_coords,
    TView<Int, 1, Dev> rot_coord_offset,
    TView<Int, 1, Dev> pose_ind_for_atom,
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

    // InterBlockBondsep: [pose, block1, slot, (block2, min separation)]
    // and [pose, block1, slot, conn1, conn2]
    TView<Int, 4, Dev> pose_stack_near_blocks,
    TView<int8_t, 5, Dev> pose_stack_inter_block_bondsep,

    //////////////////////
    // Chemical properties
    // how many atoms for a given block
    // Dimsize n_block_types
    TView<Int, 1, Dev> block_type_n_atoms,  // ?? needed ?? I think so

    // how many inter-block chemical bonds are there
    // Dimsize: n_block_types
    TView<Int, 1, Dev> block_type_n_interblock_bonds,

    // what atoms form the inter-block chemical bonds
    // Dimsize: n_block_types x max_n_interblock_bonds
    TView<Int, 2, Dev> block_type_atoms_forming_chemical_bonds,

    TView<Int, 1, Dev> block_type_n_all_bonds,
    TView<Vec<Int, 3>, 2, Dev> block_type_all_bonds,
    TView<Vec<Int, 2>, 2, Dev> block_type_atom_all_bond_ranges,
    // How many chemical bonds separate all pairs of atoms
    // within each block type?
    // Dimsize: n_block_types x max_n_atoms x max_n_atoms
    TView<Int, 3, Dev> block_type_path_distance,

    TView<Int, 2, Dev> block_type_tile_n_donH,
    TView<Int, 2, Dev> block_type_tile_n_acc,
    TView<Int, 3, Dev> block_type_tile_donH_inds,
    TView<Int, 3, Dev> block_type_tile_acc_inds,
    TView<Int, 3, Dev> block_type_tile_donor_type,
    TView<Int, 3, Dev> block_type_tile_acceptor_type,
    TView<Int, 3, Dev> block_type_tile_hybridization,
    TView<Int, 2, Dev> block_type_atom_is_hydrogen,

    //////////////////////

    // HBond potential parameters
    TView<HBondPairParams<Real>, 2, Dev> pair_params,
    TView<HBondPolysFor<Dev>, 2, Dev> pair_polynomials,
    TView<HBondGlobalParams<Real>, 1, Dev> global_params,

    // Derived-atom coords + source-atom indices produced by the
    // hbond pre-pass kernel (GenerateHBondBases).
    TView<Vec<Real, 3>, 2, Dev> derived_coords,
    TView<Int, 2, Dev> derived_atom_inds,

    TView<Int, 2, Dev> dispatch_indices,  // from forward pass
    TView<Real, 2, Dev> dTdV              // nterms x nposes x len x len
    ) -> TPack<Vec<Real, 3>, 1, Dev>      // TODO: add extra dimension for terms
{
  using tmol::score::common::accumulate;
  using Real3 = Vec<Real, 3>;

  int const n_poses = n_rots_for_pose.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);
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

  auto n_rots = rot_coord_offset.size(0);
  // auto rot_max_n_atoms = rot_coords.size(1);
  auto n_atoms = rot_coords.size(0);

  auto dV_dcoords_t = TPack<Vec<Real, 3>, 1, Dev>::zeros({n_atoms});  // TODO
  auto dV_dcoords = dV_dcoords_t.view;

  // Optimal launch box on v100 and a100 is nt=32, vt=1
  LAUNCH_BOX_32;
  // Define nt and reduce_t
  CTA_REAL_REDUCE_T_TYPEDEF;

  // State the kernel bodies read, named explicitly rather than captured
  // implicitly by [=]; see the forward pass. The last context parameter is
  // the derivative rank: a backward pass accumulates into a rank-1 tensor
  // where a forward pass writes the rank-2 one.
  HBondCtx<Real, Int, Dev, 2, 2, 1> ctx;
  ctx.rot_coords = rot_coords;
  ctx.rot_coord_offset = rot_coord_offset;
  ctx.block_ind_for_rot = block_ind_for_rot;
  ctx.block_type_ind_for_rot = block_type_ind_for_rot;
  ctx.rot_offset_for_block = rot_offset_for_block;
  ctx.pose_stack_near_blocks = pose_stack_near_blocks;
  ctx.pose_stack_inter_block_bondsep = pose_stack_inter_block_bondsep;
  ctx.block_type_n_atoms = block_type_n_atoms;
  ctx.block_type_n_interblock_bonds = block_type_n_interblock_bonds;
  ctx.block_type_atoms_forming_chemical_bonds =
      block_type_atoms_forming_chemical_bonds;
  ctx.block_type_path_distance = block_type_path_distance;
  ctx.block_type_tile_n_donH = block_type_tile_n_donH;
  ctx.block_type_tile_n_acc = block_type_tile_n_acc;
  ctx.block_type_tile_donH_inds = block_type_tile_donH_inds;
  ctx.block_type_tile_acc_inds = block_type_tile_acc_inds;
  ctx.block_type_tile_donor_type = block_type_tile_donor_type;
  ctx.block_type_tile_acceptor_type = block_type_tile_acceptor_type;
  ctx.block_type_tile_hybridization = block_type_tile_hybridization;
  ctx.pair_params = pair_params;
  ctx.pair_polynomials = pair_polynomials;
  ctx.global_params = global_params;
  ctx.derived_coords = derived_coords;
  ctx.derived_atom_inds = derived_atom_inds;
  ctx.dispatch_indices = dispatch_indices;
  ctx.output = dTdV;
  ctx.dTdV = dTdV;
  ctx.dV_dcoords = dV_dcoords;
  ctx.max_n_blocks = max_n_blocks;
  // Unused by the rotamer-pair body -- it walks dispatch_indices rather
  // than the upper triangle -- but the Metal launcher copies every scalar
  // slot, so it must not be left uninitialized.
  ctx.max_n_upper_triangle_inds = 0;
  ctx.use_shared_compact_block_neighbors = false;
  ctx.output_block_pair_energies = true;
  ctx.accumulate_derivs = true;
  // Empty on CPU and CUDA; on Metal it carries the thread index and the
  // reduction scratch the workgroup primitives need. See workgroup_lane.hh.
  WorkgroupLane<Dev> lane;

  auto eval_derivs = ([=] TMOL_DEVICE_FUNC(int cta) {
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

    auto hbond_atom_energy =
        ([=] TMOL_DEVICE_FUNC(
             int don_ind,
             int acc_ind,
             int don_tile_ind,
             int acc_tile_ind,
             int don_start,
             int acc_start,
             TMOL_THREAD HBondSingleResData<Real> const& don_dat,
             TMOL_THREAD HBondSingleResData<Real> const& acc_dat,
             TMOL_THREAD HBondRotPairData<Real> const& respair_dat,
             int cp_separation) {
          if (cp_separation < 5) {
            return Real(0.0);
          }
          Real val = hbond_atom_derivs<TILE_SIZE, Dev, Real, Int>(
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
              dTdV[0][cta],
              derived_coords,
              derived_atom_inds,
              pair_params,
              pair_polynomials,
              dV_dcoords);
          return Real(0.0);
        });

    auto score_inter_hbond_atom_pair = ([=] SCORE_INTER_HBOND_ATOM_PAIR);

    auto score_intra_hbond_atom_pair = ([=] SCORE_INTRA_HBOND_ATOM_PAIR);

    typedef HBondSharedMem<Real, TILE_SIZE, MAX_N_CONN, launch_t>
        shared_mem_union;
    SHARED_MEMORY shared_mem_union shared;

    auto load_tile_invariant_interres_data =
        ([=] LOAD_TILE_INVARIANT_INTERRES_DATA);

    auto load_interres1_tile_data_to_shared =
        ([=] LOAD_INTERRES1_TILE_DATA_TO_SHARED);

    auto load_interres2_tile_data_to_shared =
        ([=] LOAD_INTERRES2_TILE_DATA_TO_SHARED);

    auto load_interres_data_from_shared = ([=] LOAD_INTERRES_DATA_FROM_SHARED);

    auto eval_interres_atom_pair_scores = ([=] EVAL_INTERRES_ATOM_PAIR_SCORES);

    auto store_calculated_energies =
        ([=] TMOL_DEVICE_FUNC(
             TMOL_THREAD HBondScoringData<Real>& score_dat,
             TMOL_SHARED_REF(shared_mem_union) shared) { ; });

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
        Dev,
        HBondScoringData<Real>,
        HBondScoringData<Real>,
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

  DeviceDispatch<Dev>::template foreach_workgroup<launch_t>(
      mgr, dispatch_indices.size(1), eval_derivs);

  return dV_dcoords_t;
}

}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol
