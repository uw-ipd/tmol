#pragma once
#include <tmol/score/elec/potentials/elec_mps_launch.hh>
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
#include <tmol/score/elec/potentials/elec.hh>
#include <tmol/score/elec/potentials/params.hh>
#include <tmol/score/elec/potentials/elec_ctx_tensor.hh>
#include <tmol/score/elec/potentials/elec_kernels.hh>
#include <tmol/score/elec/potentials/elec_pose_score.hh>
// Operator definitions; safe for CPU compilation
#include <chrono>
// The maximum number of inter-residue chemical bonds
#define MAX_N_CONN 12
#define TILE_SIZE 32
namespace tmol {
namespace score {
namespace elec {
namespace potentials {
using tmol::portable::Vec;
// MACROS
template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto ElecPoseScoreDispatch<DeviceDispatch, D, Real, Int>::forward(
    ContextManager& mgr,
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
    // InterBlockBondsep: [pose, block1, slot, (block2, min separation)]
    // and [pose, block1, slot, conn1, conn2]
    TView<Int, 4, D> pose_stack_near_blocks,
    TView<int8_t, 5, D> pose_stack_inter_block_bondsep,
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
    TView<Int, 1, D> block_type_all_atoms_ligand_typed,
    //////////////////////
    // LJ parameters
    TView<ElecGlobalParams<Real>, 1, D> global_params,
    Real max_dis,
    TView<Int, 1, D> shared_compact_block_neighbors,
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
  assert(block_type_inter_repr_path_distance.size(0) == n_block_types);
  assert(block_type_inter_repr_path_distance.size(1) == max_n_block_atoms);
  assert(block_type_inter_repr_path_distance.size(2) == max_n_block_atoms);
  assert(block_type_intra_repr_path_distance.size(0) == n_block_types);
  assert(block_type_intra_repr_path_distance.size(1) == max_n_block_atoms);
  assert(block_type_intra_repr_path_distance.size(2) == max_n_block_atoms);
  // Block-pair autograd recomputes coordinate derivatives in backward.
  bool const accumulate_derivs = compute_derivs && !output_block_pair_energies;
  auto dV_dcoords_t = accumulate_derivs
                          ? TPack<Vec<Real, 3>, 2, D>::zeros({1, n_atoms})
                          : TPack<Vec<Real, 3>, 2, D>::empty({1, 0});
  auto dV_dcoords = dV_dcoords_t.view;
  bool const use_shared_compact_block_neighbors =
      shared_compact_block_neighbors.size(0) != 0;
  TPack<Real, 3, D> scratch_rot_spheres_t;
  TPack<Int, 3, D> scratch_rot_neighbors_t;
  TView<Real, 3, D> scratch_rot_spheres;
  TView<Int, 3, D> scratch_rot_neighbors;
  if (!use_shared_compact_block_neighbors) {
    scratch_rot_spheres_t =
        D == Device::CPU ? TPack<Real, 3, D>::zeros({n_poses, max_n_blocks, 4})
                         : TPack<Real, 3, D>::empty({n_poses, max_n_blocks, 4});
    scratch_rot_neighbors_t =
        D == Device::CPU
            ? TPack<Int, 3, D>::zeros({n_poses, max_n_blocks, max_n_blocks})
            : TPack<Int, 3, D>::empty({n_poses, max_n_blocks, max_n_blocks});
    scratch_rot_spheres = scratch_rot_spheres_t.view;
    scratch_rot_neighbors = scratch_rot_neighbors_t.view;
    score::common::sphere_overlap::
        compute_block_spheres<DeviceDispatch, D, Real, Int>::f(
            mgr,
            rot_coords,
            rot_coord_offset,
            block_ind_for_rot,
            pose_ind_for_rot,
            block_type_ind_for_rot,
            block_type_n_atoms,
            scratch_rot_spheres);
    score::common::sphere_overlap::
        detect_block_neighbors<DeviceDispatch, D, Real, Int>::f(
            mgr,
            first_rot_block_type,
            scratch_rot_spheres,
            scratch_rot_neighbors,
            max_dis);
  }
  // Metal accumulates whole-pose totals into per-block-pair cells and reduces
  // them afterwards, rather than atomically adding into one cell.
  //
  // The threadgroup reduction is already the CUDA shape -- simd_sum across the
  // simdgroup, then one atomic per threadgroup -- but one atomic per
  // threadgroup is still thousands of them per pose arriving in an order the
  // GPU is free to vary, and float addition is not associative. The resulting
  // jitter is about one float32 ULP of the pose total: small, but not zero,
  // and a line search that validates its analytic derivative against a finite
  // difference cannot tolerate a non-reproducible energy. At a trial step of
  // 1e-13 the true energy change is far below the jitter.
  //
  // Writing each threadgroup's partial to its own cell removes the atomic
  // entirely, and summing those cells is a fixed-order reduction. The cells
  // are the ones block-pair scoring already writes, so this reuses the store
  // that is there rather than adding a second output path. The cost is the
  // cell buffer -- one float per block pair per pose -- and one reduction.
  // The CPU path keeps its double accumulator and the CUDA path its atomics;
  // neither changes.
  bool const mps_cell_reduction =
      D == tmol::Device::MPS && !output_block_pair_energies;
  TPack<Real, 4, D> output_t;
  TPack<Real, 4, D> cells_t;
  if (output_block_pair_energies) {
    output_t =
        TPack<Real, 4, D>::zeros({1, n_poses, max_n_blocks, max_n_blocks});
  } else {
    output_t = TPack<Real, 4, D>::zeros({1, n_poses, 1, 1});
    if (mps_cell_reduction) {
      cells_t =
          TPack<Real, 4, D>::zeros({1, n_poses, max_n_blocks, max_n_blocks});
    }
  }
  auto output = mps_cell_reduction ? cells_t.view : output_t.view;
  // This atom-pair kernel benefits from a 64-register budget on modern GPUs,
  // which leaves enough resident warps to hide its dependency latency.
  LAUNCH_BOX_32_OCC(32);
  // Define nt and reduce_t
  // The total number of unique block pairs (including self-pairs)
  int const max_n_upper_triangle_inds = score::common::checked_triangular_size(
      max_n_blocks, true, "electrostatics block-pair dispatch");
  // We define two device lambdas, one for block-pair scoring and
  // one for full pose scoring. They are nearly identical, except
  // that when we are in block-pair scoring mode, whether or not
  // the input rot_coords tensor has save_grads as true, we do not
  // need to calculate the derivaties in the forward pass. In full
  // pose scoring mode, if save_grads is true, then we'll compute
  // the atom derivatives in the forward pass and we save them for
  // rapid retrieval in the backward pass (weighting them by the
  // weight applied to the elec energy.)
  // Device lambda for block-pair scoring: no derivative evaluation
  // State the kernel bodies read, named explicitly rather than
  // captured implicitly by [=].
  ElecCtx<Real, Int, D> ctx;
  ctx.rot_coords = rot_coords;
  ctx.rot_coord_offset = rot_coord_offset;
  ctx.block_type_ind_for_rot = block_type_ind_for_rot;
  ctx.block_ind_for_rot = block_ind_for_rot;
  ctx.rot_offset_for_block = rot_offset_for_block;
  ctx.pose_stack_near_blocks = pose_stack_near_blocks;
  ctx.pose_stack_inter_block_bondsep = pose_stack_inter_block_bondsep;
  ctx.scratch_rot_neighbors = scratch_rot_neighbors;
  ctx.block_type_n_atoms = block_type_n_atoms;
  ctx.block_type_partial_charge = block_type_partial_charge;
  ctx.block_type_n_interblock_bonds = block_type_n_interblock_bonds;
  ctx.block_type_atoms_forming_chemical_bonds =
      block_type_atoms_forming_chemical_bonds;
  ctx.block_type_inter_repr_path_distance = block_type_inter_repr_path_distance;
  ctx.block_type_intra_repr_path_distance = block_type_intra_repr_path_distance;
  ctx.block_type_all_atoms_ligand_typed = block_type_all_atoms_ligand_typed;
  ctx.global_params = global_params;
  ctx.output = output;
  ctx.dV_dcoords = dV_dcoords;
  ctx.max_n_blocks = max_n_blocks;
  ctx.max_n_upper_triangle_inds = max_n_upper_triangle_inds;
  // The store writes cells whenever the output view is cell-shaped: what the
  // caller asked for in block-pair mode, and what the Metal whole-pose path
  // arranges for itself above.
  ctx.output_block_pair_energies =
      output_block_pair_energies || mps_cell_reduction;
  ctx.use_shared_compact_block_neighbors = use_shared_compact_block_neighbors;
  auto eval_energies_by_block = ([=] TMOL_DEVICE_FUNC(int cta) {
    typedef ElecSharedMem<Real, TILE_SIZE, MAX_N_CONN, launch_t>
        shared_mem_union;
    SHARED_MEMORY shared_mem_union shared;
    elec_eval_energies_by_block_2<DeviceDispatch, D, Real, Int, launch_t>(
        ctx, shared, cta);
  });
  auto eval_energies = ([=] TMOL_DEVICE_FUNC(int cta) {
    typedef ElecSharedMem<Real, TILE_SIZE, MAX_N_CONN, launch_t>
        shared_mem_union;
    SHARED_MEMORY shared_mem_union shared;
    elec_eval_energies<DeviceDispatch, D, Real, Int, launch_t>(
        ctx, shared, cta);
  });
  // Score neighboring block pairs. Sparse large CUDA poses use a
  // device-resident compact list to avoid launching dead scoring CTAs.
#ifdef __NVCC__
  if (!output_block_pair_energies && use_shared_compact_block_neighbors) {
    if (compute_derivs) {
      score::common::sphere_overlap::
          launch_precomputed_block_neighbors<DeviceDispatch, D, launch_t, Int>(
              mgr,
              shared_compact_block_neighbors,
              n_poses,
              max_n_blocks,
              eval_energies);
    } else {
      score::common::sphere_overlap::
          launch_precomputed_block_neighbors<DeviceDispatch, D, launch_t, Int>(
              mgr,
              shared_compact_block_neighbors,
              n_poses,
              max_n_blocks,
              eval_energies_by_block);
    }
  } else if (
      !output_block_pair_energies
      && score::common::sphere_overlap::should_compact_block_neighbors(
          n_poses, max_n_blocks, compute_derivs)) {
    if (compute_derivs) {
      score::common::sphere_overlap::
          launch_compact_block_neighbors<DeviceDispatch, D, launch_t, Int>(
              mgr, scratch_rot_neighbors, eval_energies);
    } else {
      score::common::sphere_overlap::
          launch_compact_block_neighbors<DeviceDispatch, D, launch_t, Int>(
              mgr, scratch_rot_neighbors, eval_energies_by_block);
    }
  } else
#endif
  if (D == tmol::Device::MPS) {
    // Metal cannot launch a device lambda -- there is nowhere for a closure
    // to live, and entry points are compiled by a separate toolchain -- so
    // the same kernel body is reached through a named entry rather than
    // through eval_energies_by_block / eval_energies. The context those
    // lambdas close over is the same object either way.
    //
    // This is tested ahead of the compact-block-neighbor paths because those
    // launch through DeviceDispatch, which on this backend would run the
    // kernel on the CPU.
    // Entry selection depends on whether derivatives are wanted, not on the
    // output shape: the cell store is independent of the derivative
    // accumulation, so a whole-pose call with gradients gets both.
    elec_mps_launch_if<D>(
        compute_derivs && !output_block_pair_energies
            ? "elec_eval_energies_f32"
            : "elec_eval_energies_by_block_2_f32",
        ctx,
        int64_t(n_poses) * int64_t(max_n_upper_triangle_inds));
    if (mps_cell_reduction) {
      // Fixed-order reduction over the cells. torch runs the same kernel with
      // the same traversal for a given shape on every call, so the result is
      // bit-reproducible.
      // Compensated, fixed-order reduction rather than torch's tree sum.
      // A tree sum is reproducible but still lets the total move by several
      // ULP when a coordinate moves by a fraction of one, because its error
      // term moves with its inputs; see pose_total_reduce_kernels.hh.
      score::common::pose_total_reduce<DeviceDispatch, D, Real>::f(
          mgr, cells_t.view, output_t.view);
    }
  } else {
    if (!output_block_pair_energies && use_shared_compact_block_neighbors) {
      if (compute_derivs) {
        score::common::sphere_overlap::launch_precomputed_block_neighbors<
            DeviceDispatch,
            D,
            launch_t,
            Int>(
            mgr,
            shared_compact_block_neighbors,
            n_poses,
            max_n_blocks,
            eval_energies);
      } else {
        score::common::sphere_overlap::launch_precomputed_block_neighbors<
            DeviceDispatch,
            D,
            launch_t,
            Int>(
            mgr,
            shared_compact_block_neighbors,
            n_poses,
            max_n_blocks,
            eval_energies_by_block);
      }
    } else if (output_block_pair_energies || !compute_derivs) {
      DeviceDispatch<D>::template foreach_pose_workgroup<launch_t>(
          mgr, n_poses, max_n_upper_triangle_inds, eval_energies_by_block);
    } else {
      DeviceDispatch<D>::template foreach_pose_workgroup<launch_t>(
          mgr, n_poses, max_n_upper_triangle_inds, eval_energies);
    }
  }
  return {output_t, dV_dcoords_t, scratch_rot_neighbors_t};
}  // namespace potentials
template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto ElecPoseScoreDispatch<DeviceDispatch, D, Real, Int>::backward(
    ContextManager& mgr,
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
    // InterBlockBondsep: [pose, block1, slot, (block2, min separation)]
    // and [pose, block1, slot, conn1, conn2]
    TView<Int, 4, D> pose_stack_near_blocks,
    TView<int8_t, 5, D> pose_stack_inter_block_bondsep,
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
    TView<Int, 1, D> block_type_all_atoms_ligand_typed,
    //////////////////////
    // LJ parameters
    TView<ElecGlobalParams<Real>, 1, D> global_params,
    TView<Int, 3, D> scratch_rot_neighbors,
    TView<Real, 4, D> dTdV  // nterms x nposes x len x len
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
  // The total number of unique block pairs (including self-pairs)
  int const max_n_upper_triangle_inds = score::common::checked_triangular_size(
      max_n_blocks, true, "electrostatics derivative block-pair dispatch");
  // State the kernel bodies read, named explicitly rather than
  // captured implicitly by [=].
  ElecCtx<Real, Int, D> ctx;
  ctx.rot_coords = rot_coords;
  ctx.rot_coord_offset = rot_coord_offset;
  ctx.block_type_ind_for_rot = block_type_ind_for_rot;
  ctx.block_ind_for_rot = block_ind_for_rot;
  ctx.rot_offset_for_block = rot_offset_for_block;
  ctx.pose_stack_near_blocks = pose_stack_near_blocks;
  ctx.pose_stack_inter_block_bondsep = pose_stack_inter_block_bondsep;
  ctx.scratch_rot_neighbors = scratch_rot_neighbors;
  ctx.block_type_n_atoms = block_type_n_atoms;
  ctx.block_type_partial_charge = block_type_partial_charge;
  ctx.block_type_n_interblock_bonds = block_type_n_interblock_bonds;
  ctx.block_type_atoms_forming_chemical_bonds =
      block_type_atoms_forming_chemical_bonds;
  ctx.block_type_inter_repr_path_distance = block_type_inter_repr_path_distance;
  ctx.block_type_intra_repr_path_distance = block_type_intra_repr_path_distance;
  ctx.block_type_all_atoms_ligand_typed = block_type_all_atoms_ligand_typed;
  ctx.global_params = global_params;
  ctx.dTdV = dTdV;
  ctx.dV_dcoords = dV_dcoords;
  ctx.max_n_blocks = max_n_blocks;
  ctx.max_n_upper_triangle_inds = max_n_upper_triangle_inds;
  auto eval_derivs = ([=] TMOL_DEVICE_FUNC(int cta) {
    typedef ElecSharedMem<Real, TILE_SIZE, MAX_N_CONN, launch_t>
        shared_mem_union;
    SHARED_MEMORY shared_mem_union shared;
    elec_eval_derivs_2<DeviceDispatch, D, Real, Int, launch_t>(
        ctx, shared, cta);
  });
  // Since we have the sphere overlap results from the forward pass,
  // there's only a single kernel launch here
  DeviceDispatch<D>::template foreach_pose_workgroup<launch_t>(
      mgr, n_poses, max_n_upper_triangle_inds, eval_derivs);
  return dV_dcoords_t;
}
template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto ElecRotamerScoreDispatch<DeviceDispatch, D, Real, Int>::forward(
    ContextManager& mgr,
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
    // [n_poses, max_n_blocks]; blocks sharing an id >= 0 move in
    // lockstep, so only matching rotamer indices ever coexist
    TView<Int, 2, D> lockstep_group_for_block,
    Int max_n_rots_per_pose,
    // InterBlockBondsep: [pose, block1, slot, (block2, min separation)]
    // and [pose, block1, slot, conn1, conn2]
    TView<Int, 4, D> pose_stack_near_blocks,
    TView<int8_t, 5, D> pose_stack_inter_block_bondsep,
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
    TView<Int, 1, D> block_type_all_atoms_ligand_typed,
    //////////////////////
    // LJ parameters
    TView<ElecGlobalParams<Real>, 1, D> global_params,
    Real max_dis,
    bool output_block_pair_energies,
    bool compute_derivs,
    TPack<Int, 2, D> shared_dispatch_indices) -> std::
    tuple<TPack<Real, 2, D>, TPack<Vec<Real, 3>, 2, D>, TPack<Int, 2, D> > {
  using tmol::score::common::accumulate;
  using Real3 = Vec<Real, 3>;
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
  assert(block_type_inter_repr_path_distance.size(0) == n_block_types);
  assert(block_type_inter_repr_path_distance.size(1) == max_n_block_atoms);
  assert(block_type_inter_repr_path_distance.size(2) == max_n_block_atoms);
  assert(block_type_intra_repr_path_distance.size(0) == n_block_types);
  assert(block_type_intra_repr_path_distance.size(1) == max_n_block_atoms);
  assert(block_type_intra_repr_path_distance.size(2) == max_n_block_atoms);
  auto dV_dcoords_t = TPack<Vec<Real, 3>, 2, D>::empty({1, 0});
  TPack<Int, 2, D> dispatch_indices_t;
  if (shared_dispatch_indices.size(0) == 3) {
    dispatch_indices_t = shared_dispatch_indices;
  } else {
    auto scratch_rot_spheres_t = D == Device::CPU
                                     ? TPack<Real, 2, D>::zeros({n_rots, 4})
                                     : TPack<Real, 2, D>::empty({n_rots, 4});
    auto scratch_rot_spheres = scratch_rot_spheres_t.view;
    auto scratch_block_spheres_t =
        D == Device::CPU ? TPack<Real, 3, D>::zeros({n_poses, max_n_blocks, 4})
                         : TPack<Real, 3, D>::empty({n_poses, max_n_blocks, 4});
    auto scratch_block_spheres = scratch_block_spheres_t.view;
    score::common::sphere_overlap::
        compute_rot_spheres<DeviceDispatch, D, Real, Int>::f(
            mgr,
            rot_coords,
            rot_coord_offset,
            block_type_ind_for_rot,
            block_type_n_atoms,
            scratch_rot_spheres);
    score::common::sphere_overlap::
        compute_block_spheres_from_rot_spheres<DeviceDispatch, D, Real, Int>::f(
            mgr,
            scratch_rot_spheres,
            n_rots_for_block,
            rot_offset_for_block,
            scratch_block_spheres);
    dispatch_indices_t = score::common::sphere_overlap::
        rot_neighbor_indices_from_block_neighbors<
            DeviceDispatch,
            D,
            Real,
            Int>::
            f(mgr,
              first_rot_block_type,
              scratch_block_spheres,
              n_rots_for_block,
              rot_offset_for_block,
              scratch_rot_spheres,
              lockstep_group_for_block,
              max_dis);
  }
  auto dispatch_indices = dispatch_indices_t.view;
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
  // State the kernel bodies read, named explicitly rather than
  // captured implicitly by [=].
  ElecCtx<Real, Int, D, 2> ctx;
  ctx.rot_coords = rot_coords;
  ctx.rot_coord_offset = rot_coord_offset;
  ctx.block_type_ind_for_rot = block_type_ind_for_rot;
  ctx.block_ind_for_rot = block_ind_for_rot;
  ctx.rot_offset_for_block = rot_offset_for_block;
  ctx.pose_stack_near_blocks = pose_stack_near_blocks;
  ctx.pose_stack_inter_block_bondsep = pose_stack_inter_block_bondsep;
  ctx.block_type_n_atoms = block_type_n_atoms;
  ctx.block_type_partial_charge = block_type_partial_charge;
  ctx.block_type_n_interblock_bonds = block_type_n_interblock_bonds;
  ctx.block_type_atoms_forming_chemical_bonds =
      block_type_atoms_forming_chemical_bonds;
  ctx.block_type_inter_repr_path_distance = block_type_inter_repr_path_distance;
  ctx.block_type_intra_repr_path_distance = block_type_intra_repr_path_distance;
  ctx.block_type_all_atoms_ligand_typed = block_type_all_atoms_ligand_typed;
  ctx.global_params = global_params;
  ctx.dispatch_indices = dispatch_indices;
  ctx.output = output;
  ctx.max_n_blocks = max_n_blocks;
  ctx.output_block_pair_energies = output_block_pair_energies;
  // Unused by the rotamer-pair body -- it walks dispatch_indices rather than
  // the upper triangle -- but the Metal launcher copies every scalar slot, so
  // they must not be left uninitialized.
  ctx.max_n_upper_triangle_inds = 0;
  ctx.use_shared_compact_block_neighbors = false;
  auto eval_energies_by_block = ([=] TMOL_DEVICE_FUNC(int cta) {
    typedef ElecSharedMem<Real, TILE_SIZE, MAX_N_CONN, launch_t>
        shared_mem_union;
    SHARED_MEMORY shared_mem_union shared;
    elec_eval_energies_by_block<DeviceDispatch, D, Real, Int, launch_t>(
        ctx, shared, cta);
  });
  // Each pair writes its own score; backward recomputes its derivatives.
  if (D == tmol::Device::MPS) {
    // As in the whole-pose path: Metal has nowhere for a closure to live, so
    // the identical body is reached through a named entry point instead.
    // Every pair writes its own cell of `output`, so unlike the whole-pose
    // kernel there are no atomics here and no reduction to make
    // deterministic.
    elec_mps_launch_if<D>(
        "elec_eval_energies_by_block_f32", ctx, dispatch_indices.size(1));
  } else {
    DeviceDispatch<D>::template foreach_independent_workgroup<launch_t>(
        mgr, dispatch_indices.size(1), eval_energies_by_block);
  }
  return {output_t, dV_dcoords_t, dispatch_indices_t};
}  // namespace potentials
template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto ElecRotamerScoreDispatch<DeviceDispatch, D, Real, Int>::backward(
    ContextManager& mgr,
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
    // InterBlockBondsep: [pose, block1, slot, (block2, min separation)]
    // and [pose, block1, slot, conn1, conn2]
    TView<Int, 4, D> pose_stack_near_blocks,
    TView<int8_t, 5, D> pose_stack_inter_block_bondsep,
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
    TView<Int, 1, D> block_type_all_atoms_ligand_typed,
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
  // State the kernel bodies read, named explicitly rather than
  // captured implicitly by [=].
  ElecCtx<Real, Int, D, 2> ctx;
  ctx.rot_coords = rot_coords;
  ctx.rot_coord_offset = rot_coord_offset;
  ctx.block_type_ind_for_rot = block_type_ind_for_rot;
  ctx.block_ind_for_rot = block_ind_for_rot;
  ctx.rot_offset_for_block = rot_offset_for_block;
  ctx.pose_stack_near_blocks = pose_stack_near_blocks;
  ctx.pose_stack_inter_block_bondsep = pose_stack_inter_block_bondsep;
  ctx.block_type_n_atoms = block_type_n_atoms;
  ctx.block_type_partial_charge = block_type_partial_charge;
  ctx.block_type_n_interblock_bonds = block_type_n_interblock_bonds;
  ctx.block_type_atoms_forming_chemical_bonds =
      block_type_atoms_forming_chemical_bonds;
  ctx.block_type_inter_repr_path_distance = block_type_inter_repr_path_distance;
  ctx.block_type_intra_repr_path_distance = block_type_intra_repr_path_distance;
  ctx.block_type_all_atoms_ligand_typed = block_type_all_atoms_ligand_typed;
  ctx.global_params = global_params;
  ctx.dispatch_indices = dispatch_indices;
  ctx.dTdV = dTdV;
  ctx.dV_dcoords = dV_dcoords;
  ctx.max_n_blocks = max_n_blocks;
  auto eval_derivs = ([=] TMOL_DEVICE_FUNC(int cta) {
    typedef ElecSharedMem<Real, TILE_SIZE, MAX_N_CONN, launch_t>
        shared_mem_union;
    SHARED_MEMORY shared_mem_union shared;
    elec_eval_derivs<DeviceDispatch, D, Real, Int, launch_t>(ctx, shared, cta);
  });
  // Since we have the sphere overlap results from the forward pass,
  // there's only a single kernel launch here
  DeviceDispatch<D>::template foreach_workgroup<launch_t>(
      mgr, dispatch_indices.size(1), eval_derivs);
  return dV_dcoords_t;
}
}  // namespace potentials
}  // namespace elec
}  // namespace score
}  // namespace tmol