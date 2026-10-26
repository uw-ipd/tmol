#pragma once

// The state an LK-ball kernel body reads.
//
// These fields were previously captured implicitly by `[=]` on a device
// lambda. A lambda closure has nowhere to live in Metal, whose kernel entry
// points are `kernel void` functions compiled by a separate toolchain, so the
// body takes this struct explicitly instead and can be invoked from any
// backend.
//
// Templated on the view so the field list is declared exactly once: the host
// and CUDA builds instantiate it with TView, and a Metal translation unit --
// which cannot include the torch tensor layer at all -- instantiates it with
// tmol::portable::MetalView. See lk_ball_ctx_tensor.hh for the host alias.
//
// Only the forward energy kernel is covered. LK-ball computes derivatives in
// a separate `backward` dispatch rather than inside the forward pass, so
// unlike elec and ljlk there is one kernel family here, not two.

#include <tmol/score/lk_ball/potentials/params_types.hh>
#include <tmol/utility/portable/device.hh>
#include <tmol/utility/portable/matrix.hh>

namespace tmol {
namespace score {
namespace lk_ball {
namespace potentials {

using tmol::portable::Vec;

template <
    template <typename, int> class View,
    typename Real,
    typename Int,
    int OutRank>
struct LKBallCtxT {
  // Coordinates, the attached waters, and the block indexing that locates them
  View<Vec<Real, 3>, 1> rot_coords;
  View<Vec<Real, 3>, 2> water_coords;
  View<Int, 1> rot_coord_offset;
  View<Int, 2> rot_offset_for_block;
  View<Int, 1> block_type_ind_for_rot;
  View<Int, 1> block_ind_for_rot;

  // Inter-block proximity and bond separation
  View<Vec<Int, 2>, 3> pose_stack_inter_residue_connections;
  View<Int, 4> pose_stack_near_blocks;
  View<int8_t, 5> pose_stack_inter_block_bondsep;
  View<Int, 3> scratch_rot_neighbors;
  // Rotamer-pair dispatch list, [3, n_pairs]; bound only by the rotamer
  // entry point, which walks it instead of the block-pair upper triangle.
  View<Int, 2> dispatch_indices;

  // Chemical properties of each block type
  View<Int, 1> block_type_n_atoms;
  View<Int, 1> block_type_n_interblock_bonds;
  View<Int, 2> block_type_atoms_forming_chemical_bonds;
  View<Int, 3> block_type_path_distance;
  View<Int, 2> block_type_tile_n_polar_atoms;
  View<Int, 2> block_type_tile_n_occluder_atoms;
  View<Int, 3> block_type_tile_pol_occ_inds;
  View<LKBallTypeParams<Real>, 3> block_type_tile_lk_ball_params;

  View<LKBallGlobalParams<Real>, 1> global_params;

  // Output
  View<Real, OutRank> output;

  // Backward pass only: the upstream weights and the two derivative
  // accumulators. Left unbound by the forward entries, which never read them.
  View<Real, 4> dTdV;
  View<Vec<Real, 3>, 1> dV_d_pose_coords;
  View<Vec<Real, 3>, 2> dV_d_water_coords;

  // Enumerate the views in a fixed order, applying f to each.
  //
  // One ordering, two uses. The host walks it to bind each view's buffer and
  // to record its shape; the Metal entry walks the same order to rebuild the
  // views. See the comment in elec_ctx.hh for why the list lives in exactly
  // one place.
  template <typename F>
  TMOL_HD void visit_views(F f) {
    f(rot_coords);
    f(water_coords);
    f(rot_coord_offset);
    f(rot_offset_for_block);
    f(block_type_ind_for_rot);
    f(block_ind_for_rot);
    f(pose_stack_inter_residue_connections);
    f(pose_stack_near_blocks);
    f(pose_stack_inter_block_bondsep);
    f(scratch_rot_neighbors);
    f(dispatch_indices);
    f(block_type_n_atoms);
    f(block_type_n_interblock_bonds);
    f(block_type_atoms_forming_chemical_bonds);
    f(block_type_path_distance);
    f(block_type_tile_n_polar_atoms);
    f(block_type_tile_n_occluder_atoms);
    f(block_type_tile_pol_occ_inds);
    f(block_type_tile_lk_ball_params);
    f(global_params);
    f(output);
    f(dTdV);
    f(dV_d_pose_coords);
    f(dV_d_water_coords);
  }

  enum { n_views = 24 };

  // Launch scalars in the order the host writes and the shader reads.
  template <typename F>
  TMOL_HD void visit_scalars(F f) {
    f(max_n_blocks);
    f(n_block_pairs);
    f(output_block_pair_energies);
    f(use_shared_compact_block_neighbors);
    f(block_pair_scoring);
  }

  // Launch-shape constants
  int max_n_blocks = {};
  int n_block_pairs = {};
  bool output_block_pair_energies = {};
  bool use_shared_compact_block_neighbors = {};
  // Backward pass only: whether dTdV is indexed per block pair or per pose.
  bool block_pair_scoring = {};
};

}  // namespace potentials
}  // namespace lk_ball
}  // namespace score
}  // namespace tmol
