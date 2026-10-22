#pragma once

// The state an LJ/LK kernel body reads.
//
// These fields were previously captured implicitly by `[=]` on a device
// lambda. A lambda closure has nowhere to live in Metal, whose kernel entry
// points are `kernel void` functions compiled by a separate toolchain, so the
// bodies take this struct explicitly instead and can be invoked from any
// backend.
//
// Templated on the view so the field list is declared exactly once: the host
// and CUDA builds instantiate it with TView, and a Metal translation unit --
// which cannot include the torch tensor layer at all -- instantiates it with
// tmol::portable::MetalView. See ljlk_ctx_tensor.hh for the host alias.
//
// OutRank parameterizes the energy tensor, which differs between the two
// kernel families: whole-pose scoring writes [term, pose, block, block] while
// rotamer scoring writes [term, block_pair].

#include <tmol/score/ljlk/potentials/params_types.hh>
#include <tmol/utility/portable/device.hh>
#include <tmol/utility/portable/matrix.hh>

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

using tmol::portable::Vec;

template <
    template <typename, int> class View,
    typename Real,
    typename Int,
    int OutRank>
struct LJLKCtxT {
  // Coordinates and the rotamer/block indexing that locates them
  View<Vec<Real, 3>, 1> rot_coords;
  View<Int, 1> rot_coord_offset;
  View<Int, 1> block_ind_for_rot;
  View<Int, 1> pose_ind_for_rot;
  View<Int, 1> block_type_ind_for_rot;
  View<Int, 2> first_rot_block_type;
  View<Int, 2> rot_offset_for_block;

  // Inter-block proximity and bond separation
  View<Int, 4> pose_stack_near_blocks;
  View<int8_t, 5> pose_stack_inter_block_bondsep;
  View<Int, 3> scratch_rot_neighbors;
  // Rotamer-pair dispatch list, [3, n_pairs]; bound only by the rotamer
  // entry points, which walk it instead of the block-pair upper triangle.
  View<Int, 2> dispatch_indices;
  View<Int, 1> shared_compact_block_neighbors;

  // Chemical properties of each block type
  View<Int, 1> block_type_n_atoms;
  View<Int, 2> block_type_atom_types;
  View<Int, 1> block_type_n_interblock_bonds;
  View<Int, 2> block_type_atoms_forming_chemical_bonds;
  View<Int, 3> block_type_path_distance;
  View<Int, 1> block_type_all_atoms_ligand_typed;

  View<LJLKTypeParams<Real>, 1> type_params;
  View<LJGlobalParams<Real>, 1> global_params;

  // Outputs
  View<Real, OutRank> output;
  View<Real, OutRank> dTdV;
  View<Vec<Real, 3>, 2> dV_dcoords;

  // Launch-shape constants and the interaction cutoff
  int max_n_blocks;
  int max_n_upper_triangle_inds;
  bool output_block_pair_energies;
  bool use_shared_compact_block_neighbors;
  bool require_gradient;
  Real max_dis;

  // Enumerate the views in a fixed order, applying f to each.
  //
  // One ordering, two uses. The host walks it to bind each view's buffer and
  // to record its shape; the Metal entry walks the same order to rebuild the
  // views. See the comment in elec_ctx.hh for why the list lives in exactly
  // one place.
  template <typename F>
  TMOL_HD void visit_views(F f) {
    f(rot_coords);
    f(rot_coord_offset);
    f(block_ind_for_rot);
    f(pose_ind_for_rot);
    f(block_type_ind_for_rot);
    f(first_rot_block_type);
    f(rot_offset_for_block);
    f(pose_stack_near_blocks);
    f(pose_stack_inter_block_bondsep);
    f(scratch_rot_neighbors);
    f(dispatch_indices);
    f(shared_compact_block_neighbors);
    f(block_type_n_atoms);
    f(block_type_atom_types);
    f(block_type_n_interblock_bonds);
    f(block_type_atoms_forming_chemical_bonds);
    f(block_type_path_distance);
    f(block_type_all_atoms_ligand_typed);
    f(type_params);
    f(global_params);
    f(output);
    f(dTdV);
    f(dV_dcoords);
  }

  enum { n_views = 23 };

  // Scalar slots in the metadata buffer, after the per-view shapes.
  enum {
    scalar_max_n_blocks = 0,
    scalar_max_n_upper_triangle_inds = 1,
    scalar_output_block_pair_energies = 2,
    scalar_use_shared_compact_block_neighbors = 3,
    scalar_require_gradient = 4,
    scalar_max_dis = 5,
    n_scalars = 6
  };
};

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
