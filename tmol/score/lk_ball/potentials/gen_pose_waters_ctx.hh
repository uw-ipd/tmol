#pragma once

// The state the water-generation kernel body reads.
//
// These fields were previously captured implicitly by `[=]` on a device
// lambda. A lambda closure has nowhere to live in Metal, whose kernel entry
// points are `kernel void` functions compiled by a separate toolchain, so the
// body takes this struct explicitly instead and can be invoked from any
// backend.
//
// LK-ball needs two Metal entries, not one: water generation is a separate
// op that runs before scoring, and the waters it writes are an input to the
// scoring kernel. Leaving it on the CPU would mean copying the pose to the
// host and back on every call.
//
// Templated on the view so the field list is declared exactly once; see
// gen_pose_waters_ctx_tensor.hh for the host alias and lk_ball_ctx.hh for
// the same pattern on the scoring side.

#include <tmol/score/lk_ball/potentials/params_types.hh>
#include <tmol/score/lk_ball/potentials/water.hh>
#include <tmol/utility/portable/device.hh>
#include <tmol/utility/portable/matrix.hh>

namespace tmol {
namespace score {
namespace lk_ball {
namespace potentials {

using tmol::portable::Vec;

template <template <typename, int> class View, typename Real, typename Int>
struct WaterGenCtxT {
  // Coordinates and the block indexing that locates them
  View<Vec<Real, 3>, 1> rot_coords;
  View<Int, 1> rot_coord_offset;
  View<Int, 1> pose_ind_for_rot;
  View<Int, 1> block_ind_for_rot;
  View<Int, 1> block_type_ind_for_rot;
  View<Int, 2> first_rot_for_block;
  View<Int, 2> first_rot_block_type;
  View<Int, 2> lockstep_group_for_block;
  View<Int, 2> rot_offset_for_block;
  View<Vec<Int, 2>, 3> pose_stack_inter_residue_connections;

  // Chemical properties of each block type
  View<Int, 1> block_type_n_atoms;
  View<Int, 1> block_type_n_interblock_bonds;
  View<Int, 2> block_type_atoms_forming_chemical_bonds;
  View<Int, 1> block_type_n_all_bonds;
  View<Vec<Int, 3>, 2> block_type_all_bonds;
  View<Vec<Int, 2>, 2> block_type_atom_all_bond_ranges;
  View<Int, 2> block_type_atom_is_hydrogen;

  // Per-tile donor and acceptor listings
  View<Int, 2> block_type_tile_n_donH;
  View<Int, 2> block_type_tile_n_acc;
  View<Int, 3> block_type_tile_donH_inds;
  View<Int, 3> block_type_tile_don_hvy_inds;
  View<Int, 3> block_type_tile_which_donH_for_hvy;
  View<Int, 3> block_type_tile_acc_inds;
  View<Int, 3> block_type_tile_hybridization;
  View<Int, 3> block_type_tile_acc_n_attached_H;

  // Water-building geometry
  View<LKBallWaterGenGlobalParams<Real>, 1> global_params;
  View<Real, 1> sp2_water_tors;
  View<Real, 1> sp3_water_tors;
  View<Real, 1> ring_water_tors;

  // Output
  View<Vec<Real, 3>, 2> water_coords;

  // Enumerate the views in a fixed order, applying f to each.
  //
  // One ordering, two uses: the host binds buffers in this order, the Metal
  // entry rebuilds the views in the same order. See elec_ctx.hh.
  template <typename F>
  TMOL_HD void visit_views(F f) {
    f(rot_coords);
    f(rot_coord_offset);
    f(pose_ind_for_rot);
    f(block_ind_for_rot);
    f(block_type_ind_for_rot);
    f(first_rot_for_block);
    f(first_rot_block_type);
    f(lockstep_group_for_block);
    f(rot_offset_for_block);
    f(pose_stack_inter_residue_connections);
    f(block_type_n_atoms);
    f(block_type_n_interblock_bonds);
    f(block_type_atoms_forming_chemical_bonds);
    f(block_type_n_all_bonds);
    f(block_type_all_bonds);
    f(block_type_atom_all_bond_ranges);
    f(block_type_atom_is_hydrogen);
    f(block_type_tile_n_donH);
    f(block_type_tile_n_acc);
    f(block_type_tile_donH_inds);
    f(block_type_tile_don_hvy_inds);
    f(block_type_tile_which_donH_for_hvy);
    f(block_type_tile_acc_inds);
    f(block_type_tile_hybridization);
    f(block_type_tile_acc_n_attached_H);
    f(global_params);
    f(sp2_water_tors);
    f(sp3_water_tors);
    f(ring_water_tors);
    f(water_coords);
  }

  // 30 views plus the metadata buffer is 31 bindings, which is the per-kernel
  // buffer limit on Apple GPUs. There is no room for another field here
  // without packing several of the small per-block-type tables into one
  // buffer.
  enum { n_views = 30 };

  // Launch scalars in the order the host writes and the shader reads.
  template <typename F>
  TMOL_HD void visit_scalars(F f) {
    f(n_rots_total);
  }

  // Launch-shape constant
  int n_rots_total = {};
};

// Backward pass. Its own field list rather than a reuse of WaterGenCtxT:
// it needs two accumulators the forward does not, and Metal's 31-binding
// limit leaves no room for both those and the forward's water_coords.
template <template <typename, int> class View, typename Real, typename Int>
struct WaterGenDerivCtxT {
  // Coordinates and the block indexing that locates them
  View<Vec<Real, 3>, 1> rot_coords;
  View<Int, 1> rot_coord_offset;
  View<Int, 1> pose_ind_for_rot;
  View<Int, 1> block_ind_for_rot;
  View<Int, 1> block_type_ind_for_rot;
  View<Int, 2> first_rot_for_block;
  View<Int, 2> first_rot_block_type;
  View<Int, 2> lockstep_group_for_block;
  View<Int, 2> rot_offset_for_block;
  View<Vec<Int, 2>, 3> pose_stack_inter_residue_connections;

  // Chemical properties of each block type
  View<Int, 1> block_type_n_atoms;
  View<Int, 1> block_type_n_interblock_bonds;
  View<Int, 2> block_type_atoms_forming_chemical_bonds;
  View<Int, 1> block_type_n_all_bonds;
  View<Vec<Int, 3>, 2> block_type_all_bonds;
  View<Vec<Int, 2>, 2> block_type_atom_all_bond_ranges;
  View<Int, 2> block_type_atom_is_hydrogen;

  // Per-tile donor and acceptor listings
  View<Int, 2> block_type_tile_n_donH;
  View<Int, 2> block_type_tile_n_acc;
  View<Int, 3> block_type_tile_donH_inds;
  View<Int, 3> block_type_tile_don_hvy_inds;
  View<Int, 3> block_type_tile_which_donH_for_hvy;
  View<Int, 3> block_type_tile_acc_inds;
  View<Int, 3> block_type_tile_hybridization;
  View<Int, 3> block_type_tile_acc_n_attached_H;

  // Water-building geometry. The three torsion tables -- at most a couple of
  // entries each -- are packed into one [3, max_len] row-major table rather
  // than three views: a backward context needs two accumulators the forward
  // does not, and Metal binds at most 31 buffers (the metadata at 0 plus 30
  // views). The true row lengths travel as launch scalars.
  View<LKBallWaterGenGlobalParams<Real>, 1> global_params;
  View<Real, 2> water_tors;

  // Backward outputs
  View<Vec<Real, 3>, 2> dE_dWxyz;
  View<Vec<Real, 3>, 1> dE_d_pose_coords;

  template <typename F>
  TMOL_HD void visit_views(F f) {
    f(rot_coords);
    f(rot_coord_offset);
    f(pose_ind_for_rot);
    f(block_ind_for_rot);
    f(block_type_ind_for_rot);
    f(first_rot_for_block);
    f(first_rot_block_type);
    f(lockstep_group_for_block);
    f(rot_offset_for_block);
    f(pose_stack_inter_residue_connections);
    f(block_type_n_atoms);
    f(block_type_n_interblock_bonds);
    f(block_type_atoms_forming_chemical_bonds);
    f(block_type_n_all_bonds);
    f(block_type_all_bonds);
    f(block_type_atom_all_bond_ranges);
    f(block_type_atom_is_hydrogen);
    f(block_type_tile_n_donH);
    f(block_type_tile_n_acc);
    f(block_type_tile_donH_inds);
    f(block_type_tile_don_hvy_inds);
    f(block_type_tile_which_donH_for_hvy);
    f(block_type_tile_acc_inds);
    f(block_type_tile_hybridization);
    f(block_type_tile_acc_n_attached_H);
    f(global_params);
    f(water_tors);
    f(dE_dWxyz);
    f(dE_d_pose_coords);
  }

  // 30 views plus the metadata buffer is 31 bindings, which is the per-kernel
  // buffer limit on Apple GPUs. There is no room for another field here
  // without packing several of the small per-block-type tables into one
  // buffer.
  enum { n_views = 29 };

  // Launch scalars in the order the host writes and the shader reads.
  template <typename F>
  TMOL_HD void visit_scalars(F f) {
    f(n_sp2_water_tors);
    f(n_sp3_water_tors);
    f(n_ring_water_tors);
  }

  int n_sp2_water_tors = {};
  int n_sp3_water_tors = {};
  int n_ring_water_tors = {};
};

}  // namespace potentials
}  // namespace lk_ball
}  // namespace score
}  // namespace tmol
