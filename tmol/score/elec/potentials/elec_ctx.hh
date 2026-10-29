#pragma once

#include <tmol/score/elec/potentials/params_types.hh>
#include <tmol/utility/portable/matrix.hh>
#include <tmol/utility/portable/device.hh>

namespace tmol {
namespace score {
namespace elec {
namespace potentials {

using tmol::portable::Vec;

// The state an elec kernel body reads.
//
// These fields were previously captured implicitly by `[=]` on a device
// lambda. A lambda closure has nowhere to live in Metal, whose kernel entry
// points are `kernel void` functions compiled by a separate toolchain, so the
// bodies take this struct explicitly instead and can be invoked from any
// backend.
//
// Naming the state also makes its size visible: the dispatch entry points take
// 94 tensor arguments, but the five kernel bodies read only these, and each
// body reads a subset -- the struct is their union, so one type serves the
// scoring and derivative kernels alike. Fields a given body does not read are
// left default-constructed.
//
// Templated on the view so the field list is declared exactly once: the host
// and CUDA builds instantiate it with TView, and a Metal translation unit --
// which cannot include the torch tensor layer at all -- instantiates it with
// tmol::portable::MetalView. The host binds it to TView in the dispatch header.
//
// OutRank parameterizes the energy/derivative tensors, which differ between
// the two kernel families: whole-pose scoring writes
// [term, pose, block, block] while rotamer scoring writes
// [term, block_pair]. Everything else they read is identical.
template <
    template <typename, int> class View,
    typename Real,
    typename Int,
    int OutRank>
struct ElecCtxT {
  // Coordinates and the rotamer/block indexing that locates them
  View<Vec<Real, 3>, 1> rot_coords;
  View<Int, 1> rot_coord_offset;
  View<Int, 1> block_type_ind_for_rot;
  View<Int, 1> block_ind_for_rot;
  View<Int, 2> rot_offset_for_block;

  // Inter-block proximity and bond separation
  View<Int, 4> pose_stack_near_blocks;
  View<int8_t, 5> pose_stack_inter_block_bondsep;
  View<Int, 3> scratch_rot_neighbors;

  // Chemical properties of each block type
  View<Int, 1> block_type_n_atoms;
  View<Real, 2> block_type_partial_charge;
  View<Int, 1> block_type_n_interblock_bonds;
  View<Int, 2> block_type_atoms_forming_chemical_bonds;
  View<Int, 3> block_type_inter_repr_path_distance;
  View<Int, 3> block_type_intra_repr_path_distance;
  View<Int, 1> block_type_all_atoms_ligand_typed;

  View<ElecGlobalParams<Real>, 1> global_params;

  // Work-item mapping used by the block-pair and derivative kernels
  View<Int, 2> dispatch_indices;

  // Outputs
  View<Real, OutRank> output;
  View<Real, OutRank> dTdV;
  View<Vec<Real, 3>, 2> dV_dcoords;

  // Enumerate the views in a fixed order, applying f to each.
  //
  // One ordering, two uses. The host walks it to pack the read-only parameter
  // tables into a single device buffer and to record each one's offset; the
  // Metal entry walks the same order to rebuild the views from that buffer. A
  // Metal kernel entry cannot take twenty-one separately bound buffers'
  // worth of parameters without the host and the shader each spelling the
  // list out, and two hand-maintained lists that drift produce wrong energies
  // rather than a build error -- so the list is written exactly once, here.
  //
  // f is applied to the view itself, so a visitor may read it (packing) or
  // assign to it (rebuilding). Rank and element type come from the view's own
  // type, which is why the visitor has to be a generic callable.
  template <typename F>
  TMOL_HD void visit_views(F f) {
    f(rot_coords);
    f(rot_coord_offset);
    f(block_type_ind_for_rot);
    f(block_ind_for_rot);
    f(rot_offset_for_block);
    f(pose_stack_near_blocks);
    f(pose_stack_inter_block_bondsep);
    f(scratch_rot_neighbors);
    f(block_type_n_atoms);
    f(block_type_partial_charge);
    f(block_type_n_interblock_bonds);
    f(block_type_atoms_forming_chemical_bonds);
    f(block_type_inter_repr_path_distance);
    f(block_type_intra_repr_path_distance);
    f(block_type_all_atoms_ligand_typed);
    f(global_params);
    f(dispatch_indices);
    f(output);
    f(dTdV);
    f(dV_dcoords);
  }

  // Number of views visit_views enumerates. Checked against the buffer
  // bindings in the Metal entry points so that adding a field without adding
  // it to visit_views is caught at build time rather than silently left
  // unbound.
  enum { n_views = 20 };

  // Launch scalars in the order the host writes and the shader reads.
  template <typename F>
  TMOL_HD void visit_scalars(F f) {
    f(max_n_blocks);
    f(max_n_upper_triangle_inds);
    f(output_block_pair_energies);
    f(use_shared_compact_block_neighbors);
  }

  // Launch-shape constants
  int max_n_blocks = {};
  int max_n_upper_triangle_inds = {};
  bool output_block_pair_energies = {};
  bool use_shared_compact_block_neighbors = {};
};

}  // namespace potentials
}  // namespace elec
}  // namespace score
}  // namespace tmol
