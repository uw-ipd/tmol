#pragma once

#include <tmol/score/hbond/potentials/params_types.hh>
#include <tmol/utility/portable/device.hh>
#include <tmol/utility/portable/matrix.hh>

namespace tmol {
namespace score {
namespace hbond {
namespace potentials {

using tmol::portable::Vec;

// The state an hbond kernel body reads.
//
// These fields were previously captured implicitly by `[=]` on a device
// lambda. A lambda closure has nowhere to live in Metal, whose kernel entry
// points are `kernel void` functions compiled by a separate toolchain, so the
// bodies take this struct explicitly instead and can be invoked from any
// backend.
//
// Naming the state also makes its size visible: the dispatch entry points take
// upwards of fifty tensor arguments, but the kernel bodies read only these --
// most of the rest feed the derived-atom pre-pass, not the scoring itself.
// Each body reads a subset; the struct is their union, so one type serves the
// whole-pose and rotamer-pair families alike.
//
// Templated on the view so the field list is declared exactly once: the host
// and CUDA builds instantiate it with TView, and a Metal translation unit --
// which cannot include the torch tensor layer at all -- instantiates it with
// tmol::portable::MetalView. See hbond_ctx_tensor.hh for the host alias.
//
// PolyReal is the polynomial coefficient type, which is the one field whose
// element type is backend-dependent: float64 on the host and CUDA, an
// unevaluated hi/lo float32 pair on Metal, which has no float64 at all. Both
// occupy eight bytes, so the table's layout is the same either way. See
// params_types.hh.
//
// OutRank and DispRank parameterize the tensors that differ between the two
// kernel families: whole-pose scoring writes [term, pose, block, block] and
// dispatches over [3, pose, block_pair], while rotamer scoring writes
// [term, rot_pair] and dispatches over [3, rot_pair]. Everything else they
// read is identical.
//
// DerivRank differs between the passes rather than the families: a forward
// pass accumulates into [term, atom] and a backward pass into [atom].
template <
    template <typename, int> class View,
    typename Real,
    typename Int,
    typename PolyReal,
    int OutRank,
    int DispRank,
    int DerivRank = 2>
struct HBondCtxT {
  // Coordinates and the rotamer/block indexing that locates them
  View<Vec<Real, 3>, 1> rot_coords;
  View<Int, 1> rot_coord_offset;
  View<Int, 1> block_ind_for_rot;
  View<Int, 1> block_type_ind_for_rot;
  View<Int, 2> rot_offset_for_block;

  // Inter-block proximity and bond separation
  View<Int, 4> pose_stack_near_blocks;
  View<int8_t, 5> pose_stack_inter_block_bondsep;

  // Chemical properties of each block type
  View<Int, 1> block_type_n_atoms;
  View<Int, 1> block_type_n_interblock_bonds;
  View<Int, 2> block_type_atoms_forming_chemical_bonds;
  View<Int, 3> block_type_path_distance;

  // Per-tile donor and acceptor rosters
  View<Int, 2> block_type_tile_n_donH;
  View<Int, 2> block_type_tile_n_acc;
  View<Int, 3> block_type_tile_donH_inds;
  View<Int, 3> block_type_tile_acc_inds;
  View<Int, 3> block_type_tile_donor_type;
  View<Int, 3> block_type_tile_acceptor_type;
  View<Int, 3> block_type_tile_hybridization;

  // Potential parameters
  View<HBondPairParams<Real>, 2> pair_params;
  View<HBondPolynomials<PolyReal>, 2> pair_polynomials;
  View<HBondGlobalParams<Real>, 1> global_params;

  // Derived-atom coordinates and source-atom indices, from the pre-pass
  View<Vec<Real, 3>, 2> derived_coords;
  View<Int, 2> derived_atom_inds;

  // Work-item mapping
  View<Int, DispRank> dispatch_indices;

  // Outputs
  View<Real, OutRank> output;
  View<Real, OutRank> dTdV;
  View<Vec<Real, 3>, DerivRank> dV_dcoords;

  // Enumerate the views in a fixed order, applying f to each.
  //
  // One ordering, two uses. The host walks it to bind each tensor's buffer and
  // to record its shape in the metadata buffer; the Metal entry walks the same
  // order to rebuild the views from those bindings. A Metal kernel entry
  // cannot take this many separately bound buffers without the host and the
  // shader each spelling the list out, and two hand-maintained lists that
  // drift produce wrong energies rather than a build error -- so the list is
  // written exactly once, here.
  //
  // f is applied to the view itself, so a visitor may read it (binding) or
  // assign to it (rebuilding). Rank and element type come from the view's own
  // type, which is why the visitor has to be a generic callable.
  template <typename F>
  TMOL_HD void visit_views(F f) {
    f(rot_coords);
    f(rot_coord_offset);
    f(block_ind_for_rot);
    f(block_type_ind_for_rot);
    f(rot_offset_for_block);
    f(pose_stack_near_blocks);
    f(pose_stack_inter_block_bondsep);
    f(block_type_n_atoms);
    f(block_type_n_interblock_bonds);
    f(block_type_atoms_forming_chemical_bonds);
    f(block_type_path_distance);
    f(block_type_tile_n_donH);
    f(block_type_tile_n_acc);
    f(block_type_tile_donH_inds);
    f(block_type_tile_acc_inds);
    f(block_type_tile_donor_type);
    f(block_type_tile_acceptor_type);
    f(block_type_tile_hybridization);
    f(pair_params);
    f(pair_polynomials);
    f(global_params);
    f(derived_coords);
    f(derived_atom_inds);
    f(dispatch_indices);
    f(output);
    f(dTdV);
    f(dV_dcoords);
  }

  // Number of views visit_views enumerates. Checked against the buffer
  // bindings in the Metal entry points so that adding a field without adding
  // it to visit_views is caught at build time rather than silently left
  // unbound.
  enum { n_views = 27 };

  // Scalar slots in the metadata buffer, after the per-view shapes.
  enum {
    scalar_max_n_blocks = 0,
    scalar_output_block_pair_energies = 1,
    scalar_accumulate_derivs = 2,
    scalar_max_n_upper_triangle_inds = 3,
    scalar_use_shared_compact_block_neighbors = 4,
    n_scalars = 5
  };

  // Launch-shape constants
  int max_n_blocks;
  bool output_block_pair_energies;
  // Whether the forward pass also accumulates coordinate derivatives into
  // dV_dcoords. Block-pair autograd recomputes them in backward instead, and
  // the packer never wants them at all.
  bool accumulate_derivs;
  // Whole-pose only: the workgroup index decomposes into a pose and a position
  // in the block-pair upper triangle, and a precomputed neighbor list -- when
  // one was supplied -- replaces the per-pair neighbor test.
  int max_n_upper_triangle_inds;
  bool use_shared_compact_block_neighbors;
};

}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol
