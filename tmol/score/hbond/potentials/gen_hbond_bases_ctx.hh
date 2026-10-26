#pragma once

// The state the derived-atom pre-pass reads.
//
// Declared once, templated on the view, so the CPU and CUDA builds bind it to
// tmol::TView while a `.metal` translation unit -- which cannot include the
// torch tensor layer at all -- binds it to tmol::portable::MetalView and runs
// the same kernel body. The host binds it to TView in the dispatch header.

#include <tmol/utility/portable/device.hh>
#include <tmol/utility/portable/matrix.hh>

namespace tmol {
namespace score {
namespace hbond {
namespace potentials {

using tmol::portable::Vec;

template <template <typename, int> class View, typename Real, typename Int>
struct HBondBasesCtxT {
  // The bond walk constructs a RotamerCentricIndexedBonds, which is itself
  // templated on the view, so the body needs a name for it that does not
  // depend on knowing which backend it was instantiated for.
  template <typename T, int N>
  using ViewT = View<T, N>;

  // Coordinates and the rotamer/block indexing that locates them
  View<Vec<Real, 3>, 1> rot_coords;
  View<Int, 1> rot_coord_offset;
  View<Int, 2> first_rot_for_block;
  View<Int, 2> first_rot_block_type;
  View<Int, 1> block_ind_for_rot;
  View<Int, 1> pose_ind_for_rot;
  View<Int, 1> block_type_ind_for_rot;

  // Connectivity the bond walk crosses when an acceptor base lies in a
  // neighboring block
  View<Vec<Int, 2>, 3> pose_stack_inter_residue_connections;

  // Chemical properties of each block type
  View<Int, 1> block_type_n_atoms;
  View<Int, 1> block_type_n_interblock_bonds;
  View<Int, 2> block_type_atoms_forming_chemical_bonds;
  View<Int, 1> block_type_n_all_bonds;
  View<Vec<Int, 3>, 2> block_type_all_bonds;
  View<Vec<Int, 2>, 2> block_type_atom_all_bond_ranges;
  View<Int, 2> block_type_atom_is_hydrogen;

  // Per-tile donor and acceptor rosters
  View<Int, 2> block_type_tile_n_donH;
  View<Int, 2> block_type_tile_n_acc;
  View<Int, 3> block_type_tile_donH_inds;
  View<Int, 3> block_type_tile_acc_inds;
  View<Int, 3> block_type_tile_hybridization;

  // Outputs, indexed by the global pose-atom index:
  //   [atom][0] : D  (donor heavy) for donor H atoms
  //   [atom][1] : B  (acceptor base) for acceptor atoms
  //   [atom][2] : B0 (acceptor base 2) for acceptor atoms
  // derived_atom_inds mirrors the layout, holding the source atom's global
  // pose-atom index (-1 if unused).
  View<Vec<Real, 3>, 2> derived_coords;
  View<Int, 2> derived_atom_inds;

  // One ordering, two uses: the host walks it to bind each tensor's buffer and
  // record its shape, and the Metal entry walks the same order to rebuild the
  // views. Written exactly once, here, because two hand-maintained lists that
  // drift produce wrong coordinates rather than a build error.
  template <typename F>
  TMOL_HD void visit_views(F f) {
    f(rot_coords);
    f(rot_coord_offset);
    f(first_rot_for_block);
    f(first_rot_block_type);
    f(block_ind_for_rot);
    f(pose_ind_for_rot);
    f(block_type_ind_for_rot);
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
    f(block_type_tile_acc_inds);
    f(block_type_tile_hybridization);
    f(derived_coords);
    f(derived_atom_inds);
  }

  enum { n_views = 22 };

  // This kernel's launch shape is just the rotamer count, which the dispatch
  // grid already carries, so there are no scalar slots.
  template <typename F>
  TMOL_HD void visit_scalars(F) {}
};

}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol
