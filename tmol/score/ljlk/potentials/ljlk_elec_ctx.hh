#pragma once

// The state the fused LJ/LK + elec rotamer-pair kernel reads.
//
// The fused CPU and CUDA kernel captures these as TView locals of one large
// lambda. A Metal entry point has nowhere for such a closure to live, so the
// Metal path names them here instead, in the same way ljlk_ctx.hh and
// elec_ctx.hh do for the unfused terms.
//
// Only the packing variant is described: rotamer pairs, weighted, no
// gradient. That is the only one Metal runs; see ljlk_elec_kernels.hh.
//
// Templated on the view so the field list is declared exactly once: the host
// instantiates it with TView, and a Metal translation unit -- which cannot
// include the torch tensor layer -- with tmol::portable::MetalView.

#include <tmol/score/elec/potentials/params_types.hh>
#include <tmol/score/ljlk/potentials/params_types.hh>
#include <tmol/utility/portable/device.hh>
#include <tmol/utility/portable/matrix.hh>

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

template <template <typename, int> class View, typename Real, typename Int>
struct LJLKElecRotCtxT {
  // Coordinates and the rotamer/block indexing that locates them
  View<tmol::portable::Vec<Real, 3>, 1> rot_coords;
  View<Int, 1> rot_coord_offset;
  View<Int, 1> block_ind_for_rot;
  View<Int, 1> block_type_ind_for_rot;

  // Inter-block proximity and bond separation
  View<Int, 4> pose_stack_near_blocks;
  View<int8_t, 5> pose_stack_inter_block_bondsep;

  // Rotamer-pair dispatch list, [3, n_pairs]: pose, rotamer 1, rotamer 2
  View<Int, 2> dispatch_indices;

  // Chemical properties of each block type
  View<Int, 1> block_type_n_atoms;
  View<Int, 2> block_type_atom_types;
  View<Int, 1> block_type_n_interblock_bonds;
  View<Int, 2> block_type_atoms_forming_chemical_bonds;
  View<Int, 3> block_type_ljlk_path_distance;
  View<Int, 1> block_type_all_atoms_ligand_typed;
  View<LJLKTypeParams<Real>, 1> ljlk_type_params;
  View<LJGlobalParams<Real>, 1> ljlk_global_params;
  View<Real, 2> block_type_partial_charge;
  View<Int, 3> block_type_elec_inter_repr_path_distance;
  View<Int, 3> block_type_elec_intra_repr_path_distance;
  View<tmol::score::elec::potentials::ElecGlobalParams<Real>, 1>
      elec_global_params;

  // The four score weights (ljatr, ljrep, lk, elec), and one weighted energy
  // per dispatch entry, [1, 1, 1, n_pairs].
  View<Real, 1> score_weights;
  View<Real, 4> output;

  // Enumerate the views in a fixed order, applying f to each. The host walks
  // it to bind each view's buffer; the Metal entry walks the same order to
  // rebuild the views.
  template <typename F>
  TMOL_HD void visit_views(F f) {
    f(rot_coords);
    f(rot_coord_offset);
    f(block_ind_for_rot);
    f(block_type_ind_for_rot);
    f(pose_stack_near_blocks);
    f(pose_stack_inter_block_bondsep);
    f(dispatch_indices);
    f(block_type_n_atoms);
    f(block_type_atom_types);
    f(block_type_n_interblock_bonds);
    f(block_type_atoms_forming_chemical_bonds);
    f(block_type_ljlk_path_distance);
    f(block_type_all_atoms_ligand_typed);
    f(ljlk_type_params);
    f(ljlk_global_params);
    f(block_type_partial_charge);
    f(block_type_elec_inter_repr_path_distance);
    f(block_type_elec_intra_repr_path_distance);
    f(elec_global_params);
    f(score_weights);
    f(output);
  }

  enum { n_views = 21 };
  enum { n_scalars = 0 };
};

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
