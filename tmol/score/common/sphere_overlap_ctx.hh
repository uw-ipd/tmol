#pragma once

// The state the two ported sphere-overlap kernels read.
//
// These fields were previously captured implicitly by `[=]` on device
// lambdas. A lambda closure has nowhere to live in Metal, whose kernel entry
// points are `kernel void` functions compiled by a separate toolchain, so the
// bodies take these structs explicitly instead and can be invoked from any
// backend.
//
// Templated on the view so each field list is declared exactly once: the host
// and CUDA builds instantiate with TView, and a Metal translation unit --
// which cannot include the torch tensor layer at all -- instantiates with
// tmol::portable::MetalView.

#include <tmol/score/common/mps_layout.hh>
#include <tmol/utility/portable/device.hh>
#include <tmol/utility/portable/matrix.hh>

namespace tmol {
namespace score {
namespace common {
namespace sphere_overlap {

using tmol::portable::Vec;

// Per-block bounding spheres: one workgroup per rotamer.
template <template <typename, int> class View, typename Real, typename Int>
struct BlockSpheresCtxT {
  View<Vec<Real, 3>, 1> rot_coords;
  View<Int, 1> rot_coord_offset;
  View<Int, 1> block_ind_for_rot;
  View<Int, 1> pose_ind_for_rot;
  View<Int, 1> block_type_ind_for_rot;
  View<Int, 1> block_type_n_atoms;
  View<Real, 3> block_spheres;

  template <typename F>
  TMOL_HD void visit_views(F f) {
    f(rot_coords);
    f(rot_coord_offset);
    f(block_ind_for_rot);
    f(pose_ind_for_rot);
    f(block_type_ind_for_rot);
    f(block_type_n_atoms);
    f(block_spheres);
  }

  template <typename F>
  TMOL_HD void visit_scalars(F) {}

  enum { n_views = 7 };
};

// Pairwise neighbor detection: one thread per candidate block pair.
template <template <typename, int> class View, typename Real, typename Int>
struct BlockNeighborsCtxT {
  View<Int, 2> pose_stack_block_type;
  View<Real, 3> block_spheres;
  View<Int, 3> block_neighbors;

  // Launch shape and the reach cutoff, carried in the metadata buffer.
  int max_n_blocks = {};
  int block_pairs_per_pose = {};
  Real reach = {};

  template <typename F>
  TMOL_HD void visit_views(F f) {
    f(pose_stack_block_type);
    f(block_spheres);
    f(block_neighbors);
  }

  enum { n_views = 3 };

  // Launch scalars in the order the host writes and the shader reads.
  template <typename F>
  TMOL_HD void visit_scalars(F f) {
    f(max_n_blocks);
    f(block_pairs_per_pose);
    f(reach);
  }
};

// ---------------------------------------------------------------------------
// Rotamer-pair dispatch. The packer scores every pair of rotamers whose
// bounding spheres overlap, and the list of those pairs is built by the four
// kernels below. They exist on Metal because that list is the entry condition
// for rotamer-pair energies, which are the largest phase of a relax cycle --
// without them the scoring kernels have nothing to iterate over.

// Per-rotamer bounding spheres: one workgroup per rotamer.
template <template <typename, int> class View, typename Real, typename Int>
struct RotSpheresCtxT {
  View<Vec<Real, 3>, 1> rot_coords;
  View<Int, 1> rot_coord_offset;
  View<Int, 1> block_type_ind_for_rot;
  View<Int, 1> block_type_n_atoms;
  View<Real, 2> rot_spheres;

  template <typename F>
  TMOL_HD void visit_views(F f) {
    f(rot_coords);
    f(rot_coord_offset);
    f(block_type_ind_for_rot);
    f(block_type_n_atoms);
    f(rot_spheres);
  }

  enum { n_views = 5 };
  template <typename F>
  TMOL_HD void visit_scalars(F) {}
};

// A block's sphere from the spheres of its rotamers: one thread per block.
template <template <typename, int> class View, typename Real, typename Int>
struct BlockSpheresFromRotCtxT {
  View<Real, 2> rot_spheres;
  View<Int, 2> n_rots_for_block;
  View<Int, 2> rot_offset_for_block;
  View<Real, 3> block_spheres;

  template <typename F>
  TMOL_HD void visit_views(F f) {
    f(rot_spheres);
    f(n_rots_for_block);
    f(rot_offset_for_block);
    f(block_spheres);
  }

  enum { n_views = 4 };
  // Launch scalars in the order the host writes and the shader reads.
  template <typename F>
  TMOL_HD void visit_scalars(F f) {
    f(max_n_blocks);
  }

  int max_n_blocks = {};
};

// Counting and filling the rotamer-pair list. One thread per candidate block
// pair in both passes, so the two share a context; the fill pass additionally
// reads the exclusive scan of the counts.
template <template <typename, int> class View, typename Real, typename Int>
struct RotPairListCtxT {
  View<Int, 2> pose_stack_block_type;
  View<Real, 3> block_spheres;
  View<Int, 2> n_rots_for_block;
  View<Int, 2> rot_offset_for_block;
  View<Real, 2> rot_spheres;
  View<Int, 2> lockstep_group_for_block;
  View<Int, 1> pair_counts;
  View<Int, 1> pair_offsets;
  View<Int, 2> indices;
  // Set to 1 by any candidate whose rotamer-pair count will not fit in Int;
  // every writer stores the same value, so no atomic is needed.
  View<int64_t, 1> pair_count_overflow;

  template <typename F>
  TMOL_HD void visit_views(F f) {
    f(pose_stack_block_type);
    f(block_spheres);
    f(n_rots_for_block);
    f(rot_offset_for_block);
    f(rot_spheres);
    f(lockstep_group_for_block);
    f(pair_counts);
    f(pair_offsets);
    f(indices);
    f(pair_count_overflow);
  }

  enum { n_views = 10 };
  // Launch scalars in the order the host writes and the shader reads.
  template <typename F>
  TMOL_HD void visit_scalars(F f) {
    f(max_n_blocks);
    f(block_pairs_per_pose);
    f(candidate_begin);
    f(reach);
  }

  int max_n_blocks = {};
  int block_pairs_per_pose = {};
  int candidate_begin = {};
  Real reach = {};
};

}  // namespace sphere_overlap
}  // namespace common
}  // namespace score
}  // namespace tmol
