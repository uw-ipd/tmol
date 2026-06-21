#pragma once

#include <tmol/extern/moderngpu/operators.hxx>
#include <tmol/tests/score/bonded_atom/test.hh>
#include <tmol/utility/tensor/TensorPack.h>
#include <Eigen/Core>

namespace tmol {
namespace tests {
namespace score {
namespace bonded_atom {

template <template <Device> class Dispatch, Device D, typename Int>
auto BondedAtomTests<Dispatch, D, Int>::f(
    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<Int, 2, D> pose_stack_block_type,

    TView<Int, 1, D> block_type_n_atoms,

    // Properties of the block types in this molecular system
    TView<Int, 1, D> block_type_n_all_bonds,
    TView<Vec<Int, 3>, 2, D> block_type_all_bonds,
    TView<Vec<Int, 2>, 2, D> block_type_atom_all_bond_ranges,
    TView<Int, 2, D> block_type_atoms_forming_chemical_bonds)
    -> std::tuple<TPack<Vec<Int, 2>, 3, D>, TPack<Vec<Int, 2>, 3, D>> {
  int const n_poses = pose_stack_inter_block_connections.size(0);
  int const max_n_blocks = pose_stack_inter_block_connections.size(1);
  int const max_n_conn = pose_stack_inter_block_connections.size(2);
  int const n_block_types = block_type_n_atoms.size(0);
  int const max_n_all_bonds = block_type_all_bonds.size(1);
  int const max_n_atoms = block_type_atom_all_bond_ranges.size(1);

  assert(n_poses == pose_stack_block_type.size(0));
  assert(n_block_types == block_type_n_all_bonds.size(0));
  assert(n_block_types == block_type_atom_all_bond_ranges.size(0));
  assert(n_block_types == block_type_atoms_forming_chemical_bonds.size(0));
  assert(max_n_conn == block_type_atoms_forming_chemical_bonds.size(1));

  auto one_step_t =
      TPack<Vec<Int, 2>, 3, D>::empty({n_poses, max_n_blocks, max_n_atoms});
  auto one_step = one_step_t.view;
  auto two_steps_t =
      TPack<Vec<Int, 2>, 3, D>::empty({n_poses, max_n_blocks, max_n_atoms});
  auto two_steps = two_steps_t.view;

  auto take_two_steps = ([=] EIGEN_DEVICE_FUNC(int ind) {
    using tmol::score::bonded_atom::BlockCentricIndexedBonds;
    using tmol::score::bonded_atom::BlockCentricAtom;

    int const pose_ind = ind / (max_n_blocks * max_n_atoms);
    int const pose_rem = ind % (max_n_blocks * max_n_atoms);
    int const block_ind = pose_rem / max_n_atoms;
    int const atom_ind = pose_rem % max_n_atoms;
    int const block_type = pose_stack_block_type[pose_ind][block_ind];

    // Default: mark all neighbors as unresolved and change when we find the
    // neighbor
    one_step[pose_ind][block_ind][atom_ind][0] = -1;
    one_step[pose_ind][block_ind][atom_ind][1] = -1;
    two_steps[pose_ind][block_ind][atom_ind][0] = -1;
    two_steps[pose_ind][block_ind][atom_ind][1] = -1;

    if (block_type == -1) {
      return;
    }

    int const n_atoms = block_type_n_atoms[block_type];
    if (atom_ind >= n_atoms) {
      return;
    }

    BlockCentricIndexedBonds<Int, D> indexed_bonds{
        pose_stack_inter_block_connections[pose_ind],
        pose_stack_block_type[pose_ind],
        block_type_n_all_bonds,
        block_type_all_bonds,
        block_type_atom_all_bond_ranges,
        block_type_atoms_forming_chemical_bonds};

    BlockCentricAtom<Int> focused_atom{block_ind, block_type, atom_ind};
    BlockCentricAtom<Int> nb1{-1, -1, -1};
    BlockCentricAtom<Int> nb2{-1, -1, -1};

    for (auto neighb : indexed_bonds.bound_to(focused_atom)) {
      if (neighb.atom != -1) {
        nb1 = neighb;
        break;
      }
    }

    one_step[pose_ind][block_ind][atom_ind][0] = nb1.block;
    one_step[pose_ind][block_ind][atom_ind][1] = nb1.atom;

    if (nb1.atom == -1) {
      return;
    }

    for (auto neighb : indexed_bonds.bound_to(nb1)) {
      if (neighb != focused_atom && neighb.atom != -1) {
        nb2 = neighb;
        break;
      }
    }

    two_steps[pose_ind][block_ind][atom_ind][0] = nb2.block;
    two_steps[pose_ind][block_ind][atom_ind][1] = nb2.atom;
  });

  Dispatch<D>::forall(n_poses * max_n_blocks * max_n_atoms, take_two_steps);

  return {one_step_t, two_steps_t};
};

}  // namespace bonded_atom
}  // namespace score
}  // namespace tests
}  // namespace tmol
