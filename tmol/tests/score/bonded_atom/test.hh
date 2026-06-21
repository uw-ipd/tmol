#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>

#include <tmol/score/common/dispatch.hh>
#include <tuple>

namespace tmol {
namespace tests {
namespace score {
namespace bonded_atom {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <template <Device> class Dispatch, Device D, typename Int>
struct BondedAtomTests {
  static auto f(
      TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
      TView<Int, 2, D> pose_stack_block_type,

      TView<Int, 1, D> block_type_n_atoms,

      // Properties of the block types in this molecular system
      TView<Int, 1, D> block_type_n_all_bonds,
      TView<Vec<Int, 3>, 2, D> block_type_all_bonds,
      TView<Vec<Int, 2>, 2, D> block_type_atom_all_bond_ranges,
      TView<Int, 2, D> block_type_atoms_forming_chemical_bonds)
      -> std::tuple<TPack<Vec<Int, 2>, 3, D>, TPack<Vec<Int, 2>, 3, D>>;
};

}  // namespace bonded_atom
}  // namespace score
}  // namespace tests
}  // namespace tmol
