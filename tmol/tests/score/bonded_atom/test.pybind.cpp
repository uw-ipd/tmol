#include <tmol/utility/tensor/pybind.h>

#include <tmol/score/common/forall_dispatch.hh>
#include <tmol/tests/score/bonded_atom/test.hh>

namespace tmol {
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace pybind11::literals;

  m.def(
      "two_steps",
      &tmol::tests::score::bonded_atom::BondedAtomTests<
          tmol::score::common::ForallDispatch,
          Device::CPU,
          int32_t>::f,
      "pose_stack_inter_block_connections"_a,
      "pose_stack_block_type"_a,
      "block_type_n_atoms"_a,
      "block_type_n_all_bonds"_a,
      "block_type_atom_all_bonds"_a,
      "block_type_atom_all_bond_ranges"_a,
      "block_type_atoms_forming_chemical_bonds"_a);

#ifdef WITH_CUDA
  m.def(
      "two_steps",
      &tmol::tests::score::bonded_atom::BondedAtomTests<
          tmol::score::common::ForallDispatch,
          Device::CUDA,
          int32_t>::f,
      "pose_stack_inter_block_connections"_a,
      "pose_stack_block_type"_a,
      "block_type_n_atoms"_a,
      "block_type_n_all_bonds"_a,
      "block_type_atom_all_bonds"_a,
      "block_type_atom_all_bond_ranges"_a,
      "block_type_atoms_forming_chemical_bonds"_a);

#endif
}
}  // namespace tmol
