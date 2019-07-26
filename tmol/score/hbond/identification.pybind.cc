#include <pybind11/eigen.h>
#include <tmol/utility/tensor/pybind.h>
#include <torch/extension.h>
#include <cppitertools/range.hpp>

#include "../bonded_atom.pybind.hh"
#include "identification.hh"

namespace tmol {
namespace score {
namespace hbond {

template <typename Int, tmol::Device D>
void id_acceptor_bases(
    TView<Int, 2, D> A_idx,
    TView<Int, 2, D> B_idx,
    TView<Int, 2, D> B0_idx,
    TView<Int, 2, D> atom_hybridization,
    TView<bool, 2, D> atom_is_hydrogen,
    TView<Vec<Int, 2>, 2, D> bonds,
    TView<Vec<Int, 2>, 2, D> bond_spans) {
  // fd temporary:
  // fd  currently is_hydrogen is denormalized here but not in lk_ball
  // fd  handle both cases w/o duplicating logic
  // fd this layer will be unneeded following torchscripting of hbond
  auto is_hydrogen = ([&] EIGEN_DEVICE_FUNC(int stack, int j) {
    return atom_is_hydrogen[stack][j];
  });

  for (int stack : iter::range(A_idx.size(0))) {
    IndexedBonds<Int, D> indexed_bonds({bonds[stack], bond_spans[stack]});
    for (int ai : iter::range(A_idx.size(1))) {
      // atoms with negative indices are not real
      // the negative index is a sentinel value that
      // pads entires so systems of different size/composition
      // can be stacked together
      if (A_idx[stack][ai] < 0) continue;

      auto bases = AcceptorBases<Int>::for_acceptor(
          stack,
          A_idx[stack][ai],
          atom_hybridization[stack][A_idx[stack][ai]],
          indexed_bonds,
          is_hydrogen);

      A_idx[stack][ai] = bases.A;
      B_idx[stack][ai] = bases.B;
      B0_idx[stack][ai] = bases.B0;
    }
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace pybind11::literals;

  m.def(
      "id_acceptor_bases",
      &id_acceptor_bases<int64_t, tmol::Device::CPU>,
      "A_idx"_a,
      "B_idx"_a,
      "B0_idx"_a,
      "atom_hybridization"_a,
      "atom_is_hydrogen"_a,
      "bonds"_a,
      "bond_spans"_a);
}

}  // namespace hbond
}  // namespace score
}  // namespace tmol
