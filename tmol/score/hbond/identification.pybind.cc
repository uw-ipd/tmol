#include <pybind11/eigen.h>
#include <tmol/utility/tensor/pybind.h>
#include <torch/torch.h>
#include <cppitertools/range.hpp>

#include "../bonded_atom.pybind.hh"
#include "identification.hh"

namespace tmol {
namespace score {
namespace hbond {

template <typename Int, tmol::Device D>
void id_acceptor_bases(
    TView<Int, 1, D> A_idx,
    TView<Int, 1, D> B_idx,
    TView<Int, 1, D> B0_idx,
    TView<Int, 1, D> atom_hybridization,
    TView<bool, 1, D> atom_is_hydrogen,
    IndexedBonds<Int, D> bonds) {
  for (int ai : iter::range(A_idx.size(0))) {
    auto bases = AcceptorBases<Int>::for_acceptor(
        *A_idx[ai], *atom_hybridization[*A_idx[ai]], bonds, atom_is_hydrogen);

    *A_idx[ai] = bases.A;
    *B_idx[ai] = bases.B;
    *B0_idx[ai] = bases.B0;
  };
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
      "bonds"_a);
}

}  // namespace hbond
}  // namespace score
}  // namespace tmol
