#pragma once

#include <tmol/utility/tensor/TensorAccessor.h>
#include <Eigen/Core>

namespace tmol {
namespace score {
namespace bonded_atom {

#define def auto EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <typename Int, tmol::Device D>
struct IndexedBonds {
  TensorAccessor<Vec<Int, 2>, 1, D> bonds;
  TensorAccessor<Vec<Int, 2>, 1, D> bond_spans;

  struct BondJIter {
    Int bidx;
    IndexedBonds<Int, D> const& parent;

    def operator++()->BondJIter& {
      bidx++;
      return *this;
    }

    def operator==(BondJIter& other) const->bool { return bidx == other.bidx; }
    def operator!=(BondJIter& other) const->bool { return !(*this == other); }
    def operator*() const->Int& { return parent.bonds[bidx][1]; }
  };

  struct BoundAtomRange {
    Int i;
    IndexedBonds<Int, D> const& parent;

    def begin()->BondJIter {
      return BondJIter{parent.bond_spans[i][0], parent};
    }

    def end()->BondJIter { return BondJIter{parent.bond_spans[i][1], parent}; }
  };

  EIGEN_DEVICE_FUNC BoundAtomRange bound_to(Int i) const {
    return BoundAtomRange{i, *this};
  }
};

#undef def

}  // namespace bonded_atom
}  // namespace score
}  // namespace tmol
