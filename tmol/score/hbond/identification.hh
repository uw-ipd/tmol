#pragma once

#include <tmol/utility/tensor/TensorAccessor.h>
#include <Eigen/Core>
#include <tmol/score/bonded_atom.hh>

#undef B0

namespace tmol {
namespace score {
namespace hbond {

using tmol::score::bonded_atom::IndexedBonds;

#define def auto EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE

struct AcceptorHybridization {
  static constexpr int none = 0;
  static constexpr int sp2 = 1;
  static constexpr int sp3 = 2;
  static constexpr int ring = 3;
};

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <typename Int>
struct AcceptorBases {
  Int A;
  Int B;
  Int B0;

  template <typename func_t, tmol::Device D>
  static def sp2_acceptor_base(
      Int A,
      Int hybridization,
      IndexedBonds<Int, D> bonds,
      func_t atom_is_hydrogen)
      ->AcceptorBases {
    Int B = -1, B0 = -1;

    for (Int other_atom : bonds.bound_to(A)) {
      if (!atom_is_hydrogen(other_atom)) {
        B = other_atom;
        break;
      }
    }

    if (B == -1) {
      return {A, -1, -1};
    }

    for (Int other_atom : bonds.bound_to(B)) {
      if (other_atom != A) {
        B0 = other_atom;
        break;
      }
    }

    if (B0 == -1) {
      return {A, -1, -1};
    }

    return {A, B, B0};
  }

  template <typename func_t, tmol::Device D>
  static def sp3_acceptor_base(
      Int A,
      Int hybridization,
      IndexedBonds<Int, D> bonds,
      func_t atom_is_hydrogen)
      ->AcceptorBases {
    Int B = -1;
    Int B0 = -1;

    for (Int other_atom : bonds.bound_to(A)) {
      if (atom_is_hydrogen(other_atom)) {
        B0 = other_atom;
        break;
      }
    }

    if (B0 == -1) {
      return {A, -1, -1};
    }

    for (Int other_atom : bonds.bound_to(A)) {
      if (other_atom != B0) {
        B = other_atom;
        break;
      }
    }

    if (B0 == -1) {
      return {A, -1, -1};
    }

    return {A, B, B0};
  }

  template <typename func_t, tmol::Device D>
  static def ring_acceptor_base(
      Int A,
      Int hybridization,
      IndexedBonds<Int, D> bonds,
      func_t atom_is_hydrogen)
      ->AcceptorBases {
    Int B = -1;
    Int B0 = -1;

    for (Int other_atom : bonds.bound_to(A)) {
      if (!atom_is_hydrogen(other_atom)) {
        B = other_atom;
        break;
      }
    }

    if (B == -1) {
      return {A, -1, -1};
    }

    for (Int other_atom : bonds.bound_to(A)) {
      if (other_atom != B && !atom_is_hydrogen(other_atom)) {
        B0 = other_atom;
        break;
      }
    }

    if (B0 == -1) {
      return {A, -1, -1};
    }

    return {A, B, B0};
  }

  template <typename func_t, tmol::Device D>
  static def for_acceptor(
      Int A,
      Int hybridization,
      IndexedBonds<Int, D> bonds,
      func_t atom_is_hydrogen)
      ->AcceptorBases {
    if (hybridization == AcceptorHybridization::sp2) {
      return sp2_acceptor_base(A, hybridization, bonds, atom_is_hydrogen);
    } else if (hybridization == AcceptorHybridization::sp3) {
      return sp3_acceptor_base(A, hybridization, bonds, atom_is_hydrogen);
    } else if (hybridization == AcceptorHybridization::ring) {
      return ring_acceptor_base(A, hybridization, bonds, atom_is_hydrogen);
    } else {
      return {A, -1, -1};
    }
  }
};

#undef def

}  // namespace hbond
}  // namespace score
}  // namespace tmol
