#pragma once

#include <tmol/utility/tensor/TensorAccessor.h>
#include <Eigen/Core>
#include <tmol/score/bonded_atom.hh>

#undef B0

namespace tmol {
namespace score {
namespace hbond {

using tmol::score::bonded_atom::BlockCentricAtom;
using tmol::score::bonded_atom::BlockCentricIndexedBonds;
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

  template <typename func_t, tmol::Device Dev>
  static def sp2_acceptor_base(
      Int stack,
      Int A,
      Int hybridization,
      IndexedBonds<Int, Dev> bonds,
      func_t atom_is_hydrogen)
      ->AcceptorBases {
    Int B = -1, B0 = -1;

    for (Int other_atom : bonds.bound_to(A)) {
      if (!atom_is_hydrogen(stack, other_atom)) {
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

  template <typename func_t, tmol::Device Dev>
  static def sp3_acceptor_base(
      Int stack,
      Int A,
      Int hybridization,
      IndexedBonds<Int, Dev> bonds,
      func_t atom_is_hydrogen)
      ->AcceptorBases {
    Int B = -1;
    Int B0 = -1;

    for (Int other_atom : bonds.bound_to(A)) {
      if (atom_is_hydrogen(stack, other_atom)) {
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

  template <typename func_t, tmol::Device Dev>
  static def ring_acceptor_base(
      Int stack,
      Int A,
      Int hybridization,
      IndexedBonds<Int, Dev> bonds,
      func_t atom_is_hydrogen)
      ->AcceptorBases {
    Int B = -1;
    Int B0 = -1;

    for (Int other_atom : bonds.bound_to(A)) {
      if (!atom_is_hydrogen(stack, other_atom)) {
        B = other_atom;
        break;
      }
    }

    if (B == -1) {
      return {A, -1, -1};
    }

    for (Int other_atom : bonds.bound_to(A)) {
      if (other_atom != B && !atom_is_hydrogen(stack, other_atom)) {
        B0 = other_atom;
        break;
      }
    }

    if (B0 == -1) {
      return {A, -1, -1};
    }

    return {A, B, B0};
  }

  template <typename func_t, tmol::Device Dev>
  static def for_acceptor(
      Int stack,
      Int A,
      Int hybridization,
      IndexedBonds<Int, Dev> bonds,
      func_t atom_is_hydrogen)
      ->AcceptorBases {
    if (hybridization == AcceptorHybridization::sp2) {
      return sp2_acceptor_base(
          stack, A, hybridization, bonds, atom_is_hydrogen);
    } else if (hybridization == AcceptorHybridization::sp3) {
      return sp3_acceptor_base(
          stack, A, hybridization, bonds, atom_is_hydrogen);
    } else if (hybridization == AcceptorHybridization::ring) {
      return ring_acceptor_base(
          stack, A, hybridization, bonds, atom_is_hydrogen);
    } else {
      return {A, -1, -1};
    }
  }
};

template <typename Int>
struct BlockCentricAcceptorBases {
  BlockCentricAtom<Int> A;
  BlockCentricAtom<Int> B;
  BlockCentricAtom<Int> B0;

  template <tmol::Device Dev>
  static def sp2_acceptor_base(
      BlockCentricAtom<Int> A,
      BlockCentricIndexedBonds<Int, Dev> bonds,
      TView<Int, 2, Dev> bt_atom_is_hydrogen)
      ->BlockCentricAcceptorBases {
    BlockCentricAtom<Int> B({-1, -1, -1});
    BlockCentricAtom<Int> B0({-1, -1, -1});

    for (BlockCentricAtom<Int> other_atom : bonds.bound_to(A)) {
      if (!bt_atom_is_hydrogen[other_atom.block_type][other_atom.atom]) {
        B = other_atom;
        break;
      }
    }

    if (B.atom == -1) {
      return {A, {-1, -1, -1}, {-1, -1, -1}};
    }

    for (BlockCentricAtom<Int> other_atom : bonds.bound_to(B)) {
      // If we have left the starting residue, then
      // skip other_atom if it has walked back to the
      // starting residue. It may very well be that other_atom != A
      // but only because A.block is being redesigned and we
      // are trying to define the hydrogen bond / lk-ball
      // energy for a block type that's not the one on the Pose
      if (B.block != A.block && other_atom.block == A.block) {
        continue;
      }

      if (other_atom != A) {
        B0 = other_atom;
        break;
      }
    }

    if (B0.atom == -1) {
      return {A, {-1, -1, -1}, {-1, -1, -1}};
    }

    return {A, B, B0};
  }

  template <tmol::Device Dev>
  static def sp3_acceptor_base(
      BlockCentricAtom<Int> A,
      BlockCentricIndexedBonds<Int, Dev> bonds,
      TView<Int, 2, Dev> bt_atom_is_hydrogen)
      ->BlockCentricAcceptorBases {
    BlockCentricAtom<Int> B({-1, -1, -1});
    BlockCentricAtom<Int> B0({-1, -1, -1});

    for (BlockCentricAtom<Int> other_atom : bonds.bound_to(A)) {
      if (bt_atom_is_hydrogen[other_atom.block_type][other_atom.atom]) {
        B0 = other_atom;
        break;
      }
    }

    if (B0.atom == -1) {
      return {A, {-1, -1, -1}, {-1, -1, -1}};
    }

    for (BlockCentricAtom<Int> other_atom : bonds.bound_to(A)) {
      if (other_atom != B0) {
        B = other_atom;
        break;
      }
    }

    if (B0.atom == -1) {
      return {A, {-1, -1, -1}, {-1, -1, -1}};
    }

    return {A, B, B0};
  }

  template <tmol::Device Dev>
  static def ring_acceptor_base(
      BlockCentricAtom<Int> A,
      BlockCentricIndexedBonds<Int, Dev> bonds,
      TView<Int, 2, Dev> bt_atom_is_hydrogen)
      ->BlockCentricAcceptorBases {
    BlockCentricAtom<Int> B({-1, -1, -1});
    BlockCentricAtom<Int> B0({-1, -1, -1});

    for (BlockCentricAtom<Int> other_atom : bonds.bound_to(A)) {
      if (!bt_atom_is_hydrogen[other_atom.block_type][other_atom.atom]) {
        B = other_atom;
        break;
      }
    }

    if (B.atom == -1) {
      return {A, {-1, -1, -1}, {-1, -1, -1}};
    }

    for (BlockCentricAtom<Int> other_atom : bonds.bound_to(A)) {
      if (other_atom != B
          && !bt_atom_is_hydrogen[other_atom.block_type][other_atom.atom]) {
        B0 = other_atom;
        break;
      }
    }

    if (B0.atom == -1) {
      return {A, {-1, -1, -1}, {-1, -1, -1}};
    }

    return {A, B, B0};
  }

  template <tmol::Device Dev>
  static def for_acceptor(
      BlockCentricAtom<Int> A,
      Int hybridization,
      BlockCentricIndexedBonds<Int, Dev> bonds,
      TView<Int, 2, Dev> bt_atom_is_hydrogen)
      ->BlockCentricAcceptorBases {
    if (hybridization == AcceptorHybridization::sp2) {
      return sp2_acceptor_base(A, bonds, bt_atom_is_hydrogen);
    } else if (hybridization == AcceptorHybridization::sp3) {
      return sp3_acceptor_base(A, bonds, bt_atom_is_hydrogen);
    } else if (hybridization == AcceptorHybridization::ring) {
      return ring_acceptor_base(A, bonds, bt_atom_is_hydrogen);
    } else {
      return {A, {-1, -1, -1}, {-1, -1, -1}};
    }
  }
};

template <typename Int>
struct BlockCentricDonorBase {
  BlockCentricAtom<Int> H;
  BlockCentricAtom<Int> D;

  template <tmol::Device Dev>
  static def for_polar_H(
      BlockCentricAtom<Int> H,
      BlockCentricIndexedBonds<Int, Dev> bonds,
      TView<Int, 2, Dev> bt_atom_is_hydrogen)
      ->BlockCentricDonorBase<Int> {
    BlockCentricAtom<Int> D{-1, -1, -1};
    for (BlockCentricAtom<Int> other_atom : bonds.bound_to(H)) {
      if (!bt_atom_is_hydrogen[other_atom.block_type][other_atom.atom]) {
        D = other_atom;
      }
    }
    return {H, D};
  }
};

#undef def

}  // namespace hbond
}  // namespace score
}  // namespace tmol
