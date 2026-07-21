#pragma once

#include <tmol/utility/tensor/TensorAccessor.h>
#include <Eigen/Core>
#include <tmol/score/bonded_atom.hh>

#undef B0

namespace tmol {
namespace score {
namespace hbond {

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
struct RotamerCentricAcceptorBases {
  bonded_atom::BlockCentricAtom<Int> A;
  bonded_atom::BlockCentricAtom<Int> B;
  bonded_atom::BlockCentricAtom<Int> B0;

  template <tmol::Device Dev>
  static def sp2_acceptor_base(
      bonded_atom::BlockCentricAtom<Int> A,
      bonded_atom::RotamerCentricIndexedBonds<Int, Dev> bonds,
      TView<Int, 2, Dev> bt_atom_is_hydrogen) -> RotamerCentricAcceptorBases {
    using bonded_atom::BlockCentricAtom;
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
      bonded_atom::BlockCentricAtom<Int> A,
      bonded_atom::RotamerCentricIndexedBonds<Int, Dev> bonds,
      TView<Int, 2, Dev> bt_atom_is_hydrogen) -> RotamerCentricAcceptorBases {
    using bonded_atom::BlockCentricAtom;
    BlockCentricAtom<Int> B{-1, -1, -1};
    BlockCentricAtom<Int> B0{-1, -1, -1};

    // B is the primary (heavy-atom) base that A is bonded to.
    for (BlockCentricAtom<Int> other_atom : bonds.bound_to(A)) {
      if (!bt_atom_is_hydrogen[other_atom.block_type][other_atom.atom]) {
        B = other_atom;
        break;
      }
    }

    // B0: a bonded hydrogen, else any other bonded atom, else B itself.
    for (BlockCentricAtom<Int> other_atom : bonds.bound_to(A)) {
      if (other_atom != B
          && bt_atom_is_hydrogen[other_atom.block_type][other_atom.atom]) {
        B0 = other_atom;
        break;
      }
    }
    if (B0.atom == -1) {
      for (BlockCentricAtom<Int> other_atom : bonds.bound_to(A)) {
        if (other_atom != B) {
          B0 = other_atom;
          break;
        }
      }
    }
    if (B0.atom == -1) {
      B0 = B;
    }

    // B0 == B signals no distinct second base: lk_ball skips, hbond uses B0=B.
    return {A, B, B0};
  }

  template <tmol::Device Dev>
  static def ring_acceptor_base(
      bonded_atom::BlockCentricAtom<Int> A,
      bonded_atom::RotamerCentricIndexedBonds<Int, Dev> bonds,
      TView<Int, 2, Dev> bt_atom_is_hydrogen) -> RotamerCentricAcceptorBases {
    using bonded_atom::BlockCentricAtom;
    BlockCentricAtom<Int> B{-1, -1, -1};
    BlockCentricAtom<Int> B0{-1, -1, -1};

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
      bonded_atom::BlockCentricAtom<Int> A,
      Int hybridization,
      bonded_atom::RotamerCentricIndexedBonds<Int, Dev> bonds,
      TView<Int, 2, Dev> bt_atom_is_hydrogen) -> RotamerCentricAcceptorBases {
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
struct RotamerCentricDonorBase {
  bonded_atom::BlockCentricAtom<Int> H;
  bonded_atom::BlockCentricAtom<Int> D;

  template <tmol::Device Dev>
  static def for_polar_H(
      bonded_atom::BlockCentricAtom<Int> H,
      bonded_atom::RotamerCentricIndexedBonds<Int, Dev> bonds,
      TView<Int, 2, Dev> bt_atom_is_hydrogen) -> RotamerCentricDonorBase<Int> {
    using bonded_atom::BlockCentricAtom;

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
