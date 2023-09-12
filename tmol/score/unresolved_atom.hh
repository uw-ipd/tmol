#pragma once

namespace tmol {

//  unresolved atom ids are three integers:
//   1st integer: an atom index, -1 if the atom is unresolved
//   2nd integer: the connection index for this block that the unresolved atom
//                is on ther other side of, -1 if
//   3rd integer: the number of chemical bonds into the other block that the
//                unresolved atom is found.
// E.g
// -- the tuple A: (0, -1, -1) would represent the first atom for a particular
//    block type. For an Alanine, this would be the N atom.
// -- the tuple B: (-1, 1, 0) would represent the atom on the other side of
//    the inter-block bond at connection point 1; if this were alanine
//    then the C atom is the "upper connect" atom, (upper connect being
//    connection #1), and if the next residue were another amino acid,
//    then tuple B would resolve to the next residue's N atom.
// -- the tuple C: (-1, 1, 1) would represent one bond deep into the
//    next residue on the other side of the inter-block bond at connection
//    point 1. If this were alanine, then the C atom is the "upper connect"
//    atom, and if the next residue were another (alpha) amino acid, then
//    tuple C would resolve to the next residue's CA atom.
template <typename Int>
struct UnresolvedAtomID {
  Int atom_id;
  Int conn_id;
  Int n_bonds_from_conn;
};

template <typename Int>
struct enable_tensor_view<UnresolvedAtomID<Int>> {
  static const bool enabled = enable_tensor_view<Int>::enabled;
  static const at::ScalarType scalar_type() {
    return enable_tensor_view<Int>::scalar_type();
  }

  static const int nconsumed_dims = 1;
  static const int consumed_dims(int i) {
    return (i == 0) ? sizeof(UnresolvedAtomID<Int>) / sizeof(Int) : 0;
  }

  typedef typename enable_tensor_view<Int>::PrimitiveType PrimitiveType;
};

}  // namespace tmol
