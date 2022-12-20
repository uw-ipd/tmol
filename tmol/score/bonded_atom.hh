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

template <typename Int>
struct BlockCentricAtom {
  Int block;
  Int block_type;
  Int atom;
};

template <typename Int, tmol::Device D>
struct BlockCentricIndexedBonds {
  // inter_block_connections and block_type have already been sliced
  // to focus on a particular Pose in the PoseStack
  TensorAccessor<Vec<Int, 2>, 2, D> inter_block_connections;
  TensorAccessor<Int, 1, D> block_type;

  // Properties of the block types in this molecular system
  TView<Int, 1, D> block_type_n_all_bonds;
  TView<Vec<Int, 3>, 2, D> block_type_all_bonds;
  TView<Vec<Int, 2>, 2, D> block_type_atom_all_bond_ranges;
  TView<Int, 2, D> block_type_atoms_forming_chemical_bonds;

  struct BondJIter {
    Int block;
    Int block_type;
    Int bidx;
    BlockCentricIndexedBonds<Int, D> const& parent;

    def operator++()->BondJIter& {
      bidx++;
      return *this;
    }
    def operator==(BondJIter& other) const->bool {
      return bidx == other.bidx && block == other.block
             && block_type == other.block_type;
    }
    def operator!=(BondJIter& other) const->bool { return !(*this == other); }
    def operator*() const->BlockCentricAtom<Int> {
      Int neighb_atm = parent.block_type_all_bonds[block_type][bidx][1];
      if (neighb_atm >= 0) {
        return {block, block_type, neighb_atm};
      } else {
        // Inter-block chemical bond:
        // The neighbor atom is the one on the other side of the connection
        // on this residue. 1. Look up the connection on this residue, conn_id.
        // 2. Look up for this structure the connection id on the other
        // block to which conn_id is connected, nbr_conn_id. 3. Look up
        // for this struture the block id for the other block to which
        // conn_id is connected, nbr_block. 4. Look up the block type of the
        // neighbor block. 5. Look up the atom index for the connection atom
        // on the neighbor, nbr_conn_atom.
        Int conn_id = parent.block_type_all_bonds[block_type][bidx][2];
        Int nbr_conn_id = parent.inter_block_connections[block][conn_id][1];
        Int nbr_block = parent.inter_block_connections[block][conn_id][0];
        Int nbr_block_type = parent.block_type[nbr_block];
        Int nbr_conn_atom =
            parent.block_type_atoms_forming_chemical_bonds[nbr_block_type]
                                                          [nbr_conn_id];
        return {nbr_block, nbr_block_type, nbr_conn_atom};
      }
    }
  };

  struct BondAtomRange {
    BlockCentricAtom<Int> bcat;
    BlockCentricIndexedBonds<Int, D> const& parent;

    def begin()->BondJIter {
      return BondJIter{
          bcat.block,
          bcat.block_type,
          parent.block_type_atom_all_bond_ranges[bcat.block_type][bcat.atom][0],
          parent};
    }
    def end()->BondJIter {
      return BondJIter{
          bcat.block,
          bcat.block_type,
          parent.block_type_atom_all_bond_ranges[bcat.block_type][bcat.atom][1],
          parent};
    }
  };

  EIGEN_DEVICE_FUNC BondAtomRange bound_to(BlockCentricAtom<Int> bcat) const {
    return BondAtomRange{bcat, *this};
  }
};

#undef def

}  // namespace bonded_atom
}  // namespace score
}  // namespace tmol
