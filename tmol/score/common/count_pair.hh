#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/geom.hh>
#include <tmol/score/common/tuple.hh>

namespace tmol {
namespace score {
namespace common {
namespace count_pair {

// Slot of block2 in a near-block row, else of an empty slot whose separations
// are the cap (see InterBlockBondsep).
template <typename NearBlockRow>
EIGEN_DEVICE_FUNC int near_block_slot(NearBlockRow near_blocks, int block2) {
  int const last = int(near_blocks.size(0)) - 1;
  int slot = 0;
  while (slot < last && near_blocks[slot][0] != block2
         && near_blocks[slot][0] >= 0) {
    ++slot;
  }
  return slot;
}

// For doing inter-residue count pair entirely in shared memory
// Templated on the number of atoms in the tile and the datatype
// of the integer-typed T that
template <int N, class T>
static EIGEN_DEVICE_FUNC int shared_mem_inter_block_separation(
    int const max_important_bond_separation,
    int atom_ind_1,
    int atom_ind_2,
    int const n_conn1,
    int const n_conn2,
    T const* path_dist1,  // size max_n_conn * N
    T const* path_dist2,  // size max_n_conn * N
    T const* conn_seps) {
  int separation = max_important_bond_separation + 1;

  // if (n_conn1 > 4) {
  //   printf("error n_conn1 %d\n", n_conn1);
  // }
  // if (n_conn2 > 4) {
  //   printf("error n_conn2 %d\n", n_conn2);
  // }

  for (int ii = 0; ii < n_conn1; ++ii) {
    // if (ii * N + atom_ind_1 >= 4 * N) {
    //   printf("errror ii * N + atom_ind_1: %d", ii * N + atom_ind_1);
    // }

    int const ii_alt_bonds_to_conn = path_dist1[ii * N + atom_ind_1];
    for (int jj = 0; jj < n_conn2; ++jj) {
      // if (ii * n_conn2 + jj >= 16) {
      //   printf("errror ii * n_conn2 + jj: %d", ii * n_conn2 + jj);
      // }
      int ii_jj_interblock_bond_sep = conn_seps[ii * n_conn2 + jj];

      // if (jj * N + atom_ind_2 >= 4 * N) {
      //   printf(
      //       "errror jj * N + atom_ind_2: %d", jj * N + atom_ind_2);
      // }

      int const jj_neighb_bonds_to_conn = path_dist2[jj * N + atom_ind_2];
      int const ii_jj_sep = ii_alt_bonds_to_conn + ii_jj_interblock_bond_sep
                            + jj_neighb_bonds_to_conn;
      if (ii_jj_sep < separation) {
        separation = ii_jj_sep;
      }
    }
  }
  return separation;
}

}  // namespace count_pair
}  // namespace common
}  // namespace score
}  // namespace tmol
