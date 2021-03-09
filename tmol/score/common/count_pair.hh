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

#include <tmol/score/ljlk/potentials/lj.hh>
#include <tmol/score/ljlk/potentials/rotamer_pair_energy_lj.hh>

namespace tmol {
namespace score {
namespace common {
namespace count_pair {

template <tmol::Device D, typename Int>
struct CountPair {
  static EIGEN_DEVICE_FUNC int inter_block_separation(
      int const max_important_bond_separation,
      int alt_block_ind,
      int neighb_block_ind,
      int alt_block_type,
      int neighb_block_type,
      int alt_atom_ind,
      int neighb_atom_ind,
      TensorAccessor<Int, 2, D> min_bond_separation,
      TensorAccessor<Int, 4, D> inter_block_bondsep,
      TView<Int, 1, D> block_type_n_interblock_bonds,
      TView<Int, 2, D> block_type_atoms_forming_chemical_bonds,
      TView<Int, 3, D> block_type_path_distance) {
    int separation = min_bond_separation[alt_block_ind][neighb_block_ind];
    if (separation <= max_important_bond_separation) {
      separation = max_important_bond_separation + 1;
      int const alt_n_interres_bonds =
          block_type_n_interblock_bonds[alt_block_type];
      int const neighb_n_interres_bonds =
          block_type_n_interblock_bonds[neighb_block_type];
      for (int ii = 0; ii < alt_n_interres_bonds; ++ii) {
        int const ii_alt_conn_atom =
            block_type_atoms_forming_chemical_bonds[alt_block_type][ii];
        int const ii_alt_bonds_to_conn =
            block_type_path_distance[alt_block_type][ii_alt_conn_atom]
                                    [alt_atom_ind];
        // if (ii_alt_bonds_to_conn >= separation) {
        //   continue;
        // }
        for (int jj = 0; jj < neighb_n_interres_bonds; ++jj) {
          int ii_jj_interblock_bond_sep =
              inter_block_bondsep[alt_block_ind][neighb_block_ind][ii][jj];
          // if (ii_jj_interblock_bond_sep >= separation) {
          //   continue;
          // }

          // if (ii_alt_bonds_to_conn + ii_jj_interblock_bond_sep >= separation)
          // {
          //   continue;
          // }
          int const jj_neighb_conn_atom =
              block_type_atoms_forming_chemical_bonds[neighb_block_type][jj];
          int const jj_neighb_bonds_to_conn =
              block_type_path_distance[neighb_block_type][jj_neighb_conn_atom]
                                      [neighb_atom_ind];

          if (ii_alt_bonds_to_conn + ii_jj_interblock_bond_sep
                  + jj_neighb_bonds_to_conn
              < separation) {
            separation = ii_alt_bonds_to_conn + ii_jj_interblock_bond_sep
                         + jj_neighb_bonds_to_conn;
          }
        }
      }
    }
    return separation;
  }

  static EIGEN_DEVICE_FUNC int inter_block_separation(
      int const max_important_bond_separation,
      int alt_block_ind,
      int neighb_block_ind,
      int alt_block_type,
      int neighb_block_type,
      int alt_atom_ind,
      int neighb_atom_ind,
      TensorAccessor<Int, 4, D> inter_block_bondsep,
      TView<Int, 1, D> block_type_n_interblock_bonds,
      TView<Int, 2, D> block_type_atoms_forming_chemical_bonds,
      TView<Int, 3, D> block_type_path_distance) {
    int separation = max_important_bond_separation + 1;
    int const alt_n_interres_bonds =
        block_type_n_interblock_bonds[alt_block_type];
    int const neighb_n_interres_bonds =
        block_type_n_interblock_bonds[neighb_block_type];

    // if ( alt_block_ind + 1 == neighb_block_ind ) {
    //   int const ii_alt_conn_atom =
    // 	block_type_atoms_forming_chemical_bonds[alt_block_type][1];
    //   int const jj_neighb_conn_atom =
    // 	block_type_atoms_forming_chemical_bonds[neighb_block_type][0];
    //   return (
    // 	block_type_path_distance[alt_block_type][ii_alt_conn_atom][alt_atom_ind]
    // +
    // 	block_type_path_distance[neighb_block_type][jj_neighb_conn_atom][neighb_atom_ind]
    // + 	inter_block_bondsep[alt_block_ind][neighb_block_ind][1][0]
    //   );
    // } else if ( alt_block_ind == neighb_block_ind + 1 ) {
    //   int const ii_alt_conn_atom =
    // 	block_type_atoms_forming_chemical_bonds[alt_block_type][0];
    //   int const jj_neighb_conn_atom =
    // 	block_type_atoms_forming_chemical_bonds[neighb_block_type][1];
    //   return (
    // 	block_type_path_distance[alt_block_type][ii_alt_conn_atom][alt_atom_ind]
    // +
    // 	block_type_path_distance[neighb_block_type][jj_neighb_conn_atom][neighb_atom_ind]
    // + 	inter_block_bondsep[alt_block_ind][neighb_block_ind][0][1]
    //   );
    // } else {

    for (int ii = 0; ii < alt_n_interres_bonds; ++ii) {
      int const ii_alt_conn_atom =
          block_type_atoms_forming_chemical_bonds[alt_block_type][ii];
      int const ii_alt_bonds_to_conn =
          block_type_path_distance[alt_block_type][ii_alt_conn_atom]
                                  [alt_atom_ind];
      // if (ii_alt_bonds_to_conn >= separation) {
      //   continue;
      // }
      for (int jj = 0; jj < neighb_n_interres_bonds; ++jj) {
        int ii_jj_interblock_bond_sep =
            inter_block_bondsep[alt_block_ind][neighb_block_ind][ii][jj];
        // if (ii_jj_interblock_bond_sep >= separation) {
        //   continue;
        // }

        if (ii_alt_bonds_to_conn + ii_jj_interblock_bond_sep >= separation) {
          continue;
        }
        int const jj_neighb_conn_atom =
            block_type_atoms_forming_chemical_bonds[neighb_block_type][jj];
        int const jj_neighb_bonds_to_conn =
            block_type_path_distance[neighb_block_type][jj_neighb_conn_atom]
                                    [neighb_atom_ind];

        int const ii_jj_sep = ii_alt_bonds_to_conn + ii_jj_interblock_bond_sep
                              + jj_neighb_bonds_to_conn;
        if (ii_jj_sep < separation) {
          separation = ii_jj_sep;
        }
      }
    }
    return separation;
  }
  //}

  // Templated on the number of atoms in the tile
  template <int N, class T>
  static EIGEN_DEVICE_FUNC int inter_block_separation(
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
};

}  // namespace count_pair
}  // namespace common
}  // namespace score
}  // namespace tmol
