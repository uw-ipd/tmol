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

#include "lj.hh"
#include "rotamer_pair_energy_lj.hh"

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <
    template <tmol::Device>
    class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto LJRPEDispatch<Dispatch D, Real, Int>::f(
    TView<Vec<Real, 3>, 3, D> context_coords,
    TView<Int, 2, D> context_block_type,
    TView<Vec<Real, 3>, 2, D> alternate_coords,
    TView<Vec<Int, 3>, 1, D>
        alternate_ids,  // 0 == context id; 2 == block id; 3 == block type

    // which system does a given context belong to
    TView<Int, 1, D> context_system_ids,

    // dims: n-systems x max-n-blocks x max-n-blocks
    // Quick lookup: given the inds of two blocks, ask: what is the minimum
    // number of chemical bonds that separate any pair of atoms in those blocks?
    // If this minimum is greater than the crossover, then no further logic for
    // deciding whether two atoms in those blocks should have their interaction
    // energies calculated: all should. intentionally small to (possibly) fit in
    // constant cache
    TView<Int, 3, D> system_min_bond_separation,

    // dims: n-systems x max-n-blocks x max-n-blocks x
    // max-n-interblock-connections vec inds:
    //    0: which inter-block connection point is bonded; -1 sentinel
    //    1: how many bonds separate the two blocks along this connection
    TView<Vec<Int, 2>, 4, D> system_interblock_bonds,

    // dims n-systems x max-n-blocks x max-n-neighbors
    // -1 as the sentinel
    TView<Int, 3, D> system_neighbor_list,

    //////////////////////
    // Chemical properties
    // how many atoms for a given block
    // Dimsize n_block_types
    TView<Int, 1, D> block_type_n_atoms,

    // what are the atom types for these atoms
    // Dimsize: n_block_types x max_n_atoms
    TView<Int, 2, D> block_type_atom_types,

    // how many inter-block chemical bonds are there
    // Dimsize: n_block_types
    TView<Int, 1, D> block_type_n_interblock_bonds,

    // what atoms form the inter-block chemical bonds
    // Dimsize: n_block_types x max_n_interblock_bonds
    TView<Int, 2, D> block_type_atoms_forming_chemical_bonds,

    // what is the path distance between pairs of atoms in the block
    // Dimsize: n_block_types x max_n_atoms x max_n_atoms
    TView<Int, 3, D> block_type_path_distance,
    //////////////////////

    // LJ parameters
    TView<LJParams<Real>, 1, D> type_params,
    TView<LJGlobalParams<Real>, 1, D> global_params) -> TPack<Real, 1, D> {
  int const n_systems = system_min_bond_separation.size(0);
  int const n_contexts = context_coords.size(0);
  int const n_alternate_blocks = alternate_coords.size(0);
  int const max_n_blocks = context_coords.size(1);
  int const max_n_atoms = context_coords.size(2);
  int const n_block_types = block_type_n_atoms.size(0);
  int const max_n_interblock_bonds =
      block_type_atoms_forming_chemical_bonds.size(1);
  int const max_n_neighbors = system_neighbor_list.size(2);

  assert(alternate_coords.size(1) == max_n_atoms);
  assert(alternate_ids.size(0) == n_alternate_blocks);
  assert(context_coords.size(0) == context_block_type.size(0));
  assert(context_system_ids.size(0) == n_contexts);

  assert(system_min_bond_separation.size(1) == max_n_blocks);
  assert(system_min_bond_separation.size(2) == max_n_blocks);

  assert(system_interblock_bonds.size(0) == n_systems);
  assert(system_interblock_bonds.size(1) == max_n_blocks);
  assert(system_interblock_bonds.size(2) == max_n_blocks);
  assert(system_interblock_bonds.size(3) == max_n_interblock_bonds);
  assert(system_neighbor_list.size(0) == n_systems);
  assert(system_neighbor_list.size(1) == max_n_blocks);

  assert(block_type_atom_types.size(0) == n_block_types);
  assert(block_type_atom_types.size(1) == max_n_atoms);
  assert(block_type_n_interblock_bonds.size(0) == n_block_types);
  assert(block_type_atoms_forming_chemical_bonds.size(0) == n_block_types);
  assert(block_type_path_distance.size(0) == n_block_types);
  assert(block_type_path_distance.size(1) == max_n_atoms);
  assert(block_type_path_distance.size(2) == max_n_atoms);

  auto output_t = TPack<Real, 1, D>::zeros({n_alternate_blocks});
  auto output = output_t.view;

  auto eval_atom_pair = [=] EIGEN_DEVICE_FUN(
                            int alt_ind, int neighbor_ind, int atom_pair_ind) {
    int const alt_context = alternate_ids[alt_ind][0];
    if (alt_context == -1) {
      return;
    }

    int const alt_block_ind = alternate_ids[alt_ind][1];
    int const alt_block_type = alternate_ids[alt_ind][2];
    int const system = context_system_ids[alt_context];

    int const neighb_block_ind =
        system_neighbor_list[system][alt_block_ind][neighbor_ind];
    if (neighb_block_ind == -1) {
      return;
    }

    int const neighb_block_type =
        context_block_type[alt_context][neighbor_block_ind];
    int const alt_n_atoms = block_type_natoms[alt_block_type];
    int const neighb_n_atoms = block_type_n_atoms[neighb_block_type];

    // for best warp cohesion, mod the atom-pair indices after
    // we have figured out the number of atoms in both blocks;
    // if we modded *before* based on the maximum number of atoms
    // per block, lots of warps with inactive atom-pairs
    // (because they are off the end of the list) would run.
    // By waiting to figure out the i/j inds until after we know
    // how many atoms pairs there will be, we can push all the inactive
    // threads into the same warps and kill those warps early
    if (atom_pair_ind >= alt_n_atoms * neighb_n_atoms) {
      return;
    }

    int alt_atom_ind = atom_pair_ind / neighb_n_atoms;
    int neighb_atom_ind = atom_pair_ind % neighb_n_atoms;

    // "count pair" logic
    float count_pair_weight = 1.0;
    int separation =
        system_min_bond_separation[system][alt_block_ind][neighb_block_ind];
    if (separation <= 4) {
      separation = 6;
      int const alt_n_interres_bonds =
          block_type_n_interblock_bonds[alt_block_type];

      for (int ii = 0; ii < ant_n_interres_bonds; ++ii) {
        int ii_interblock_bond_sep =
            system_interblock_bonds[system][alt_block_ind][neighb_block_ind][1];
        if (ii_interblock_bond_sep == -1) {
          continue;
        }
        if (ii_interblock_bond_sep >= separation) {
          continue;
        }

        int const alt_conn_atom =
            block_type_atoms_forming_chemical_bonds[alt_block_type][ii];
        int const alt_bonds_to_conn =
            block_type_path_distance[alt_block_type][alt_conn_atom]
                                    [alt_atom_ind];
        if (alt_bonds_to_conn + ii_interblock_bond_sep >= separation) {
          continue;
        }
        int const neighb_conn_port =
            system_interblock_bonds[system][alt_block_ind][neighb_block_ind][0];
        int const neighb_conn_atom =
            block_type_atoms_forming_chemical_bonds[neighb_block_type]
                                                   [neighb_conn_port];
        int const neighb_bonds_to_conn =
            block_type_path_distance[neighb_block_type][neighb_conn_atom]
                                    [neighb_atom_ind];

        if (alt_bonds_to_conn + ii_interblock_bond_sep + neighb_bonds_to_conn
            < separation) {
          separation =
              alt_bonds_to_conn + ii_interblock_bond_sep + neighb_bonds_to_conn;
        }
      }
    }
    Real dist = distance<Real>::V(
        context_coords[alt_context][neighb_block_ind][neighb_atom_ind],
        alternate_coords[alt_ind][alt_atom_ind]);
    Real lj = lj_score<Real>::V(
        dist,
        separation,
        type_params[block_type_atom_types[alt_block_type][alt_atom_ind]],
        type_params[block_type_atom_types[neighb_block_type][neighb_atom_ind]],
        global_params[0]);

    accumulate<D, Real>::add(output[alt_block_ind], lj);
  };

  Dispatch<D>::foreach_combination(
      n_alternate_blocks,
      max_n_neighbors,
      max_n_atoms * max_n_atoms,
      eval_atom_pair);
}
