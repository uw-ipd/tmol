#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/count_pair.hh>
#include <tmol/score/common/geom.hh>
#include <tmol/score/common/tuple.hh>

#include <tmol/score/ljlk/potentials/lj.hh>
#include <tmol/score/ljlk/potentials/rotamer_pair_energy_lj.hh>

#include <chrono>

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto LJRPEDispatch<DeviceDispatch, D, Real, Int>::f(
    TView<Vec<Real, 3>, 3, D> context_coords,
    TView<Int, 2, D> context_block_type,
    TView<Vec<Real, 3>, 2, D> alternate_coords,
    TView<Vec<Int, 3>, 1, D>
        alternate_ids,  // 0 == context id; 1 == block id; 2 == block type

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
    // max-n-interblock-connections x max-n-interblock-connections
    TView<Int, 5, D> system_inter_block_bondsep,

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
    TView<LJTypeParams<Real>, 1, D> type_params,
    TView<LJGlobalParams<Real>, 1, D> global_params,
    TView<Real, 1, D> lj_lk_weights) -> TPack<Real, 1, D> {
  int const n_systems = system_min_bond_separation.size(0);
  int const n_contexts = context_coords.size(0);
  int64_t const n_alternate_blocks = alternate_coords.size(0);
  int const max_n_blocks = context_coords.size(1);
  int64_t const max_n_atoms = context_coords.size(2);
  int const n_block_types = block_type_n_atoms.size(0);
  int const max_n_interblock_bonds =
      block_type_atoms_forming_chemical_bonds.size(1);
  int64_t const max_n_neighbors = system_neighbor_list.size(2);

  assert(alternate_coords.size(1) == max_n_atoms);
  assert(alternate_ids.size(0) == n_alternate_blocks);
  assert(context_coords.size(0) == context_block_type.size(0));
  assert(context_system_ids.size(0) == n_contexts);

  assert(system_min_bond_separation.size(1) == max_n_blocks);
  assert(system_min_bond_separation.size(2) == max_n_blocks);

  assert(system_inter_block_bondsep.size(0) == n_systems);
  assert(system_inter_block_bondsep.size(1) == max_n_blocks);
  assert(system_inter_block_bondsep.size(2) == max_n_blocks);
  assert(system_inter_block_bondsep.size(3) == max_n_interblock_bonds);
  assert(system_inter_block_bondsep.size(4) == max_n_interblock_bonds);
  assert(system_neighbor_list.size(0) == n_systems);
  assert(system_neighbor_list.size(1) == max_n_blocks);

  assert(block_type_atom_types.size(0) == n_block_types);
  assert(block_type_atom_types.size(1) == max_n_atoms);
  assert(block_type_n_interblock_bonds.size(0) == n_block_types);
  assert(block_type_atoms_forming_chemical_bonds.size(0) == n_block_types);
  assert(block_type_path_distance.size(0) == n_block_types);
  assert(block_type_path_distance.size(1) == max_n_atoms);
  assert(block_type_path_distance.size(2) == max_n_atoms);

  assert(lj_lk_weights.size(0) == 2);

  // auto wcts = std::chrono::system_clock::now();
  // clock_t start_time = clock();

  auto output_t = TPack<Real, 1, D>::zeros({n_alternate_blocks});
  auto output = output_t.view;
  auto count_t = TPack<int, 1, D>::zeros({1});
  auto count = count_t.view;

  auto eval_atom_pair = ([=] EIGEN_DEVICE_FUNC(
                             int alt_ind, int neighb_ind, int atom_pair_ind) {
    int const max_important_bond_separation = 4;
    int const alt_context = alternate_ids[alt_ind][0];
    if (alt_context == -1) {
      return;
    }

    int const alt_block_ind = alternate_ids[alt_ind][1];
    int const alt_block_type = alternate_ids[alt_ind][2];
    int const system = context_system_ids[alt_context];

    int const neighb_block_ind =
        system_neighbor_list[system][alt_block_ind][neighb_ind];
    if (neighb_block_ind == -1) {
      return;
    }

    int atom_1_type = -1;
    int atom_2_type = -1;
    int separation = max_important_bond_separation + 1;
    Real dist(-1);

    Vec<Real, 3> coord1, coord2;
    int at1, at2;

    if (alt_block_ind != neighb_block_ind) {
      // Inter-block interaction. One atom from "alt", one atom from
      // "context."

      int const neighb_block_type =
          context_block_type[alt_context][neighb_block_ind];
      int const alt_n_atoms = block_type_n_atoms[alt_block_type];
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
      separation =
          common::count_pair::CountPair<D, Int>::inter_block_separation(
              max_important_bond_separation,
              alt_block_ind,
              neighb_block_ind,
              alt_block_type,
              neighb_block_type,
              alt_atom_ind,
              neighb_atom_ind,
              system_min_bond_separation[system],
              system_inter_block_bondsep[system],
              block_type_n_interblock_bonds,
              block_type_atoms_forming_chemical_bonds,
              block_type_path_distance);

      at1 = alt_atom_ind;
      at2 = neighb_atom_ind;
      coord1 = alternate_coords[alt_ind][alt_atom_ind];
      coord2 = context_coords[alt_context][neighb_block_ind][neighb_atom_ind];

      dist = distance<Real>::V(
          context_coords[alt_context][neighb_block_ind][neighb_atom_ind],
          alternate_coords[alt_ind][alt_atom_ind]);
      atom_1_type = block_type_atom_types[alt_block_type][alt_atom_ind];
      atom_2_type = block_type_atom_types[neighb_block_type][neighb_atom_ind];
    } else {
      // alt_block_ind == neighb_block_ind:
      // intra-block interaction.
      int const alt_n_atoms = block_type_n_atoms[alt_block_type];
      // see comment in the inter-block interaction regarding the delay of
      // the atom1/atom2 resolution until we know how many atoms are in the
      // particular block we're looking at.
      if (atom_pair_ind >= alt_n_atoms * alt_n_atoms) {
        return;
      }
      int const atom_1_ind = atom_pair_ind / alt_n_atoms;
      int const atom_2_ind = atom_pair_ind % alt_n_atoms;
      at1 = atom_1_ind;
      at2 = atom_2_ind;
      coord1 = alternate_coords[alt_ind][atom_1_ind];
      coord2 = alternate_coords[alt_ind][atom_2_ind];
      if (atom_1_ind >= atom_2_ind) {
        // count each intra-block interaction only once
        return;
      }
      dist = distance<Real>::V(
          alternate_coords[alt_ind][atom_1_ind],
          alternate_coords[alt_ind][atom_2_ind]);
      separation =
          block_type_path_distance[alt_block_type][atom_1_ind][atom_2_ind];
      atom_1_type = block_type_atom_types[alt_block_type][atom_1_ind];
      atom_2_type = block_type_atom_types[alt_block_type][atom_2_ind];
    }

    // printf(
    //     "%d %d (%d) %d %d %d %d\n",
    //     alt_ind,
    //     neighb_ind,
    //     neighb_block_ind,
    //     atom_pair_ind,
    //     separation,
    //     atom_1_type,
    //     atom_2_type);
    Real lj = lj_score<Real>::V(
        dist,
        separation,
        type_params[atom_1_type],
        type_params[atom_2_type],
        global_params[0]);
    lj *= lj_lk_weights[0];

    // if ( lj != 0 ) {
    //   printf("cpu  %d %d %6.3f %6.3f %6.3f vs %6.3f %6.3f %6.3f e= %8.4f\n",
    //     at1, at2,
    //     coord1[0],
    //     coord1[1],
    //     coord1[2],
    //     coord2[0],
    //     coord2[1],
    //     coord2[2],
    //     lj
    //   );
    // }

    accumulate<D, Real>::add_one_dst(output, alt_ind, lj);
    // accumulate<D, Real>::add(output[alt_ind], lj);
  });

  DeviceDispatch<D>::foreach_combination_triple(
      n_alternate_blocks,
      max_n_neighbors,
      max_n_atoms * max_n_atoms,
      eval_atom_pair);

#ifdef __CUDACC__
  // float first;
  // cudaMemcpy(&first, &output[0], sizeof(float), cudaMemcpyDeviceToHost);
  //
  // clock_t stop_time = clock();
  // std::chrono::duration<double> wctduration =
  // (std::chrono::system_clock::now() - wcts);
  //
  // std::cout << n_systems << " " << n_contexts << " " <<n_alternate_blocks <<
  // " "; std::cout << n_alternate_blocks * max_n_neighbors * max_n_atoms *
  // max_n_atoms << " "; std::cout << "runtime? " << ((double)stop_time -
  // start_time) / CLOCKS_PER_SEC
  //           << " wall time: " << wctduration.count() << " " << first
  //           << std::endl;
#endif
  return output_t;
}

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
