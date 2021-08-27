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
#include <tmol/score/ljlk/potentials/lk_isotropic.hh>
#include <tmol/score/ljlk/potentials/ljlk_pose_score.hh>

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
auto LJLKPoseScoreDispatch::f(
    TView<Vec<Real, 3>, 3, D> coords,
    TView<Int, 2, D> pose_stack_block_type,

    // dims: n-poses x max-n-blocks x max-n-blocks
    // Quick lookup: given the inds of two blocks, ask: what is the minimum
    // number of chemical bonds that separate any pair of atoms in those
    // blocks? If this minimum is greater than the crossover, then no further
    // logic for deciding whether two atoms in those blocks should have their
    // interaction energies calculated: all should. intentionally small to
    // (possibly) fit in constant cache
    TView<Int, 3, D> pose_stack_min_bond_separation,

    // dims: n-poses x max-n-blocks x max-n-blocks x
    // max-n-interblock-connections x max-n-interblock-connections
    TView<Int, 5, D> pose_stack_inter_block_bondsep,

    //////////////////////
    // Chemical properties
    // how many atoms for a given block
    // Dimsize n_block_types
    TView<Int, 1, D> block_type_n_atoms,

    TView<Int, 2, D> block_type_n_heavy_atoms_in_tile,
    TView<Int, 2, D> block_type_heavy_atoms_in_tile,

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
    TView<LJLKTypeParams<Real>, 1, D> type_params,
    TView<LJGlobalParams<Real>, 1, D> global_params

    ) -> std::tuple<TPack<Real, 2, D>, TPack<Vec<Real, 3>, 4, D>> {
  int const n_poses = coords.size(0);
  int const max_n_blocks = coords.size(1);
  int64_t const max_n_atoms = coords.size(2);
  int const n_block_types = block_type_n_atoms.size(0);
  int const max_n_interblock_bonds =
      block_type_atoms_forming_chemical_bonds.size(1);

  assert(pose_stack_block_type.size(0) == n_poses);
  assert(pose_stack_block_type.size(1) == max_n_blocks);

  assert(pose_stack_min_bond_separation.size(0) == n_poses);
  assert(pose_stack_min_bond_separation.size(1) == max_n_blocks);
  assert(pose_stack_min_bond_separation.size(2) == max_n_blocks);

  assert(pose_stack_inter_block_bondsep.size(0) == n_poses);
  assert(pose_stack_inter_block_bondsep.size(1) == max_n_blocks);
  assert(pose_stack_inter_block_bondsep.size(2) == max_n_blocks);
  assert(pose_stack_inter_block_bondsep.size(3) == max_n_interblock_bonds);
  assert(pose_stack_inter_block_bondsep.size(4) == max_n_interblock_bonds);

  assert(block_type_atom_types.size(0) == n_block_types);
  assert(block_type_atom_types.size(1) == max_n_atoms);
  assert(block_type_n_interblock_bonds.size(0) == n_block_types);
  assert(block_type_atoms_forming_chemical_bonds.size(0) == n_block_types);
  assert(block_type_path_distance.size(0) == n_block_types);
  assert(block_type_path_distance.size(1) == max_n_atoms);
  assert(block_type_path_distance.size(2) == max_n_atoms);

  // auto wcts = std::chrono::system_clock::now();
  // clock_t start_time = clock();

  auto output_t = TPack<Real, 2, D>::zeros({2, n_poses});
  auto output = output_t.view;

  auto dV_dcoords_t =
      TPack<Vec<Real, 3>, 4, D>::zeros({2, n_poses, max_n_blocks, max_n_atoms});
  auto dV_dcoords = dV_dcoords_t.view;

  auto eval_atom_pair = ([=] EIGEN_DEVICE_FUNC(
                             int pose_ind,
                             int block_pair_ind,
                             int atom_pair_ind) {
    int const max_important_bond_separation = 4;

    int const block_ind1 = block_pair_ind / max_n_blocks;
    int const block_ind2 = block_pair_ind % max_n_blocks;
    if (block_ind1 > block_ind2) {
      return;
    }

    int const block_type1 = pose_stack_block_type[pose_ind][block_ind1];
    int const block_type2 = pose_stack_block_type[pose_ind][block_ind2];
    if (block_type1 == -1 || block_type2 == -1) {
      return;
    }

    int atom_type1 = -1;
    int atom_type2 = -1;
    int separation = max_important_bond_separation + 1;
    common::distance<Real>::V_dV_T dist;

    Vec<Real, 3> coord1, coord2;
    int at1, at2;

    if (block_ind1 != block_ind2) {
      // Inter-block interaction

      int const n_atoms1 = block_type_n_atoms[block_type1];
      int const n_atoms2 = block_type_n_atoms[block_type2];

      // for best warp cohesion, mod the atom-pair indices after
      // we have figured out the number of atoms in both blocks;
      // if we modded *before* based on the maximum number of atoms
      // per block, lots of warps with inactive atom-pairs
      // (because they are off the end of the list) would run.
      // By waiting to figure out the i/j inds until after we know
      // how many atoms pairs there will be, we can push all the inactive
      // threads into the same warps and kill those warps early
      if (atom_pair_ind >= n_atoms1 * n_atoms2) {
        return;
      }

      int atom_ind1 = atom_pair_ind / n_atoms2;
      int atom_ind2 = atom_pair_ind % n_atoms2;

      // "count pair" logic
      separation =
          common::count_pair::CountPair<D, Int>::inter_block_separation(
              max_important_bond_separation,
              block_ind1,
              block_ind2,
              block_type1,
              block_type2,
              atom_ind1,
              atom_ind2,
              pose_stack_min_bond_separation[pose_ind],
              pose_stack_inter_block_bondsep[pose_ind],
              block_type_n_interblock_bonds,
              block_type_atoms_forming_chemical_bonds,
              block_type_path_distance);

      at1 = atom_ind1;
      at2 = atom_ind2;
      coord1 = coords[pose_ind][block_ind1][atom_ind1];
      coord2 = coords[pose_ind][block_ind2][atom_ind2];

      dist = distance<Real>::V_dV(coord1, coord2);

      atom_type1 = block_type_atom_types[block_type1][atom_ind1];
      atom_type2 = block_type_atom_types[block_type2][atom_ind2];
    } else {
      // alt_block_ind == neighb_block_ind:
      // intra-block interaction.
      int const n_atoms = block_type_n_atoms[block_type1];
      // see comment in the inter-block interaction regarding the delay of
      // the atom1/atom2 resolution until we know how many atoms are in the
      // particular block we're looking at.
      if (atom_pair_ind >= alt_n_atoms * alt_n_atoms) {
        return;
      }
      int const atom_ind1 = atom_pair_ind / n_atoms;
      int const atom_ind2 = atom_pair_ind % n_atoms;
      if (atom_ind1 >= atom_ind2) {
        // count each intra-block interaction only once
        return;
      }
      at1 = atom_ind1;
      at2 = atom_ind2;

      // TEMP HACK! DON'T CALCULATE ENERGIES WITH BACKBONE ATOMS
      // if (at1 <= 3 || at2 <= 3) {
      //   return;
      // }

      coord1 = coords[pose_ind][block_ind1][atom_ind1];
      coord2 = coords[pose_ind][block_ind1][atom_ind2];
      dist = distance<Real>::V_dV(coord1, coord2);

      separation = block_type_path_distance[block_type1][atom_ind1][atom_ind2];
      atom_type1 = block_type_atom_types[block_type1][atom_ind1];
      atom_type2 = block_type_atom_types[block_type1][atom_ind2];
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
    auto lj = lj_score<Real>::V_dV(
        dist.V,
        separation,
        type_params[atom_type1].lj_params(),
        type_params[atom_type2].lj_params(),
        global_params[0]);

    lk_isotropic_score<Real>::V_dV_T lk;
    lk.V = 0;
    lk.dV_ddist = 0;

    if (type_params[atom_type1].lk_volume > 0
        && type_params[atom_type2].lk_volume > 0) {
      lk = lk_isotropic_score<Real>::V_dV(
          dist.V,
          separation,
          type_params[atom_type1].lk_params(),
          type_params[atom_type2].lk_params(),
          global_params[0]);
    }

    accumulate<D, Real>::add_one_dst(&output[0], pose_ind, lj);
    accumulate<D, Real>::add_one_dst(&output[1], pose_ind, lk);
    accumulate<D, 3, Real>::add(
        &dV_dcoords[0][pose_ind][block_ind1][at1], lj.dV_ddist * dist.dV_dA);
    accumulate<D, 3, Real>::add(
        &dV_dcoords[0][pose_ind][block_ind2][at2], lj.dV_ddist * dist.dV_dB);
    accumulate<D, 3, Real>::add(
        &dV_dcoords[1][pose_ind][block_ind1][at1], lk.dV_ddist * dist.dV_dA);
    accumulate<D, 3, Real>::add(
        &dV_dcoords[1][pose_ind][block_ind2][at2], lk.dV_ddist * dist.dV_dB);
  });

  DeviceDispatch<D>::foreach_combination_triple(
      n_poses,
      max_n_blocks * max_n_blocks,
      max_n_atoms * max_n_atoms,
      eval_atom_pair);

  return {output_t, dV_dcoords_t};

#ifdef __CUDACC__
  // float first;
  // cudaMemcpy(&first, &output[0], sizeof(float), cudaMemcpyDeviceToHost);
  //
  // clock_t stop_time = clock();
  // std::chrono::duration<double> wctduration =
  // (std::chrono::system_clock::now() - wcts);
  //
  // std::cout << n_poses << " " << n_contexts << " " <<n_alternate_blocks <<
  // " "; std::cout << n_alternate_blocks * max_n_neighbors * max_n_atoms *
  // max_n_atoms << " "; std::cout << "runtime? " << ((double)stop_time -
  // start_time) / CLOCKS_PER_SEC
  //           << " wall time: " << wctduration.count() << " " << first
  //           << std::endl;
#endif
}

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
