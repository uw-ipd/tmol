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

#include "water.hh"

namespace tmol {
namespace score {
namespace lk_ball {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int,
    int MAX_WATER>
struct LKBallRPEDispatch {
  static auto f(
      TView<Vec<Real, 3>, 3, D> context_coords,
      TView<Int, 2, D> context_block_type,
      TView<Vec<Real, 3>, 2, D> alternate_coords,
      TView<Vec<Int, 3>, 1, D>
          alternate_ids,  // 0 == context id; 1 == block id; 2 == block type

      //
      TView<Vec<Real, 3>, 4, D> context_water_coords,

      // which system does a given context belong to
      TView<Int, 1, D> context_system_ids,

      // dims: n-systems x max-n-blocks x max-n-blocks
      // Quick lookup: given the inds of two blocks, ask: what is the minimum
      // number of chemical bonds that separate any pair of atoms in those
      // blocks? If this minimum is greater than the crossover, then no further
      // logic for deciding whether two atoms in those blocks should have their
      // interaction energies calculated: all should. intentionally small to
      // (possibly) fit in constant cache
      TView<Int, 3, D> system_min_bond_separation,

      // dims: n-systems x max-n-blocks x max-n-blocks x
      // max-n-interblock-connections x max-n-interblock-connections
      TView<Int, 5, D> system_inter_block_bondsep,

      // dims n-systems x max-n-blocks x max-n-neighbors
      // -1 as the sentinel
      TView<Int, 3, D> system_neighbor_list,

      //////////////////////
      // Chemical properties
      // Dimsize n_block_types x max_n_atoms
      TView<uint8_t, 2, D> bt_is_acceptor,
      TView<Int, 2, D> bt_acceptor_type,
      TView<Int, 2, D> bt_acceptor_hybridization,
      TView<Int, 3, D> bt_acceptor_base_ind,

      TView<uint8_t, 2, D> bt_is_donor,
      TView<Int, 2, D> bt_donor_type,

      // Indices of the attached hydrogens on a donor
      TView<Int, 3, D> bt_donor_attached_hydrogens,

      // // TView<Int, 1, D> block_type_n_atoms,
      //
      // // what are the atom types for these atoms
      // // Dimsize: n_block_types x max_n_atoms
      // TView<Int, 2, D> block_type_atom_types,
      //
      // // how many inter-block chemical bonds are there
      // // Dimsize: n_block_types
      // TView<Int, 1, D> block_type_n_interblock_bonds,
      //
      // // what atoms form the inter-block chemical bonds
      // // Dimsize: n_block_types x max_n_interblock_bonds
      // TView<Int, 2, D> block_type_atoms_forming_chemical_bonds,
      //
      // // what is the path distance between pairs of atoms in the block
      // // Dimsize: n_block_types x max_n_atoms x max_n_atoms
      // TView<Int, 3, D> block_type_path_distance,
      // //////////////////////
      //
      // // LJ parameters
      // TView<LJTypeParams<Real>, 1, D> type_params,
      // TView<LJGlobalParams<Real>, 1, D> global_params,
      // TView<Real, 1, D> lj_lk_weights

      TView<LKBallWaterGenGlobalParams<Real>, 1, D> global_params,
      TView<Real, 1, D> sp2_water_tors,
      TView<Real, 1, D> sp3_water_tors,
      TView<Real, 1, D> ring_water_tors)
      -> std::tuple<TPack<Real, 1, D>, TPack<int64_t, 1, D>>;
};

}  // namespace potentials
}  // namespace lk_ball
}  // namespace score
}  // namespace tmol
