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

#include "params.hh"

namespace tmol {
namespace score {
namespace cartbonded {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <
    template <tmol::Device>
    class DeviceOps,
    tmol::Device D,
    typename Real,
    typename Int>
struct CartBondedPoseScoreDispatch {
  static auto forward(
      TView<Vec<Real, 3>, 2, D> coords,
      TView<Int, 2, D> pose_stack_block_coord_offset,
      TView<Int, 2, D> pose_stack_block_type,
      TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
      TView<Vec<Int, 3>, 3, D> atom_paths_from_conn,
      TView<Int, 2, D> atom_unique_ids,
      TView<Int, 2, D> atom_wildcard_ids,
      TView<Vec<Int, 5>, 1, D> hash_keys,
      TView<Vec<Real, 7>, 1, D> hash_values,
      TView<Vec<Int, 4>, 1, D> cart_subgraphs,
      TView<Int, 1, D> cart_subgraph_offsets,

      int max_subgraphs_per_block,
      bool output_block_pair_energies,

      bool compute_derivs

      ) -> std::tuple<TPack<Real, 4, D>, TPack<Vec<Real, 3>, 3, D>>;

  static auto backward(
      TView<Vec<Real, 3>, 2, D> coords,
      TView<Int, 2, D> pose_stack_block_coord_offset,
      TView<Int, 2, D> pose_stack_block_type,
      TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
      TView<Vec<Int, 3>, 3, D> atom_paths_from_conn,
      TView<Int, 2, D> atom_unique_ids,
      TView<Int, 2, D> atom_wildcard_ids,
      TView<Vec<Int, 5>, 1, D> hash_keys,
      TView<Vec<Real, 7>, 1, D> hash_values,
      TView<Vec<Int, 4>, 1, D> cart_subgraphs,
      TView<Int, 1, D> cart_subgraph_offsets,

      int max_subgraphs_per_block,

      TView<Real, 4, D> dTdV  // nterms x nposes x (1|len) x (1|len)
      ) -> TPack<Vec<Real, 3>, 3, D>;
};

}  // namespace potentials
}  // namespace cartbonded
}  // namespace score
}  // namespace tmol
