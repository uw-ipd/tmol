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
#include <tmol/score/common/data_loading.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/geom.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/uaid_util.hh>
#include <tmol/score/common/warp_segreduce.hh>
#include <tmol/score/common/warp_stride_reduce.hh>

#include <tmol/score/cartbonded/potentials/cartbonded_pose_score.hh>

// Operator definitions; safe for CPU compilation
#include <moderngpu/operators.hxx>

#include <chrono>

#include "potentials.hh"

namespace tmol {
namespace score {
namespace cartbonded {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;
template <typename Real>
using CoordQuad = Eigen::Matrix<Real, 4, 3>;

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto CartBondedPoseScoreDispatch<DeviceDispatch, D, Real, Int>::f(
    TView<Vec<Real, 3>, 2, D> coords,
    TView<Int, 2, D> pose_stack_block_coord_offset,
    TView<Int, 2, D> pose_stack_block_type,
    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<Int, 3, D> block_type_atom_downstream_of_conn,

    TView<CartBondedGlobalParams<Real>, 1, D> global_params,

    bool compute_derivs

    ) -> std::tuple<TPack<Real, 2, D>, TPack<Vec<Real, 3>, 3, D>> {
  int const n_poses = coords.size(0);
  int const max_n_atoms = coords.size(1);
  auto V_t = TPack<Real, 2, D>::zeros({1, n_poses});
  auto dV_dx_t = TPack<Vec<Real, 3>, 3, D>::zeros({1, n_poses, max_n_atoms});

  auto V = V_t.view;
  auto dV_dx = dV_dx_t.view;

  // Optimal launch box on v100 and a100 is nt=32, vt=1
  LAUNCH_BOX_32;

  auto func = ([=] TMOL_DEVICE_FUNC(int pose_index, int block_index) {
    int block_type_index = pose_stack_block_type[pose_index][block_index];

    int block_coord_offset =
        pose_stack_block_coord_offset[pose_index][block_index];

    CoordQuad<Real> cartbondedcoords;
    Int cartbonded_indices[4];
    for (int i = 0; i < 4; i++) {
      const TensorAccessor<Int, 1, D>& cartbonded_atom_uaid =
          block_type_cartbonded_quad_uaids[block_type_index][i];

      // Check to see if the cartbonded uaids are actually defined for this
      // block type. If the atom offset [0] or the connection index [1] are both
      // -1, this is a sentinel for an undefined cartbonded.
      if (cartbonded_atom_uaid[0] == -1 && cartbonded_atom_uaid[1] == -1)
        return;

      int cartbonded_atom_ind = resolve_atom_from_uaid<Real, Int, D>(
          cartbonded_atom_uaid,
          block_index,
          pose_index,

          pose_stack_block_coord_offset,
          pose_stack_block_type,
          pose_stack_inter_block_connections,
          block_type_atom_downstream_of_conn);

      if (cartbonded_atom_ind == -1) {
        // The UAID resolution failed! In this case, we should just skip the
        // cartbonded for this block
        return;
      }

      const Vec<Real, 3>& cartbonded_coord =
          coords[pose_index][cartbonded_atom_ind];

      cartbondedcoords.row(i) = cartbonded_coord;
      cartbonded_indices[i] = (cartbonded_atom_ind);
    }

    // auto cartbonded = cartbonded_V_dV<D, Real, Int>(cartbondedcoords,
    // global_params[0].K);

    // accumulate<D, Real>::add(V[0][pose_index], common::get<0>(cartbonded));
    for (int j = 0; j < 4; ++j) {
      // accumulate<D, Vec<Real, 3>>::add(
      // dV_dx[0][pose_index][cartbonded_indices[j]],
      // common::get<1>(cartbonded).row(j));
    }
  });

  int total_blocks = pose_stack_block_coord_offset.size(1);
  DeviceDispatch<D>::forall_stacks(n_poses, total_blocks, func);

  return {V_t, dV_dx_t};
}  // namespace potentials

}  // namespace potentials
}  // namespace cartbonded
}  // namespace score
}  // namespace tmol
