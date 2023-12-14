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

#include <tmol/score/disulfide/potentials/disulfide_pose_score.hh>

// Operator definitions; safe for CPU comM_PIlation
#include <moderngpu/operators.hxx>

#include <chrono>
#include <cmath>

#include "params.hh"
#include "potentials.hh"

namespace tmol {
namespace score {
namespace disulfide {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto DisulfidePoseScoreDispatch<DeviceDispatch, D, Real, Int>::f(
    TView<Vec<Real, 3>, 2, D> coords,
    TView<Int, 2, D> pose_stack_block_coord_offset,
    TView<Int, 2, D> pose_stack_block_type,
    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<bool, 2, D> disulfide_conns,
    TView<Int, 3, D> block_type_atom_downstream_of_conn,

    TView<DisulfideGlobalParams<Real>, 1, D> global_params,
    bool output_block_pair_energies,
    bool compute_derivs

    ) -> std::tuple<TPack<Real, 4, D>, TPack<Vec<Real, 3>, 3, D>> {
  int const n_poses = coords.size(0);
  int const max_n_blocks = pose_stack_block_type.size(1);
  int const max_n_atoms = coords.size(1);
  int const max_n_conns = disulfide_conns.size(1);

  // auto V_t = TPack<Real, 2, D>::zeros({1, n_poses});
  TPack<Real, 4, D> V_t;
  if (output_block_pair_energies) {
    V_t = TPack<Real, 4, D>::zeros({1, n_poses, max_n_blocks, max_n_blocks});
  } else {
    V_t = TPack<Real, 4, D>::zeros({1, n_poses, 1, 1});
  }

  auto dV_dx_t = TPack<Vec<Real, 3>, 3, D>::zeros({1, n_poses, max_n_atoms});

  auto V = V_t.view;
  auto dV_dx = dV_dx_t.view;

  // Optimal launch box on v100 and a100 is nt=32, vt=1
  LAUNCH_BOX_32;

  auto func = ([=] TMOL_DEVICE_FUNC(int pose_index, int block_index) {
    const auto& params = global_params[0];
    const auto& inter_block_connections =
        pose_stack_inter_block_connections[pose_index];
    const auto& block_coord_offset = pose_stack_block_coord_offset[pose_index];
    const auto& block_type = pose_stack_block_type[pose_index];

    auto& pose_V = V[0][pose_index];
    auto& pose_dV_dx = dV_dx[0][pose_index];

    int block_type_index = pose_stack_block_type[pose_index][block_index];
    int block_atom_offset =
        pose_stack_block_coord_offset[pose_index][block_index];

    for (int conn_index = 0; conn_index < max_n_conns; conn_index++) {
      if (disulfide_conns[block_type_index][conn_index]) {
        int other_block_index =
            inter_block_connections[block_index][conn_index][0];
        int other_conn_index =
            inter_block_connections[block_index][conn_index][1];
        int other_block_atom_offset = block_coord_offset[other_block_index];

        // Make sure we only calculate the disulfide once per pair of blocks
        if (other_block_index < block_index) continue;

        // Skip if the other block doesn't exist
        if (other_block_index == -1) continue;

        int other_block_type_index = block_type[other_block_index];

        // Skip if the other end isn't capable of a disulfide
        // bond
        if (!disulfide_conns[other_block_type_index][other_conn_index])
          continue;

        int block1_ind = block_index;
        int block2_ind = other_block_index;
        // Get the 6 atoms that we need for the disulfides
        auto block1_CA_ind =
            block_atom_offset
            + block_type_atom_downstream_of_conn[block_type_index][conn_index]
                                                [2];
        auto block1_CB_ind =
            block_atom_offset
            + block_type_atom_downstream_of_conn[block_type_index][conn_index]
                                                [1];
        auto block1_S_ind =
            block_atom_offset
            + block_type_atom_downstream_of_conn[block_type_index][conn_index]
                                                [0];
        auto block2_S_ind =
            other_block_atom_offset
            + block_type_atom_downstream_of_conn[other_block_type_index]
                                                [other_conn_index][0];
        auto block2_CB_ind =
            other_block_atom_offset
            + block_type_atom_downstream_of_conn[other_block_type_index]
                                                [other_conn_index][1];
        auto block2_CA_ind =
            other_block_atom_offset
            + block_type_atom_downstream_of_conn[other_block_type_index]
                                                [other_conn_index][2];

        // Calculate score and derivatives and put them in the out tensors
        accumulate_disulfide_potential<Real, D>(
            coords[pose_index],
            block1_ind,
            block1_CA_ind,
            block1_CB_ind,
            block1_S_ind,
            block2_ind,
            block2_S_ind,
            block2_CB_ind,
            block2_CA_ind,

            params,

            output_block_pair_energies,
            pose_V,
            pose_dV_dx);
      }
    }
  });

  int total_blocks = pose_stack_block_coord_offset.size(1);
  DeviceDispatch<D>::forall_stacks(n_poses, total_blocks, func);

  return {V_t, dV_dx_t};
}  // namespace potentials

}  // namespace potentials
}  // namespace disulfide
}  // namespace score
}  // namespace tmol
