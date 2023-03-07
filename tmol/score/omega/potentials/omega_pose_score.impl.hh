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
#include <tmol/score/common/sphere_overlap.impl.hh>
#include <tmol/score/common/tile_atom_pair_evaluation.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/warp_segreduce.hh>
#include <tmol/score/common/warp_stride_reduce.hh>

#include <tmol/score/omega/potentials/omega_pose_score.hh>

// Operator definitions; safe for CPU compilation
#include <moderngpu/operators.hxx>

#include <chrono>

#include "potentials.hh"

// The maximum number of inter-residue chemical bonds
#define MAX_N_CONN 4
#define TILE_SIZE 32

namespace tmol {
namespace score {
namespace omega {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;
template <typename Real>
using CoordQuad = Eigen::Matrix<Real, 4, 3>;

template <typename Real, typename Int, tmol::Device D>
TMOL_DEVICE_FUNC int resolve_atom_from_uaid(
    TensorAccessor<Int, 1, D> uaid,
    int block_index,
    int pose_index,

    TView<Vec<Real, 3>, 2, D> coords,
    TView<Int, 2, D> pose_stack_block_coord_offset,
    TView<Int, 2, D> pose_stack_block_type,
    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<Int, 3, D> block_type_omega_quad_uaids,
    TView<Int, 3, D> block_type_atom_downstream_of_conn) {
  if (uaid[0] != -1) {  // This uaid resides in this block
    int block_coord_offset =
        pose_stack_block_coord_offset[pose_index][block_index];
    return block_coord_offset + uaid[0];  // The actual global atom index
  } else {                                // We need to follow to another block
    int connection_index = uaid[1];
    int sep = uaid[2];

    const Vec<Int, 2>& connection =
        pose_stack_inter_block_connections[pose_index][block_index]
                                          [connection_index];
    int other_block_index = connection[0];
    int other_connection_index = connection[1];
    int other_block_type_index =
        pose_stack_block_type[pose_index][other_block_index];

    int idx = block_type_atom_downstream_of_conn[other_block_type_index]
                                                [other_connection_index]
                                                [sep];  // The offset within the
                                                        // other block
    int block_coord_offset =
        pose_stack_block_coord_offset[pose_index][other_block_index];
    return block_coord_offset + idx;
  }
}

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto OmegaPoseScoreDispatch<DeviceDispatch, D, Real, Int>::f(
    TView<Vec<Real, 3>, 2, D> coords,
    TView<Int, 2, D> pose_stack_block_coord_offset,
    TView<Int, 2, D> pose_stack_block_type,
    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<Int, 3, D> block_type_omega_quad_uaids,
    TView<Int, 3, D> block_type_atom_downstream_of_conn,

    TView<OmegaGlobalParams<Real>, 1, D> global_params,

    bool compute_derivs

    ) -> std::tuple<TPack<Real, 1, D>, TPack<Vec<Real, 3>, 2, D>> {
  int const n_poses = coords.size(0);
  auto V_t = TPack<Real, 1, D>::zeros({n_poses});
  auto dV_dx_t = TPack<Vec<Real, 3>, 2, D>::zeros({n_poses, coords.size(1)});

  auto V = V_t.view;
  auto dV_dx = dV_dx_t.view;

  // Optimal launch box on v100 and a100 is nt=32, vt=1
  LAUNCH_BOX_32;
  // Define nt and reduce_t
  CTA_REAL_REDUCE_T_TYPEDEF;

  auto func = ([=] TMOL_DEVICE_FUNC(int pose_index, int block_index) {
    // printf("%d,%d %f\n", pose_index, block_index, global_params[0].K);

    int block_type_index = pose_stack_block_type[pose_index][block_index];

    int block_coord_offset =
        pose_stack_block_coord_offset[pose_index][block_index];

    CoordQuad<Real> omegacoords;
    for (int i = 0; i < 4; i++) {
      const TensorAccessor<Int, 1, D>& omega_a_uaid =
          block_type_omega_quad_uaids[block_type_index][i];
      int omega_a_ind = resolve_atom_from_uaid<Real, Int, D>(
          omega_a_uaid,
          block_index,
          pose_index,

          coords,
          pose_stack_block_coord_offset,
          pose_stack_block_type,
          pose_stack_inter_block_connections,
          block_type_omega_quad_uaids,
          block_type_atom_downstream_of_conn);

      const Vec<Real, 3>& omega_coord = coords[pose_index][omega_a_ind];

      omegacoords.row(i) = omega_coord;

      /*const TensorAccessor<Int, 1, D> & omega_a_uaid =
      block_type_omega_quad_uaids[block_type_index][0]; const
      TensorAccessor<Int, 1, D> & omega_b_uaid =
      block_type_omega_quad_uaids[block_type_index][1]; const
      TensorAccessor<Int, 1, D> & omega_c_uaid =
      block_type_omega_quad_uaids[block_type_index][2]; const
      TensorAccessor<Int, 1, D> & omega_d_uaid =
      block_type_omega_quad_uaids[block_type_index][3];

      const Vec<Real, 3> & omega_a_coord = coords[pose_index][block_coord_offset
      + omega_a_uaid[0]]; const Vec<Real, 3> & omega_b_coord =
      coords[pose_index][block_coord_offset + omega_b_uaid[0]]; const Vec<Real,
      3> & omega_c_coord = coords[pose_index][block_coord_offset +
      omega_c_uaid[0]]; const Vec<Real, 3> & omega_d_coord =
      coords[pose_index][block_coord_offset + omega_d_uaid[0]];*/

      if (pose_index == 0 && block_index == 0) {
#ifndef __CUDACC__
        std::cout << omega_coord << std::endl << std::endl;
        // std::cout << omega_b_coord << std::endl << std::endl;
#endif
      }
    }

    auto omega = omega_V_dV<D, Real, Int>(omegacoords, global_params[0].K);

    accumulate<D, Real>::add(V[pose_index], common::get<0>(omega));
    /*for (int j = 0; j < 4; ++j) {
      accumulate<D, Vec<Real, 3>>::add(
          dV_dx[pose_index][(int)omega_indices[pose_index][i].atoms[j]],
          common::get<1>(omega).row(j));
    }*/

#ifndef __CUDACC__
    std::cout << common::get<0>(omega) << std::endl << std::endl;
    // std::cout << omega_b_coord << std::endl << std::endl;
#endif

    /*auto omega_a = block_type_omega_quad_uaids;
    for (int ii = 0; ii < omega_a.size(0); ++ii) {
      printf("%i: ", ii);
      for (int jj = 0; jj < omega_a.size(1); ++jj) {
        for (int kk = 0; kk < omega_a.size(2); ++kk) {
          printf("%i ", omega_a[ii][jj][kk]);
        }
        printf("  ");
      }
      printf("\n");
    }*/
  });

  // int num_Vs = coords.size(2);
  // DeviceDispatch<D>::template foreach_workgroup<launch_t>(num_Vs, func);
  int total_blocks = pose_stack_block_coord_offset.size(1);
  DeviceDispatch<D>::forall_stacks(n_poses, total_blocks, func);

  return {V_t, dV_dx_t};
}  // namespace potentials

}  // namespace potentials
}  // namespace omega
}  // namespace score
}  // namespace tmol
