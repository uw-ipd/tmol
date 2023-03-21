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

// Operator definitions; safe for CPU compilation
#include <moderngpu/operators.hxx>

#include <chrono>

#include "params.hh"
#include "potentials.hh"

namespace tmol {
namespace score {
namespace disulfide {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;
template <typename Real>
using CoordQuad = Eigen::Matrix<Real, 4, 3>;

template <typename Real, typename Int, tmol::Device D>
TMOL_DEVICE_FUNC int errfc(Real x) {}

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
    TView<Int, 2, D> disulfide_conns,
    TView<Int, 3, D> block_type_atom_downstream_of_conn,

    TView<DisulfideGlobalParams<Real>, 1, D> global_params,

    bool compute_derivs

    ) -> std::tuple<TPack<Real, 2, D>, TPack<Vec<Real, 3>, 3, D>> {
  int const n_poses = coords.size(0);
  int const max_n_atoms = coords.size(1);
  int const max_n_conns = disulfide_conns.size(1);
  auto V_t = TPack<Real, 2, D>::zeros({1, n_poses});
  auto dV_dx_t = TPack<Vec<Real, 3>, 3, D>::zeros({1, n_poses, max_n_atoms});

  auto V = V_t.view;
  auto dV_dx = dV_dx_t.view;

  // Optimal launch box on v100 and a100 is nt=32, vt=1
  LAUNCH_BOX_32;

  auto func = ([=] TMOL_DEVICE_FUNC(int pose_index, int block_index) {
    int block_type_index = pose_stack_block_type[pose_index][block_index];
    int block_atom_offset =
        pose_stack_block_coord_offset[pose_index][block_index];

    Real score = 0;

    for (int conn_index = 0; conn_index < max_n_conns; conn_index++) {
      if (disulfide_conns[block_type_index][conn_index]) {
        int other_block_index =
            pose_stack_inter_block_connections[pose_index][block_index]
                                              [conn_index][0];
        int other_conn_index =
            pose_stack_inter_block_connections[pose_index][block_index]
                                              [conn_index][1];  // BAD?
        int other_block_atom_offset =
            pose_stack_block_coord_offset[pose_index][other_block_index];

        if (other_block_index
            == -1) {  // Skip this disulfide if the other block doesn't exist
          printf("no partner");
          continue;
        }

        int other_block_type_index =
            pose_stack_block_type[pose_index][other_block_index];

        if (!disulfide_conns[other_block_type_index]
                            [other_conn_index]) {  // Skip this disulfide if the
                                                   // other end isn't capable of
                                                   // a disulfide bond
          printf("partner isn't a disulfide");
          continue;
        }

        auto block1_CA =
            coords[pose_index]
                  [block_atom_offset
                   + block_type_atom_downstream_of_conn[block_type_index]
                                                       [conn_index][2]];
        auto block1_CB =
            coords[pose_index]
                  [block_atom_offset
                   + block_type_atom_downstream_of_conn[block_type_index]
                                                       [conn_index][1]];
        auto block1_S =
            coords[pose_index]
                  [block_atom_offset
                   + block_type_atom_downstream_of_conn[block_type_index]
                                                       [conn_index][0]];

        auto block2_S =
            coords[pose_index]
                  [other_block_atom_offset
                   + block_type_atom_downstream_of_conn[other_block_type_index]
                                                       [other_conn_index][0]];
        auto block2_CB =
            coords[pose_index]
                  [other_block_atom_offset
                   + block_type_atom_downstream_of_conn[other_block_type_index]
                                                       [other_conn_index][1]];
        auto block2_CA =
            coords[pose_index]
                  [other_block_atom_offset
                   + block_type_atom_downstream_of_conn[other_block_type_index]
                                                       [other_conn_index][2]];

        auto distance_between_sulfers =
            distance<Real>::V_dV(block1_S, block2_S);
        auto bond_angle_1 =
            pt_interior_angle<Real>::V_dV(block1_CB, block1_S, block2_S);
        auto bond_angle_2 =
            pt_interior_angle<Real>::V_dV(block2_CB, block2_S, block1_S);
        auto disulfide_dihedral_angle = dihedral_angle<Real>::V_dV(
            block1_CB, block1_S, block2_S, block2_CB);
        auto disulfide_ca_dihedral_angle_1 = dihedral_angle<Real>::V_dV(
            block1_CA, block1_CB, block1_S, block2_S);
        auto disulfide_ca_dihedral_angle_2 = dihedral_angle<Real>::V_dV(
            block2_CA, block2_CB, block2_S, block1_S);

        const Real mest = exp(-20.0);
        // Calculate Distance
        {
          Real z = (distance_between_sulfers.V - global_params[0].d_location)
                   / global_params[0].d_scale;
          // Real score_d = z*z/2 - log( errfc( -global_params[0].d_shape*z /
          // sqrt(2.0) ) + mest );
          printf("DIST: %f", z);
          score = z;
        }
      }
    }

    int block_coord_offset =
        pose_stack_block_coord_offset[pose_index][block_index];

    Int disulfide_indices[4];
    for (int i = 0; i < 4; i++) {
    }

    /*auto disulfide = disulfide_V_dV<D, Real, Int>(disulfidecoords,
     * global_params[0].K);*/

    accumulate<D, Real>::add(V[0][pose_index], score);
    /*for (int j = 0; j < 4; ++j) {
      accumulate<D, Vec<Real, 3>>::add(
          dV_dx[0][pose_index][disulfide_indices[j]],
    common::get<1>(disulfide).row(j));
    }*/
  });

  int total_blocks = pose_stack_block_coord_offset.size(1);
  DeviceDispatch<D>::forall_stacks(n_poses, total_blocks, func);

  return {V_t, dV_dx_t};
}  // namespace potentials

}  // namespace potentials
}  // namespace disulfide
}  // namespace score
}  // namespace tmol
