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
template <typename Real>
using CoordQuad = Eigen::Matrix<Real, 4, 3>;

// template <typename Real, typename Int, tmol::Device D>
// TMOL_DEVICE_FUNC int errfc(Real x) {}

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

    for (int conn_index = 0; conn_index < max_n_conns; conn_index++) {
      if (disulfide_conns[block_type_index][conn_index]) {
        int other_block_index =
            pose_stack_inter_block_connections[pose_index][block_index]
                                              [conn_index][0];
        int other_conn_index =
            pose_stack_inter_block_connections[pose_index][block_index]
                                              [conn_index][1];
        int other_block_atom_offset =
            pose_stack_block_coord_offset[pose_index][other_block_index];

        // Make sure we only calculate the disulfide once per pair of blocks
        if (other_block_index < block_index) continue;

        // Skip this disulfide if the other block doesn't exist
        if (other_block_index == -1) continue;

        int other_block_type_index =
            pose_stack_block_type[pose_index][other_block_index];

        // Skip this disulfide if the other end isn't capable of a disulfide
        // bond
        if (!disulfide_conns[other_block_type_index][other_conn_index])
          continue;

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

        auto block1_CA = coords[pose_index][block1_CA_ind];
        auto block1_CB = coords[pose_index][block1_CB_ind];
        auto block1_S = coords[pose_index][block1_S_ind];

        auto block2_S = coords[pose_index][block2_S_ind];
        auto block2_CB = coords[pose_index][block2_CB_ind];
        auto block2_CA = coords[pose_index][block2_CA_ind];

        auto ssdist = distance<Real>::V_dV(block1_S, block2_S);
        auto csang_1 =
            pt_interior_angle<Real>::V_dV(block1_CB, block1_S, block2_S);
        auto csang_2 =
            pt_interior_angle<Real>::V_dV(block2_CB, block2_S, block1_S);
        auto dihed = dihedral_angle<Real>::V_dV(
            block1_CB, block1_S, block2_S, block2_CB);
        auto disulf_ca_dihedral_angle_1 = dihedral_angle<Real>::V_dV(
            block1_CA, block1_CB, block1_S, block2_S);
        auto disulf_ca_dihedral_angle_2 = dihedral_angle<Real>::V_dV(
            block2_CA, block2_CB, block2_S, block1_S);

        // convert to degrees
        Real radians_to_degrees(57.2958);
        csang_1.V *= radians_to_degrees;
        csang_2.V *= radians_to_degrees;
        dihed.V *= radians_to_degrees;
        disulf_ca_dihedral_angle_1.V *= radians_to_degrees;
        disulf_ca_dihedral_angle_2.V *= radians_to_degrees;

        const Real MEST = exp(-20.0);
        const Real wt_dihSS_(0.1);
        const Real wt_dihCS_(0.1);
        const Real wt_ang_(0.1);
        const Real wt_len_(0.1);
        const Real shift_(2.0);

        Real const res1_d_multiplier(1.0);
        Real const res2_d_multiplier(1.0);

        Real score = -shift_;

        {  // Calculate Distance
          Real z = (ssdist.V - global_params[0].d_location)
                   / global_params[0].d_scale;
          Real score_d =
              z * z / 2
              - log(std::erfc(-global_params[0].d_shape * z / sqrt(2.0))
                    + MEST);
          score += wt_len_ * score_d;

          Real dscore_d =
              z / global_params[0].d_scale
              - (exp(-0.5 * z * z * global_params[0].d_shape
                     * global_params[0].d_shape)
                 * sqrt(2.0 / M_PI) * global_params[0].d_shape)
                    / (global_params[0].d_scale
                           * std::erfc(
                                 -global_params[0].d_shape * z / sqrt(2.0))
                       + 1.e-12);
          dscore_d *= wt_len_;
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[0][pose_index][block1_S_ind], dscore_d * ssdist.dV_dA);
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[0][pose_index][block2_S_ind], dscore_d * ssdist.dV_dB);
        }

        {  // Calculate Angles
          Real ang_score(0);
          Real angle1(csang_1.V), angle2(csang_2.V);
          ang_score +=
              wt_ang_
              * (-global_params[0].a_logA
                 - global_params[0].a_kappa
                       * cos(M_PI / 180 * (angle1 - global_params[0].a_mu)));
          ang_score +=
              wt_ang_
              * (-global_params[0].a_logA
                 - global_params[0].a_kappa
                       * cos(M_PI / 180 * (angle2 - global_params[0].a_mu)));
          score += ang_score;

          Real dscore_a = global_params[0].a_kappa
                          * sin(M_PI / 180 * (angle1 - global_params[0].a_mu))
                          * wt_ang_;
          Real dscore_b = global_params[0].a_kappa
                          * sin(M_PI / 180 * (angle2 - global_params[0].a_mu))
                          * wt_ang_;
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[0][pose_index][block1_CB_ind], dscore_a * csang_1.dV_dA);
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[0][pose_index][block1_S_ind], dscore_a * csang_1.dV_dB);
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[0][pose_index][block2_S_ind], dscore_a * csang_1.dV_dC);
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[0][pose_index][block2_CB_ind], dscore_b * csang_2.dV_dA);
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[0][pose_index][block2_S_ind], dscore_b * csang_2.dV_dB);
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[0][pose_index][block1_S_ind], dscore_b * csang_2.dV_dC);
        }

        {  // SS dihed
          Real ang_ss(dihed.V), exp_score1(0.0), exp_score2(0.0);
          exp_score1 = exp(global_params[0].dss_logA1)
                       * exp(global_params[0].dss_kappa1
                             * cos(M_PI / 180
                                   * (res1_d_multiplier * ang_ss
                                      - global_params[0].dss_mu1)));
          exp_score2 = exp(global_params[0].dss_logA2)
                       * exp(global_params[0].dss_kappa2
                             * cos(M_PI / 180
                                   * (res1_d_multiplier * ang_ss
                                      - global_params[0].dss_mu2)));
          Real score_ss = -log(exp_score1 + exp_score2 + MEST);
          score += wt_dihSS_ * score_ss;

          Real dscore_ss(0.0);
          dscore_ss +=
              exp_score1 * global_params[0].dss_kappa1
              * sin(M_PI / 180
                    * (res1_d_multiplier * dihed.V - global_params[0].dss_mu1));
          dscore_ss +=
              exp_score2 * global_params[0].dss_kappa2
              * sin(M_PI / 180
                    * (res1_d_multiplier * dihed.V - global_params[0].dss_mu2));
          dscore_ss /= (exp_score1 + exp_score2 + MEST);
          dscore_ss *= wt_dihSS_;

          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[0][pose_index][block1_CB_ind], dscore_ss * dihed.dV_dI);
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[0][pose_index][block1_S_ind], dscore_ss * dihed.dV_dJ);
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[0][pose_index][block2_S_ind], dscore_ss * dihed.dV_dK);
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[0][pose_index][block2_CB_ind], dscore_ss * dihed.dV_dL);
        }

        {  // CB-S dihed
          Real angle1(disulf_ca_dihedral_angle_1.V),
              angle2(disulf_ca_dihedral_angle_2.V);
          Real exp_score1 = exp(global_params[0].dcs_logA1)
                            * exp(global_params[0].dcs_kappa1
                                  * cos(M_PI / 180
                                        * (res1_d_multiplier * angle1
                                           - global_params[0].dcs_mu1)));
          Real exp_score2 = exp(global_params[0].dcs_logA2)
                            * exp(global_params[0].dcs_kappa2
                                  * cos(M_PI / 180
                                        * (res1_d_multiplier * angle1
                                           - global_params[0].dcs_mu2)));
          Real exp_score3 = exp(global_params[0].dcs_logA3)
                            * exp(global_params[0].dcs_kappa3
                                  * cos(M_PI / 180
                                        * (res1_d_multiplier * angle1
                                           - global_params[0].dcs_mu3)));
          score +=
              wt_dihCS_ * (-log(exp_score1 + exp_score2 + exp_score3 + MEST));

          Real dscore_cs = 0.0;
          dscore_cs += exp_score1 * global_params[0].dcs_kappa1
                       * sin(M_PI / 180 * (angle1 - global_params[0].dcs_mu1));
          dscore_cs += exp_score2 * global_params[0].dcs_kappa2
                       * sin(M_PI / 180 * (angle1 - global_params[0].dcs_mu2));
          dscore_cs += exp_score3 * global_params[0].dcs_kappa3
                       * sin(M_PI / 180 * (angle1 - global_params[0].dcs_mu3));
          dscore_cs /= (exp_score1 + exp_score2 + exp_score3 + MEST);
          dscore_cs *= wt_dihCS_;

          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[0][pose_index][block1_CA_ind],
              dscore_cs * disulf_ca_dihedral_angle_1.dV_dI);
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[0][pose_index][block1_CB_ind],
              dscore_cs * disulf_ca_dihedral_angle_1.dV_dJ);
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[0][pose_index][block1_S_ind],
              dscore_cs * disulf_ca_dihedral_angle_1.dV_dK);
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[0][pose_index][block2_S_ind],
              dscore_cs * disulf_ca_dihedral_angle_1.dV_dL);

          exp_score1 = exp(global_params[0].dcs_logA1)
                       * exp(global_params[0].dcs_kappa1
                             * cos(M_PI / 180
                                   * (res2_d_multiplier * angle2
                                      - global_params[0].dcs_mu1)));
          exp_score2 = exp(global_params[0].dcs_logA2)
                       * exp(global_params[0].dcs_kappa2
                             * cos(M_PI / 180
                                   * (res2_d_multiplier * angle2
                                      - global_params[0].dcs_mu2)));
          exp_score3 = exp(global_params[0].dcs_logA3)
                       * exp(global_params[0].dcs_kappa3
                             * cos(M_PI / 180
                                   * (res2_d_multiplier * angle2
                                      - global_params[0].dcs_mu3)));
          score +=
              wt_dihCS_ * (-log(exp_score1 + exp_score2 + exp_score3 + MEST));

          dscore_cs = 0.0;
          dscore_cs += exp_score1 * global_params[0].dcs_kappa1
                       * sin(M_PI / 180 * (angle2 - global_params[0].dcs_mu1));
          dscore_cs += exp_score2 * global_params[0].dcs_kappa2
                       * sin(M_PI / 180 * (angle2 - global_params[0].dcs_mu2));
          dscore_cs += exp_score3 * global_params[0].dcs_kappa3
                       * sin(M_PI / 180 * (angle2 - global_params[0].dcs_mu3));
          dscore_cs /= (exp_score1 + exp_score2 + exp_score3 + MEST);
          dscore_cs *= wt_dihCS_;

          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[0][pose_index][block2_CA_ind],
              dscore_cs * disulf_ca_dihedral_angle_2.dV_dI);
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[0][pose_index][block2_CB_ind],
              dscore_cs * disulf_ca_dihedral_angle_2.dV_dJ);
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[0][pose_index][block2_S_ind],
              dscore_cs * disulf_ca_dihedral_angle_2.dV_dK);
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[0][pose_index][block1_S_ind],
              dscore_cs * disulf_ca_dihedral_angle_2.dV_dL);
        }

        accumulate<D, Real>::add(V[0][pose_index], score);
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
