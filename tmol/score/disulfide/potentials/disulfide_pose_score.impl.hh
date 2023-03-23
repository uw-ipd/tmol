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
    TView<bool, 2, D> disulfide_conns,
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
    const auto& params = global_params[0];
    const auto& inter_block_connections =
        pose_stack_inter_block_connections[pose_index];
    const auto& block_coord_offset = pose_stack_block_coord_offset[pose_index];
    const auto& block_types = pose_stack_block_type[pose_index];

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

        // Skip this disulfide if the other block doesn't exist
        if (other_block_index == -1) continue;

        int other_block_type_index = block_types[other_block_index];

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

        const Real MEST = exp(-20.0);
        const Real WT_DIH_SS(0.1);
        const Real WT_DIH_CS(0.1);
        const Real WT_ANG(0.1);
        const Real WT_LEN(0.1);
        const Real SHIFT(2.0);

        const Real res1_d_multiplier(1.0);
        const Real res2_d_multiplier(1.0);

        Real score = -SHIFT;

        {  // Calculate Distance
          Real z = (ssdist.V - params.d_location) / params.d_scale;
          Real score_d =
              z * z / 2
              - log(std::erfc(-params.d_shape * z / sqrt(2.0)) + MEST);
          score += WT_LEN * score_d;

          Real dscore_d =
              z / params.d_scale
              - (exp(-0.5 * z * z * params.d_shape * params.d_shape)
                 * sqrt(2.0 / M_PI) * params.d_shape)
                    / (params.d_scale
                           * std::erfc(-params.d_shape * z / sqrt(2.0))
                       + 1.e-12);
          dscore_d *= WT_LEN;
          accumulate<D, Vec<Real, 3>>::add(
              pose_dV_dx[block1_S_ind], dscore_d * ssdist.dV_dA);
          accumulate<D, Vec<Real, 3>>::add(
              pose_dV_dx[block2_S_ind], dscore_d * ssdist.dV_dB);
        }

        {  // Calculate Angles
          Real ang_score(0);
          Real angle1(csang_1.V), angle2(csang_2.V);
          ang_score +=
              WT_ANG
              * (-params.a_logA - params.a_kappa * cos(angle1 - params.a_mu));
          ang_score +=
              WT_ANG
              * (-params.a_logA - params.a_kappa * cos(angle2 - params.a_mu));
          score += ang_score;

          Real dscore_a = params.a_kappa * sin(angle1 - params.a_mu) * WT_ANG;
          Real dscore_b = params.a_kappa * sin(angle2 - params.a_mu) * WT_ANG;
          accumulate<D, Vec<Real, 3>>::add(
              pose_dV_dx[block1_CB_ind], dscore_a * csang_1.dV_dA);
          accumulate<D, Vec<Real, 3>>::add(
              pose_dV_dx[block1_S_ind], dscore_a * csang_1.dV_dB);
          accumulate<D, Vec<Real, 3>>::add(
              pose_dV_dx[block2_S_ind], dscore_a * csang_1.dV_dC);
          accumulate<D, Vec<Real, 3>>::add(
              pose_dV_dx[block2_CB_ind], dscore_b * csang_2.dV_dA);
          accumulate<D, Vec<Real, 3>>::add(
              pose_dV_dx[block2_S_ind], dscore_b * csang_2.dV_dB);
          accumulate<D, Vec<Real, 3>>::add(
              pose_dV_dx[block1_S_ind], dscore_b * csang_2.dV_dC);
        }

        {  // SS dihed
          Real ang_ss(dihed.V), exp_score1(0.0), exp_score2(0.0);
          exp_score1 =
              exp(params.dss_logA1)
              * exp(params.dss_kappa1
                    * cos(res1_d_multiplier * ang_ss - params.dss_mu1));
          exp_score2 =
              exp(params.dss_logA2)
              * exp(params.dss_kappa2
                    * cos(res1_d_multiplier * ang_ss - params.dss_mu2));
          Real score_ss = -log(exp_score1 + exp_score2 + MEST);
          score += WT_DIH_SS * score_ss;

          Real dscore_ss(0.0);
          dscore_ss += exp_score1 * params.dss_kappa1
                       * sin(res1_d_multiplier * dihed.V - params.dss_mu1);
          dscore_ss += exp_score2 * params.dss_kappa2
                       * sin(res1_d_multiplier * dihed.V - params.dss_mu2);
          dscore_ss /= (exp_score1 + exp_score2 + MEST);
          dscore_ss *= WT_DIH_SS;

          accumulate<D, Vec<Real, 3>>::add(
              pose_dV_dx[block1_CB_ind], dscore_ss * dihed.dV_dI);
          accumulate<D, Vec<Real, 3>>::add(
              pose_dV_dx[block1_S_ind], dscore_ss * dihed.dV_dJ);
          accumulate<D, Vec<Real, 3>>::add(
              pose_dV_dx[block2_S_ind], dscore_ss * dihed.dV_dK);
          accumulate<D, Vec<Real, 3>>::add(
              pose_dV_dx[block2_CB_ind], dscore_ss * dihed.dV_dL);
        }

        {  // CB-S dihed
          Real angle1(disulf_ca_dihedral_angle_1.V),
              angle2(disulf_ca_dihedral_angle_2.V);
          Real exp_score1 =
              exp(params.dcs_logA1)
              * exp(params.dcs_kappa1
                    * cos(res1_d_multiplier * angle1 - params.dcs_mu1));
          Real exp_score2 =
              exp(params.dcs_logA2)
              * exp(params.dcs_kappa2
                    * cos(res1_d_multiplier * angle1 - params.dcs_mu2));
          Real exp_score3 =
              exp(params.dcs_logA3)
              * exp(params.dcs_kappa3
                    * cos(res1_d_multiplier * angle1 - params.dcs_mu3));
          score +=
              WT_DIH_CS * (-log(exp_score1 + exp_score2 + exp_score3 + MEST));

          Real dscore_cs = 0.0;
          dscore_cs +=
              exp_score1 * params.dcs_kappa1 * sin(angle1 - params.dcs_mu1);
          dscore_cs +=
              exp_score2 * params.dcs_kappa2 * sin(angle1 - params.dcs_mu2);
          dscore_cs +=
              exp_score3 * params.dcs_kappa3 * sin(angle1 - params.dcs_mu3);
          dscore_cs /= (exp_score1 + exp_score2 + exp_score3 + MEST);
          dscore_cs *= WT_DIH_CS;

          accumulate<D, Vec<Real, 3>>::add(
              pose_dV_dx[block1_CA_ind],
              dscore_cs * disulf_ca_dihedral_angle_1.dV_dI);
          accumulate<D, Vec<Real, 3>>::add(
              pose_dV_dx[block1_CB_ind],
              dscore_cs * disulf_ca_dihedral_angle_1.dV_dJ);
          accumulate<D, Vec<Real, 3>>::add(
              pose_dV_dx[block1_S_ind],
              dscore_cs * disulf_ca_dihedral_angle_1.dV_dK);
          accumulate<D, Vec<Real, 3>>::add(
              pose_dV_dx[block2_S_ind],
              dscore_cs * disulf_ca_dihedral_angle_1.dV_dL);

          exp_score1 =
              exp(params.dcs_logA1)
              * exp(params.dcs_kappa1
                    * cos(res2_d_multiplier * angle2 - params.dcs_mu1));
          exp_score2 =
              exp(params.dcs_logA2)
              * exp(params.dcs_kappa2
                    * cos(res2_d_multiplier * angle2 - params.dcs_mu2));
          exp_score3 =
              exp(params.dcs_logA3)
              * exp(params.dcs_kappa3
                    * cos(res2_d_multiplier * angle2 - params.dcs_mu3));
          score +=
              WT_DIH_CS * (-log(exp_score1 + exp_score2 + exp_score3 + MEST));

          dscore_cs = 0.0;
          dscore_cs +=
              exp_score1 * params.dcs_kappa1 * sin(angle2 - params.dcs_mu1);
          dscore_cs +=
              exp_score2 * params.dcs_kappa2 * sin(angle2 - params.dcs_mu2);
          dscore_cs +=
              exp_score3 * params.dcs_kappa3 * sin(angle2 - params.dcs_mu3);
          dscore_cs /= (exp_score1 + exp_score2 + exp_score3 + MEST);
          dscore_cs *= WT_DIH_CS;

          accumulate<D, Vec<Real, 3>>::add(
              pose_dV_dx[block2_CA_ind],
              dscore_cs * disulf_ca_dihedral_angle_2.dV_dI);
          accumulate<D, Vec<Real, 3>>::add(
              pose_dV_dx[block2_CB_ind],
              dscore_cs * disulf_ca_dihedral_angle_2.dV_dJ);
          accumulate<D, Vec<Real, 3>>::add(
              pose_dV_dx[block2_S_ind],
              dscore_cs * disulf_ca_dihedral_angle_2.dV_dK);
          accumulate<D, Vec<Real, 3>>::add(
              pose_dV_dx[block1_S_ind],
              dscore_cs * disulf_ca_dihedral_angle_2.dV_dL);
        }

        accumulate<D, Real>::add(pose_V, score);
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
