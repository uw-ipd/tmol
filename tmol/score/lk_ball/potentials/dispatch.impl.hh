#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/cuda/stream.hh>
#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/dispatch.hh>
#include <tmol/score/common/geom.hh>

#include <tmol/score/ljlk/potentials/params.hh>

#include "lk_ball.hh"
#include "params.hh"

namespace tmol {
namespace score {
namespace lk_ball {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <
    template <tmol::Device>
    class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct LKBallDispatch {
  static auto forward(
      TView<Vec<Real, 3>, 1, D> coords_i,
      TView<Int, 1, D> atom_type_i,
      TView<Vec<Real, 3>, 2, D> waters_i,

      TView<Vec<Real, 3>, 1, D> coords_j,
      TView<Int, 1, D> atom_type_j,
      TView<Vec<Real, 3>, 2, D> waters_j,

      TView<Real, 2, D> bonded_path_lengths,

      TView<LKBallTypeParams<Real>, 1, D> type_params,
      TView<LKBallGlobalParams<Real>, 1, D> global_params)
      -> TPack<Real, 1, D> {
    NVTXRange _function(__FUNCTION__);

    // Retrieve the stream that the gen_waters step used.
    // It should still be the current stream.
    auto stream = utility::cuda::get_current_cuda_stream();

    NVTXRange _allocate("allocate");
    auto Vs_t = TPack<Real, 1, D>::zeros({4});
    auto Vs = Vs_t.view;
    _allocate.exit();


    NVTXRange _score("score");
    Real threshold_distance = 6.0;  // fd this should be a global param
    Dispatch<D>::forall_pairs(
        threshold_distance,
        coords_i,
        coords_j,
        [=] EIGEN_DEVICE_FUNC(int i, int j) {
          Int ati = atom_type_i[i];
          Int atj = atom_type_j[j];

          // fd: should '4' (#wats) be a templated parameter?
          Eigen::Matrix<Real, 4, 3> wmat_i;
          Eigen::Matrix<Real, 4, 3> wmat_j;

          for (int wi = 0; wi < 4; wi++) {
            wmat_i.row(wi) = waters_i[i][wi];
            wmat_j.row(wi) = waters_j[j][wi];
          }

          auto score = lk_ball_score<Real, 4>::V(
              coords_i[i],
              coords_j[j],
              wmat_i,
              wmat_j,
              bonded_path_lengths[i][j],
              type_params[ati],
              type_params[atj],
              global_params[0]);

          common::accumulate<D, Real>::add(Vs[0], score.lkball_iso);
          common::accumulate<D, Real>::add(Vs[1], score.lkball);
          common::accumulate<D, Real>::add(Vs[2], score.lkbridge);
          common::accumulate<D, Real>::add(Vs[3], score.lkbridge_uncpl);
        },
        stream);
    _score.exit();

    // Finally, restore the current stream to the default stream.
    utility::cuda::set_default_cuda_stream();

    return Vs_t;
  }

  static auto backward(
      TView<Real, 1, D> dTdV,
      TView<Vec<Real, 3>, 1, D> coords_i,
      TView<Int, 1, D> atom_type_i,
      TView<Vec<Real, 3>, 2, D> waters_i,

      TView<Vec<Real, 3>, 1, D> coords_j,
      TView<Int, 1, D> atom_type_j,
      TView<Vec<Real, 3>, 2, D> waters_j,

      TView<Real, 2, D> bonded_path_lengths,

      TView<LKBallTypeParams<Real>, 1, D> type_params,
      TView<LKBallGlobalParams<Real>, 1, D> global_params)
      -> std::tuple<
          TPack<Vec<Real, 3>, 1, D>,
          TPack<Vec<Real, 3>, 1, D>,
          TPack<Vec<Real, 3>, 2, D>,
          TPack<Vec<Real, 3>, 2, D>> {
    NVTXRange _function(__FUNCTION__);

    NVTXRange _alloc("allocate");
    // deriv w.r.t. heavyatom position
    auto dV_dI_t = TPack<Vec<Real, 3>, 1, D>::zeros({coords_i.size(0)});
    auto dV_dJ_t = TPack<Vec<Real, 3>, 1, D>::zeros({coords_j.size(0)});

    // deriv w.r.t. water position
    auto dW_dI_t = TPack<Vec<Real, 3>, 2, D>::zeros({coords_i.size(0), 4});
    auto dW_dJ_t = TPack<Vec<Real, 3>, 2, D>::zeros({coords_j.size(0), 4});

    auto dV_dI = dV_dI_t.view;
    auto dV_dJ = dV_dJ_t.view;
    auto dW_dI = dW_dI_t.view;
    auto dW_dJ = dW_dJ_t.view;
    _alloc.exit();

    NVTXRange _work("dscore");
    auto defstream = utility::cuda::get_default_stream();
    Real threshold_distance = 6.0;  // fd this should be a global param
    Dispatch<D>::forall_pairs(
        threshold_distance,
        coords_i,
        coords_j,
        [=] EIGEN_DEVICE_FUNC(int i, int j) {
          Int ati = atom_type_i[i];
          Int atj = atom_type_j[j];

          // fd: should '4' (#wats) be a templated parameter?
          Eigen::Matrix<Real, 4, 3> wmat_i;
          Eigen::Matrix<Real, 4, 3> wmat_j;

          for (int wi = 0; wi < 4; wi++) {
            wmat_i.row(wi) = waters_i[i][wi];
            wmat_j.row(wi) = waters_j[j][wi];
          }

          auto dV = lk_ball_score<Real, 4>::dV(
              coords_i[i],
              coords_j[j],
              wmat_i,
              wmat_j,
              bonded_path_lengths[i][j],
              type_params[ati],
              type_params[atj],
              global_params[0]);

          common::accumulate<D, Vec<Real, 3>>::add(
              dV_dI[i], dTdV[0] * dV.dI.d_lkball_iso);
          common::accumulate<D, Vec<Real, 3>>::add(
              dV_dJ[j], dTdV[0] * dV.dJ.d_lkball_iso);
          common::accumulate<D, Vec<Real, 3>>::add(
              dV_dI[i], dTdV[1] * dV.dI.d_lkball);
          common::accumulate<D, Vec<Real, 3>>::add(
              dV_dJ[j], dTdV[1] * dV.dJ.d_lkball);
          common::accumulate<D, Vec<Real, 3>>::add(
              dV_dI[i], dTdV[2] * dV.dI.d_lkbridge);
          common::accumulate<D, Vec<Real, 3>>::add(
              dV_dJ[j], dTdV[2] * dV.dJ.d_lkbridge);
          common::accumulate<D, Vec<Real, 3>>::add(
              dV_dI[i], dTdV[3] * dV.dI.d_lkbridge_uncpl);
          common::accumulate<D, Vec<Real, 3>>::add(
              dV_dJ[j], dTdV[3] * dV.dJ.d_lkbridge_uncpl);

          for (int wi = 0; wi < 4; wi++) {
            common::accumulate<D, Vec<Real, 3>>::add(
                dW_dI[i][wi], dTdV[0] * dV.dWI.d_lkball_iso.row(wi));
            common::accumulate<D, Vec<Real, 3>>::add(
                dW_dJ[j][wi], dTdV[0] * dV.dWJ.d_lkball_iso.row(wi));
            common::accumulate<D, Vec<Real, 3>>::add(
                dW_dI[i][wi], dTdV[1] * dV.dWI.d_lkball.row(wi));
            common::accumulate<D, Vec<Real, 3>>::add(
                dW_dJ[j][wi], dTdV[1] * dV.dWJ.d_lkball.row(wi));
            common::accumulate<D, Vec<Real, 3>>::add(
                dW_dI[i][wi], dTdV[2] * dV.dWI.d_lkbridge.row(wi));
            common::accumulate<D, Vec<Real, 3>>::add(
                dW_dJ[j][wi], dTdV[2] * dV.dWJ.d_lkbridge.row(wi));
            common::accumulate<D, Vec<Real, 3>>::add(
                dW_dI[i][wi], dTdV[3] * dV.dWI.d_lkbridge_uncpl.row(wi));
            common::accumulate<D, Vec<Real, 3>>::add(
                dW_dJ[j][wi], dTdV[3] * dV.dWJ.d_lkbridge_uncpl.row(wi));
          }
        },
	defstream);
    _work.exit();

    return {dV_dI_t, dV_dJ_t, dW_dI_t, dW_dJ_t};
  }
};

}  // namespace potentials
}  // namespace lk_ball
}  // namespace score
}  // namespace tmol
