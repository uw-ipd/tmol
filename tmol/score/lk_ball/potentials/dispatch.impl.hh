#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>

#include <tmol/score/common/dispatch.hh>
#include <tmol/score/common/geom.hh>

#include "datatypes.hh"
#include "lk_ball.hh"

namespace tmol {
namespace score {
namespace lk_ball {
namespace potentials {

using tmol::score::ljlk::potentials::LJGlobalParams;
using tmol::score::ljlk::potentials::LKTypeParamTensors;

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <
    template <tmol::Device>
    class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct LKBallDispatch {
  static auto f(
      TView<Vec<Real, 3>, 1, D> coords_i,
      TView<Vec<Real, 3>, 1, D> coords_j,

      TView<Vec<Real, 3>, 2, D> waters_i,
      TView<Vec<Real, 3>, 2, D> waters_j,

      TView<Int, 1, D> atom_type_i,
      TView<Int, 1, D> atom_type_j,

      TView<Real, 2, D> bonded_path_lengths,
      LKTypeParamTensors<Real, D> type_params,
      LKBallGlobalParameters<Real, D> global_lkb_params,
      LJGlobalParams<Real> global_lj_params)
      -> std::tuple<TPack<int64_t, 2, D>, TPack<Real, 2, D>> {
    Dispatch<D> dispatcher(coords_i.size(0), coords_j.size(0));
    Real threshold_distance = 6.0;
    auto num_Vs = dispatcher.scan(threshold_distance, coords_i, coords_j);

    auto inds_t = TPack<int64_t, 2, D>::empty({num_Vs, 2});
    auto Vs_t = TPack<Real, 2, D>::empty({num_Vs, 4});

    auto inds = inds_t.view;
    auto Vs = Vs_t.view;

    dispatcher.score([=] EIGEN_DEVICE_FUNC(int o, int i, int j) {
      inds[o][0] = i;
      inds[o][1] = j;
      Int ati = atom_type_i[i];
      Int atj = atom_type_j[j];

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
          global_lkb_params.lkb_water_dist,
          type_params[ati],
          type_params[atj],
          global_lj_params);

      Vs[o][0] = score.lkball_iso;
      Vs[o][1] = score.lkball;
      Vs[o][2] = score.lkbridge;
      Vs[o][3] = score.lkbridge_uncpl;
    });

    return {inds_t, Vs_t};
  }
};

}  // namespace potentials
}  // namespace lk_ball
}  // namespace score
}  // namespace tmol
