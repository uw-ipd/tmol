#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>

#include <tmol/score/common/dispatch.hh>
#include <tmol/score/common/geom.hh>
#include <tmol/score/common/tuple.hh>

#include "lj.hh"
#include "lk_isotropic.hh"

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <
    template <tmol::Device>
    class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct LJDispatch {
  static auto f(
      TView<Vec<Real, 3>, 1, D> coords_i,
      TView<Int, 1, D> atom_type_i,

      TView<Vec<Real, 3>, 1, D> coords_j,
      TView<Int, 1, D> atom_type_j,

      TView<Real, 2, D> bonded_path_lengths,

      LJTypeParams_targs(1, D),
      LJGlobalParams_args())
      -> std::tuple<
          TPack<int64_t, 2, D>,
          TPack<Real, 1, D>,
          TPack<Vec<Real, 3>, 1, D>,
          TPack<Vec<Real, 3>, 1, D> > {
    Dispatch<D> dispatcher(coords_i.size(0), coords_j.size(0));
    Real threshold_distance = 6.0;
    auto num_Vs = dispatcher.scan(threshold_distance, coords_i, coords_j);

    auto inds_t = TPack<int64_t, 2, D>::empty({num_Vs, 2});
    auto Vs_t = TPack<Real, 1, D>::empty({num_Vs});
    auto dV_dIs_t = TPack<Vec<Real, 3>, 1, D>::empty(num_Vs);
    auto dV_dJs_t = TPack<Vec<Real, 3>, 1, D>::empty(num_Vs);

    auto inds = inds_t.view;
    auto Vs = Vs_t.view;
    auto dV_dIs = dV_dIs_t.view;
    auto dV_dJs = dV_dJs_t.view;

    dispatcher.score([=] EIGEN_DEVICE_FUNC(int o, int i, int j) {
      Int ati = atom_type_i[i];
      Int atj = atom_type_j[j];

      auto dist_r = distance<Real>::V_dV(coords_i[i], coords_j[j]);
      auto& dist = dist_r.V;
      auto& ddist_dI = dist_r.dV_dA;
      auto& ddist_dJ = dist_r.dV_dB;

      Real V, dV_dDist;
      tie(V, dV_dDist) = lj_score_V_dV(
          dist,
          bonded_path_lengths[i][j],
          LJTypeParams_struct(, [ati]),
          LJTypeParams_struct(, [atj]),
          LJGlobalParams_struct());

      inds[o][0] = i;
      inds[o][1] = j;

      Vs[o] = V;
      dV_dIs[o] = dV_dDist * ddist_dI;
      dV_dJs[o] = dV_dDist * ddist_dJ;
    });

    return {inds_t, Vs_t, dV_dIs_t, dV_dJs_t};
  }
};

template <
    template <tmol::Device>
    class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct LKIsotropicDispatch {
  static auto f(
      TView<Vec<Real, 3>, 1, D> coords_i,
      TView<Int, 1, D> atom_type_i,

      TView<Vec<Real, 3>, 1, D> coords_j,
      TView<Int, 1, D> atom_type_j,

      TView<Real, 2, D> bonded_path_lengths,

      LKTypeParams_targs(1, D),
      LJGlobalParams_args())
      -> std::tuple<
          TPack<int64_t, 2, D>,
          TPack<Real, 1, D>,
          TPack<Vec<Real, 3>, 1, D>,
          TPack<Vec<Real, 3>, 1, D> > {
    Dispatch<D> dispatcher(coords_i.size(0), coords_j.size(0));
    Real threshold_distance = 6.0;
    auto num_Vs = dispatcher.scan(threshold_distance, coords_i, coords_j);

    auto inds_t = TPack<int64_t, 2, D>::empty({num_Vs, 2});
    auto Vs_t = TPack<Real, 1, D>::empty({num_Vs});
    auto dV_dIs_t = TPack<Vec<Real, 3>, 1, D>::empty(num_Vs);
    auto dV_dJs_t = TPack<Vec<Real, 3>, 1, D>::empty(num_Vs);

    auto inds = inds_t.view;
    auto Vs = Vs_t.view;
    auto dV_dIs = dV_dIs_t.view;
    auto dV_dJs = dV_dJs_t.view;

    dispatcher.score([=] EIGEN_DEVICE_FUNC(int o, int i, int j) {
      Int ati = atom_type_i[i];
      Int atj = atom_type_j[j];

      auto dist_r = distance<Real>::V_dV(coords_i[i], coords_j[j]);
      auto& dist = dist_r.V;
      auto& ddist_dI = dist_r.dV_dA;
      auto& ddist_dJ = dist_r.dV_dB;

      Real V, dV_dDist;
      tie(V, dV_dDist) = lk_isotropic_score_V_dV(
          dist,
          bonded_path_lengths[i][j],
          LKTypeParams_struct(, [ati]),
          LKTypeParams_struct(, [atj]),
          LJGlobalParams_struct());

      inds[o][0] = i;
      inds[o][1] = j;

      Vs[o] = V;
      dV_dIs[o] = dV_dDist * ddist_dI;
      dV_dJs[o] = dV_dDist * ddist_dJ;
    });

    return {inds_t, Vs_t, dV_dIs_t, dV_dJs_t};
  }
};

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
