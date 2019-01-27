#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
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
      -> std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> {
    using tmol::new_tensor;

    Dispatch<D> dispatcher(coords_i.size(0), coords_j.size(0));
    Real threshold_distance = 6.0;
    auto num_Vs = dispatcher.scan(threshold_distance, coords_i, coords_j);

    at::Tensor inds_t;
    TView<int64_t, 2, D> inds;
    std::tie(inds_t, inds) = new_tensor<int64_t, 2, D>({num_Vs, 2});

    at::Tensor Vs_t;
    TView<Real, 1, D> Vs;
    std::tie(Vs_t, Vs) = new_tensor<Real, 1, D>(num_Vs);

    at::Tensor dV_dIs_t;
    TView<Vec<Real, 3>, 1, D> dV_dIs;
    std::tie(dV_dIs_t, dV_dIs) = new_tensor<Vec<Real, 3>, 1, D>(num_Vs);

    at::Tensor dV_dJs_t;
    TView<Vec<Real, 3>, 1, D> dV_dJs;
    std::tie(dV_dJs_t, dV_dJs) = new_tensor<Vec<Real, 3>, 1, D>(num_Vs);

    dispatcher.score([=] EIGEN_DEVICE_FUNC(int o, int i, int j) {
      Int ati = atom_type_i[i];
      Int atj = atom_type_j[j];

      Real Dist;
      Vec<Real, 3> dDist_dI, dDist_dJ;
      tie(Dist, dDist_dI, dDist_dJ) = distance_V_dV(coords_i[i], coords_j[j]);

      Real V, dV_dDist;
      tie(V, dV_dDist) = lj_score_V_dV(
          Dist,
          bonded_path_lengths[i][j],
          LJTypeParams_struct(, [ati]),
          LJTypeParams_struct(, [atj]),
          LJGlobalParams_struct());

      inds[o][0] = i;
      inds[o][1] = j;

      Vs[o] = V;
      dV_dIs[o] = dV_dDist * dDist_dI;
      dV_dJs[o] = dV_dDist * dDist_dJ;
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
      -> std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> {
    using tmol::new_tensor;

    Dispatch<D> dispatcher(coords_i.size(0), coords_j.size(0));
    Real threshold_distance = 6.0;
    auto num_Vs = dispatcher.scan(threshold_distance, coords_i, coords_j);

    at::Tensor inds_t;
    TView<int64_t, 2, D> inds;
    std::tie(inds_t, inds) = new_tensor<int64_t, 2, D>({num_Vs, 2});

    at::Tensor Vs_t;
    TView<Real, 1, D> Vs;
    std::tie(Vs_t, Vs) = new_tensor<Real, 1, D>(num_Vs);

    at::Tensor dV_dIs_t;
    TView<Vec<Real, 3>, 1, D> dV_dIs;
    std::tie(dV_dIs_t, dV_dIs) = new_tensor<Vec<Real, 3>, 1, D>(num_Vs);

    at::Tensor dV_dJs_t;
    TView<Vec<Real, 3>, 1, D> dV_dJs;
    std::tie(dV_dJs_t, dV_dJs) = new_tensor<Vec<Real, 3>, 1, D>(num_Vs);

    dispatcher.score([=] EIGEN_DEVICE_FUNC(int o, int i, int j) {
      Int ati = atom_type_i[i];
      Int atj = atom_type_j[j];

      Real Dist;
      Vec<Real, 3> dDist_dI, dDist_dJ;
      tie(Dist, dDist_dI, dDist_dJ) = distance_V_dV(coords_i[i], coords_j[j]);

      Real V, dV_dDist;
      tie(V, dV_dDist) = lk_isotropic_score_V_dV(
          Dist,
          bonded_path_lengths[i][j],
          LKTypeParams_struct(, [ati]),
          LKTypeParams_struct(, [atj]),
          LJGlobalParams_struct());

      inds[o][0] = i;
      inds[o][1] = j;

      Vs[o] = V;
      dV_dIs[o] = dV_dDist * dDist_dI;
      dV_dJs[o] = dV_dDist * dDist_dJ;
    });

    return {inds_t, Vs_t, dV_dIs_t, dV_dJs_t};
  }
};

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
