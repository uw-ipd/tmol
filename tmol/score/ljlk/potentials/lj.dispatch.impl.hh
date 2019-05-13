#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/geom.hh>
#include <tmol/score/common/tuple.hh>

#include "lj.dispatch.hh"
#include "lj.hh"

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
auto LJDispatch<Dispatch, D, Real, Int>::f(
    TView<Vec<Real, 3>, 1, D> coords_i,
    TView<Int, 1, D> atom_type_i,

    TView<Vec<Real, 3>, 1, D> coords_j,
    TView<Int, 1, D> atom_type_j,

    TView<Real, 2, D> bonded_path_lengths,
    TView<LJTypeParams<Real>, 1, D> type_params,
    TView<LJGlobalParams<Real>, 1, D> global_params)
    -> std::tuple<
        TPack<Real, 1, D>,
        TPack<Vec<Real, 3>, 1, D>,
        TPack<Vec<Real, 3>, 1, D>> {
  NVTXRange _function(__FUNCTION__);

  nvtx_range_push("dispatch::score");
  auto V_t = TPack<Real, 1, D>::zeros({1});
  auto dV_dI_t = TPack<Vec<Real, 3>, 1, D>::zeros({coords_i.size(0)});
  auto dV_dJ_t = TPack<Vec<Real, 3>, 1, D>::zeros({coords_j.size(0)});

  auto V = V_t.view;
  auto dV_dI = dV_dI_t.view;
  auto dV_dJ = dV_dJ_t.view;
  nvtx_range_pop();

  nvtx_range_push("dispatch::score");
  Real threshold_distance = 6.0;
  Dispatch<D>::forall_pairs(
      threshold_distance,
      coords_i,
      coords_j,
      [=] EIGEN_DEVICE_FUNC(int i, int j) {
        Int ati = atom_type_i[i];
        Int atj = atom_type_j[j];

        auto dist_r = distance<Real>::V_dV(coords_i[i], coords_j[j]);
        auto& dist = dist_r.V;
        auto& ddist_dI = dist_r.dV_dA;
        auto& ddist_dJ = dist_r.dV_dB;

        auto lj = lj_score<Real>::V_dV(
            dist,
            bonded_path_lengths[i][j],
            type_params[ati],
            type_params[atj],
            global_params[0]);

        accumulate<D, Real>::add(V[0], lj.V);
        accumulate<D, Vec<Real, 3>>::add(dV_dI[i], lj.dV_ddist * ddist_dI);
        accumulate<D, Vec<Real, 3>>::add(dV_dJ[j], lj.dV_ddist * ddist_dJ);
      });
  nvtx_range_pop();

  nvtx_range_pop();

  return {V_t, dV_dI_t, dV_dJ_t};
};

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
