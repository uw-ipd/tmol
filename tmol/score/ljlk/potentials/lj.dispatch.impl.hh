#pragma once

#include <nvToolsExt.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>

#include <tmol/score/common/geom.hh>
#include <tmol/score/common/tuple.hh>

#include "lj.dispatch.hh"
#include "lj.hh"

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

#define def auto EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE

template <tmol::Device D, typename T, class Enable = void>
struct accumulate {};

template <typename T>
struct accumulate<
    tmol::Device::CPU,
    T,
    typename std::enable_if<std::is_arithmetic<T>::value>::type> {
  static def add(T& target, const T& val)->void { target += val; }
};  // namespace potentials

template <tmol::Device D, int N, typename T>
struct accumulate<
    D,
    Eigen::Matrix<T, N, 1>,
    typename std::enable_if<std::is_arithmetic<T>::value>::type> {
  typedef Eigen::Matrix<T, N, 1> V;

  static def add(V& target, const V& val)->void {
#pragma unroll
    for (int i = 0; i < N; i++) {
      accumulate<D, T>::add(target[i], val[i]);
    }
  }
};  // namespace potentials

#ifdef __CUDACC__

template <typename T>
struct accumulate<
    tmol::Device::CUDA,
    T,
    typename std::enable_if<std::is_arithmetic<T>::value>::type> {
  static def add(T& target, const T& val)->void { atomicAdd(&target, val); }
};

#endif

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

    LJTypeParamTensors<Real, D> type_params,
    LJGlobalParams<Real> global_params)
    -> std::tuple<
        TPack<Real, 1, D>,
        TPack<Vec<Real, 3>, 1, D>,
        TPack<Vec<Real, 3>, 1, D>> {
  nvtxRangePushA(__FUNCTION__);

  nvtxRangePushA("output_allocate");
  auto V_t = TPack<Real, 1, D>::zeros({1});
  auto dV_dI_t = TPack<Vec<Real, 3>, 1, D>::zeros({coords_i.size(0)});
  auto dV_dJ_t = TPack<Vec<Real, 3>, 1, D>::zeros({coords_j.size(0)});

  auto V = V_t.view;
  auto dV_dI = dV_dI_t.view;
  auto dV_dJ = dV_dJ_t.view;
  nvtxRangePop();

  nvtxRangePushA("dispatch::score");
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
            global_params);

        accumulate<D, Real>::add(V[0], lj.V);
        accumulate<D, Vec<Real, 3>>::add(dV_dI[i], lj.dV_ddist * ddist_dI);
        accumulate<D, Vec<Real, 3>>::add(dV_dJ[j], lj.dV_ddist * ddist_dJ);
      });
  nvtxRangePop();

  nvtxRangePop();

  return {V_t, dV_dI_t, dV_dJ_t};
};

#undef def

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
