#pragma once

#include <cmath>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <tuple>

#include <ATen/cuda/CUDAStream.h>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/score/common/tuple.hh>

#include <tmol/score/elec/potentials/potentials.hh>

namespace tmol {
namespace score {
namespace elec {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <
    template <tmol::Device>
    class Dispatch,
    tmol::Device Dev,
    typename Real,
    typename Int>
struct ElecDispatch {
  static auto f(
      TView<Vec<Real, 3>, 1, Dev> x_i,
      TView<Real, 1, Dev> e_i,
      TView<Vec<Real, 3>, 1, Dev> x_j,
      TView<Real, 1, Dev> e_j,
      TView<Real, 2, Dev> bonded_path_lengths,
      Real D,
      Real D0,
      Real S,
      Real min_dis,
      Real max_dis)
      -> std::tuple<
          TPack<int64_t, 2, Dev>,
          TPack<Real, 1, Dev>,
          TPack<Vec<Real, 3>, 1, Dev>,
          TPack<Vec<Real, 3>, 1, Dev> > {
    Dispatch<Dev> dispatcher(e_i.size(0), e_j.size(0));
    Real threshold_distance = 6.0;

    auto stream1 =
        at::cuda::getStreamFromPool(false, Dev == tmol::Device::CUDA ? 0 : -1);
    at::cuda::setCurrentCUDAStream(stream1);

    auto num_Vs = dispatcher.scan(threshold_distance, x_i, x_j, &stream1);

    auto inds_t = TPack<int64_t, 2, Dev>::empty({num_Vs, 2});
    auto Vs_t = TPack<Real, 1, Dev>::empty({num_Vs});
    auto dV_dIs_t = TPack<Vec<Real, 3>, 1, Dev>::empty(num_Vs);
    auto dV_dJs_t = TPack<Vec<Real, 3>, 1, Dev>::empty(num_Vs);

    auto inds = inds_t.view;
    auto Vs = Vs_t.view;
    auto dV_dIs = dV_dIs_t.view;
    auto dV_dJs = dV_dJs_t.view;

    dispatcher.score([=] EIGEN_DEVICE_FUNC(int o, int i, int j) {
      auto dist_r = distance<Real>::V_dV(x_i[i], x_j[j]);
      auto& dist = dist_r.V;
      auto& ddist_dI = dist_r.dV_dA;
      auto& ddist_dJ = dist_r.dV_dB;

      Real V, dV_dDist;
      tie(V, dV_dDist) = elec_delec_ddist(
          dist,
          e_i[i],
          e_j[j],
          bonded_path_lengths[i][j],
          D,
          D0,
          S,
          min_dis,
          max_dis);

      inds[o][0] = i;
      inds[o][1] = j;

      Vs[o] = V;
      dV_dIs[o] = dV_dDist * ddist_dI;
      dV_dJs[o] = dV_dDist * ddist_dJ;
    });

    auto default_stream =
        at::cuda::getDefaultCUDAStream(Dev == tmol::Device::CUDA ? 0 : -1);
    at::cuda::setCurrentCUDAStream(default_stream);

    return {inds_t, Vs_t, dV_dIs_t, dV_dJs_t};
  }
};

}  // namespace potentials
}  // namespace elec
}  // namespace score
}  // namespace tmol
