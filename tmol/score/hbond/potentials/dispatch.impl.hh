#pragma once

#include <cmath>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <tuple>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/tuple.hh>

#include <tmol/score/hbond/potentials/potentials.hh>
#include "dispatch.hh"

namespace tmol {
namespace score {
namespace hbond {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <
    template <tmol::Device>
    class Dispatch,
    tmol::Device Dev,
    typename Real,
    typename Int>
auto HBondDispatch<Dispatch, Dev, Real, Int>::f(
    TView<Vec<Real, 3>, 1, Dev> donor_coords,
    TView<Vec<Real, 3>, 1, Dev> acceptor_coords,

    TView<int64_t, 1, Dev> D,
    TView<int64_t, 1, Dev> H,
    TView<Int, 1, Dev> donor_type,

    TView<int64_t, 1, Dev> A,
    TView<int64_t, 1, Dev> B,
    TView<int64_t, 1, Dev> B0,
    TView<Int, 1, Dev> acceptor_type,

    TView<Int, 2, Dev> acceptor_hybridization,
    TView<Real, 2, Dev> acceptor_weight,
    TView<Real, 2, Dev> donor_weight,

    TView<Vec<double, 11>, 2, Dev> AHdist_coeffs,
    TView<Vec<double, 2>, 2, Dev> AHdist_range,
    TView<Vec<double, 2>, 2, Dev> AHdist_bound,

    TView<Vec<double, 11>, 2, Dev> cosBAH_coeffs,
    TView<Vec<double, 2>, 2, Dev> cosBAH_range,
    TView<Vec<double, 2>, 2, Dev> cosBAH_bound,

    TView<Vec<double, 11>, 2, Dev> cosAHD_coeffs,
    TView<Vec<double, 2>, 2, Dev> cosAHD_range,
    TView<Vec<double, 2>, 2, Dev> cosAHD_bound,

    TView<Real, 1, Dev> hb_sp2_range_span,
    TView<Real, 1, Dev> hb_sp2_BAH180_rise,
    TView<Real, 1, Dev> hb_sp2_outer_width,
    TView<Real, 1, Dev> hb_sp3_softmax_fade,
    TView<Real, 1, Dev> threshold_distance)
    -> std::tuple<
        TPack<Real, 1, Dev>,
        TPack<Vec<Real, 3>, 1, Dev>,
        TPack<Vec<Real, 3>, 1, Dev>> {
  AT_ASSERTM(
      donor_type.size(0) == D.size(0), "Invalid donor coordinate shapes.");
  AT_ASSERTM(
      donor_type.size(0) == H.size(0), "Invalid donor coordinate shapes.");

  AT_ASSERTM(
      acceptor_type.size(0) == A.size(0),
      "Invalid acceptor coordinate shapes.");
  AT_ASSERTM(
      acceptor_type.size(0) == B.size(0),
      "Invalid acceptor coordinate shapes.");
  AT_ASSERTM(
      acceptor_type.size(0) == B0.size(0),
      "Invalid acceptor coordinate shapes.");

  auto V_t = TPack<Real, 1, Dev>::empty({1});
  auto dV_d_don_t = TPack<Vec<Real, 3>, 1, Dev>::empty({donor_coords.size(0)});
  auto dV_d_acc_t =
      TPack<Vec<Real, 3>, 1, Dev>::empty({acceptor_coords.size(0)});

  auto V = V_t.view;
  auto dV_d_don = dV_d_don_t.view;
  auto dV_d_acc = dV_d_acc_t.view;

  Real _threshold_distance = 6.0;

  Dispatch<Dev>::forall_idx_pairs(
      _threshold_distance,

      donor_coords,
      acceptor_coords,

      D,
      A,

      [=] EIGEN_DEVICE_FUNC(int di, int ai) {
        int dt = donor_type[di];
        int at = acceptor_type[ai];

        int dind = D[di];
        int rind = D[di];

        auto hbond = hbond_score<Real, Int>::V_dV(
            donor_coords[D[di]],
            donor_coords[H[di]],

            acceptor_coords[A[ai]],
            acceptor_coords[B[ai]],
            acceptor_coords[B0[ai]],

            acceptor_hybridization[dt][at],
            acceptor_weight[dt][at],
            donor_weight[dt][at],

            AHdist_coeffs[dt][at],
            AHdist_range[dt][at],
            AHdist_bound[dt][at],

            cosBAH_coeffs[dt][at],
            cosBAH_range[dt][at],
            cosBAH_bound[dt][at],

            cosAHD_coeffs[dt][at],
            cosAHD_range[dt][at],
            cosAHD_bound[dt][at],

            hb_sp2_range_span[1],
            hb_sp2_BAH180_rise[1],
            hb_sp2_outer_width[1],
            hb_sp3_softmax_fade[1]);

        accumulate<Dev, Real>::add(V[0], hbond.V);

        accumulate<Dev, Vec<Real, 3>>::add(dV_d_don[D[di]], hbond.dV_dD);
        accumulate<Dev, Vec<Real, 3>>::add(dV_d_don[H[di]], hbond.dV_dH);

        accumulate<Dev, Vec<Real, 3>>::add(dV_d_acc[A[ai]], hbond.dV_dA);
        accumulate<Dev, Vec<Real, 3>>::add(dV_d_acc[B[ai]], hbond.dV_dB);
        accumulate<Dev, Vec<Real, 3>>::add(dV_d_acc[B0[ai]], hbond.dV_dB0);
      });

  return {V_t, dV_d_don_t, dV_d_acc_t};
}

}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol
