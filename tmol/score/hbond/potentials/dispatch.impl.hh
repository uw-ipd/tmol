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

    TView<HBondPairParams<Real>, 2, Dev> pair_params,
    TView<HBondPolynomials<double>, 2, Dev> pair_polynomials,
    TView<HBondGlobalParams<Real>, 1, Dev> global_params)
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

  AT_ASSERTM(
      pair_params.size(0) == pair_polynomials.size(0),
      "Disagreement on number of donor types.");
  AT_ASSERTM(
      pair_params.size(1) == pair_polynomials.size(1),
      "Disagreement on number of acceptor types.");

  AT_ASSERTM(
      global_params.size(0) == 1, "Invalid number of global parameters.");

  auto V_t = TPack<Real, 1, Dev>::zeros({1});
  auto dV_d_don_t = TPack<Vec<Real, 3>, 1, Dev>::zeros({donor_coords.size(0)});
  auto dV_d_acc_t =
      TPack<Vec<Real, 3>, 1, Dev>::zeros({acceptor_coords.size(0)});

  auto V = V_t.view;
  auto dV_d_don = dV_d_don_t.view;
  auto dV_d_acc = dV_d_acc_t.view;

  Real _threshold_distance = 6.0;  // what about the global threshold distance?

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

            pair_params[dt][at],
            pair_polynomials[dt][at],
            global_params[0]);

        accumulate<Dev, Real>::add(V[0], hbond.V);

        accumulate<Dev, Vec<Real, 3>>::add(dV_d_don[D[di]], hbond.dV_dD);
        accumulate<Dev, Vec<Real, 3>>::add(dV_d_don[H[di]], hbond.dV_dH);

        accumulate<Dev, Vec<Real, 3>>::add(dV_d_acc[A[ai]], hbond.dV_dA);
        accumulate<Dev, Vec<Real, 3>>::add(dV_d_acc[B[ai]], hbond.dV_dB);
        accumulate<Dev, Vec<Real, 3>>::add(dV_d_acc[B0[ai]], hbond.dV_dB0);
      });

  return {V_t, dV_d_don_t, dV_d_acc_t};
}

// template <
//     template <tmol::Device>
//     class Dispatch,
//     tmol::Device Dev,
//     typename Real,
//     typename Int>
// auto HBondDispatch<Dispatch, Dev, Real, Int>::backward(
//     TView<Real, 1, Dev> dTdV,
//     TView<Real, 2, Dev> dV_d_don,
//     TView<Real, 2, Dev> dV_d_acc)
//     -> std::tuple<
//         TPack<Real, 2, Dev>,
//         TPack<Real, 2, Dev>>
// {
//   ndon = dV_d_don.shape(0);
//   nacc = dV_d_acc.shape(0);
//   auto dT_d_don_t = TPack<Real, 2, Dev>::empty(ndon, 3);
//   auto dT_d_acc_t = TPack<Real, 2, Dev>::empty(nacc, 3);
//   NaiveDis
// }

}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol
