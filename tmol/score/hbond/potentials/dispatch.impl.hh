#pragma once

#include <cmath>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <tuple>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/utility/nvtx.hh>

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
    TView<Vec<Real, 3>, 2, Dev> donor_coords,
    TView<Vec<Real, 3>, 2, Dev> acceptor_coords,

    TView<int64_t, 2, Dev> D,
    TView<int64_t, 2, Dev> H,
    TView<Int, 2, Dev> donor_type,

    TView<int64_t, 2, Dev> A,
    TView<int64_t, 2, Dev> B,
    TView<int64_t, 2, Dev> B0,
    TView<Int, 2, Dev> acceptor_type,

    TView<HBondPairParams<Real>, 2, Dev> pair_params,
    TView<HBondPolynomials<double>, 2, Dev> pair_polynomials,
    TView<HBondGlobalParams<Real>, 1, Dev> global_params)
    -> std::tuple<
        TPack<Real, 1, Dev>,
        TPack<Vec<Real, 3>, 2, Dev>,
        TPack<Vec<Real, 3>, 2, Dev>> {
  AT_ASSERTM(
      donor_type.size(0) == acceptor_type.size(0),
      "Mismatch in number of stack dimensions");
  AT_ASSERTM(
      donor_type.size(0) == donor_coords.size(0),
      "Mismatch in number of stack dimensions");
  AT_ASSERTM(
      acceptor_type.size(0) == acceptor_coords.size(0),
      "Mismatch in number of stack dimensions");

  AT_ASSERTM(
      donor_type.size(0) == D.size(0), "Invalid donor coordinate shapes.");
  AT_ASSERTM(
      donor_type.size(0) == H.size(0), "Invalid donor coordinate shapes.");

  AT_ASSERTM(
      donor_type.size(1) == D.size(1), "Invalid donor coordinate shapes.");
  AT_ASSERTM(
      donor_type.size(1) == H.size(1), "Invalid donor coordinate shapes.");

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
      acceptor_type.size(1) == A.size(1),
      "Invalid acceptor coordinate shapes.");
  AT_ASSERTM(
      acceptor_type.size(1) == B.size(1),
      "Invalid acceptor coordinate shapes.");
  AT_ASSERTM(
      acceptor_type.size(1) == B0.size(1),
      "Invalid acceptor coordinate shapes.");

  AT_ASSERTM(
      pair_params.size(0) == pair_polynomials.size(0),
      "Disagreement on number of donor types.");
  AT_ASSERTM(
      pair_params.size(1) == pair_polynomials.size(1),
      "Disagreement on number of acceptor types.");

  AT_ASSERTM(
      global_params.size(0) == 1, "Invalid number of global parameters.");

  nvtx_range_push("hbond alloc");
  int nstacks = donor_coords.size(0);
  auto V_t = TPack<Real, 1, Dev>::zeros({nstacks});
  auto dV_d_don_t =
      TPack<Vec<Real, 3>, 2, Dev>::zeros({nstacks, donor_coords.size(1)});
  auto dV_d_acc_t =
      TPack<Vec<Real, 3>, 2, Dev>::zeros({nstacks, acceptor_coords.size(1)});

  auto V = V_t.view;
  auto dV_d_don = dV_d_don_t.view;
  auto dV_d_acc = dV_d_acc_t.view;

  Real _threshold_distance = 6.0;  // what about the global threshold distance?

  nvtx_range_pop();
  nvtx_range_push("hbond eval");
  Dispatch<Dev>::forall_stacked_idx_pairs(
      _threshold_distance,

      donor_coords,
      acceptor_coords,

      D,
      A,

      [=] EIGEN_DEVICE_FUNC(int stack, int di, int ai) {
        int dt = donor_type[stack][di];
        int at = acceptor_type[stack][ai];

        auto hbond = hbond_score<Real, Int>::V_dV(
            donor_coords[stack][D[stack][di]],
            donor_coords[stack][H[stack][di]],

            acceptor_coords[stack][A[stack][ai]],
            acceptor_coords[stack][B[stack][ai]],
            acceptor_coords[stack][B0[stack][ai]],

            pair_params[dt][at],
            pair_polynomials[dt][at],
            global_params[0]);

        accumulate<Dev, Real>::add(V[stack], hbond.V);

        accumulate<Dev, Vec<Real, 3>>::add(
            dV_d_don[stack][D[stack][di]], hbond.dV_dD);
        accumulate<Dev, Vec<Real, 3>>::add(
            dV_d_don[stack][H[stack][di]], hbond.dV_dH);

        accumulate<Dev, Vec<Real, 3>>::add(
            dV_d_acc[stack][A[stack][ai]], hbond.dV_dA);
        accumulate<Dev, Vec<Real, 3>>::add(
            dV_d_acc[stack][B[stack][ai]], hbond.dV_dB);
        accumulate<Dev, Vec<Real, 3>>::add(
            dV_d_acc[stack][B0[stack][ai]], hbond.dV_dB0);
      });

  nvtx_range_pop();
  return {V_t, dV_d_don_t, dV_d_acc_t};
}

}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol
