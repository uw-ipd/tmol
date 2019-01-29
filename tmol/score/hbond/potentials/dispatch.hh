#pragma once

#include <cmath>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/score/common/tuple.hh>

#include <cppitertools/product.hpp>
#include <cppitertools/range.hpp>
#include <tmol/score/hbond/potentials/potentials.hh>

namespace tmol {
namespace score {
namespace hbond {
namespace potentials {

using tmol::Device;
using tmol::TView;

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <typename Real, typename Int>
auto hbond_pair_score(
    TView<Vec<Real, 3>, 1, Device::CPU> D,
    TView<Vec<Real, 3>, 1, Device::CPU> H,
    TView<Int, 1, Device::CPU> donor_type,

    TView<Vec<Real, 3>, 1, Device::CPU> A,
    TView<Vec<Real, 3>, 1, Device::CPU> B,
    TView<Vec<Real, 3>, 1, Device::CPU> B0,
    TView<Int, 1, Device::CPU> acceptor_type,

    TView<Int, 2, Device::CPU> acceptor_class,
    TView<Real, 2, Device::CPU> acceptor_weight,
    TView<Real, 2, Device::CPU> donor_weight,

    TView<Vec<double, 11>, 2, Device::CPU> AHdist_coeffs,
    TView<Vec<double, 2>, 2, Device::CPU> AHdist_range,
    TView<Vec<double, 2>, 2, Device::CPU> AHdist_bound,

    TView<Vec<double, 11>, 2, Device::CPU> cosBAH_coeffs,
    TView<Vec<double, 2>, 2, Device::CPU> cosBAH_range,
    TView<Vec<double, 2>, 2, Device::CPU> cosBAH_bound,

    TView<Vec<double, 11>, 2, Device::CPU> cosAHD_coeffs,
    TView<Vec<double, 2>, 2, Device::CPU> cosAHD_range,
    TView<Vec<double, 2>, 2, Device::CPU> cosAHD_bound,

    Real hb_sp2_range_span,
    Real hb_sp2_BAH180_rise,
    Real hb_sp2_outer_width,
    Real hb_sp3_softmax_fade,
    Real threshold_distance)
    -> tuple<
        at::Tensor,
        at::Tensor,
        at::Tensor,
        at::Tensor,
        at::Tensor,
        at::Tensor,
        at::Tensor> {
  using iter::product;
  using iter::range;

  using tmol::TPack;

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

  typedef TPack<int64_t, 2, Device::CPU> IndT;
  IndT ind_t = IndT::empty({D.size(0) * A.size(0), 2});
  auto ind = ind_t.view;

  int nresult = 0;
  Real squared_threshold = threshold_distance * threshold_distance;

  int di, ai;
  for (auto t : product(range(D.size(0)), range(A.size(0)))) {
    tie(di, ai) = t;
    if ((H[di] - A[ai]).squaredNorm() < squared_threshold) {
      ind[nresult][0] = di;
      ind[nresult][1] = ai;
      nresult++;
    }
  }

  ind_t = IndT(ind_t.tensor.slice(0, 0, nresult).clone());
  ind = ind_t.view;

  auto E_t = TPack<Real, 1, Device::CPU>::empty({nresult});
  auto dE_dD_t = TPack<Vec<Real, 3>, 1, Device::CPU>::empty({nresult});
  auto dE_dH_t = TPack<Vec<Real, 3>, 1, Device::CPU>::empty({nresult});
  auto dE_dA_t = TPack<Vec<Real, 3>, 1, Device::CPU>::empty({nresult});
  auto dE_dB_t = TPack<Vec<Real, 3>, 1, Device::CPU>::empty({nresult});
  auto dE_dB0_t = TPack<Vec<Real, 3>, 1, Device::CPU>::empty({nresult});

  auto E = E_t.view;
  auto dE_dD = dE_dD_t.view;
  auto dE_dH = dE_dH_t.view;
  auto dE_dA = dE_dA_t.view;
  auto dE_dB = dE_dB_t.view;
  auto dE_dB0 = dE_dB0_t.view;

  for (auto r : range(nresult)) {
    int di = ind[r][0];
    int ai = ind[r][1];

    int dt = donor_type[di];
    int at = acceptor_type[ai];

    tie(E[r], dE_dD[r], dE_dH[r], dE_dA[r], dE_dB[r], dE_dB0[r]) =
        hbond_score_V_dV(
            D[di],
            H[di],

            A[ai],
            B[ai],
            B0[ai],

            acceptor_class[dt][at],
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

            hb_sp2_range_span,
            hb_sp2_BAH180_rise,
            hb_sp2_outer_width,
            hb_sp3_softmax_fade);
  }

  return {ind_t.tensor,
          E_t.tensor,
          dE_dD_t.tensor,
          dE_dH_t.tensor,
          dE_dA_t.tensor,
          dE_dB_t.tensor,
          dE_dB0_t.tensor};
}

}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol
