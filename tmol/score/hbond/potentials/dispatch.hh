#pragma once

#include <cmath>
#include <tuple>
#include <utility>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>

#include <cppitertools/product.hpp>
#include <cppitertools/range.hpp>
#include <tmol/score/hbond/potentials/potentials.hh>

namespace tmol {
namespace score {
namespace hbond {
namespace potentials {

using std::tie;
using std::tuple;
using tmol::TView;

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <typename Real, int N>
struct HBondPolyParams {
  TView<Vec<Real, 2>, N> range;
  TView<Vec<Real, 2>, N> bound;
  TView<Vec<Real, 11>, N> coeffs;
};

template <typename Real, int N>
struct HBondPairParams {
  TView<int32_t, N> acceptor_class;
  TView<Real, N> acceptor_weight;
  TView<Real, N> donor_weight;
  HBondPolyParams<Real, N> AHdist;
  HBondPolyParams<Real, N> cosBAH;
  HBondPolyParams<Real, N> cosAHD;
};

template <typename Real>
struct HBondGlobalParams {
  Real hb_sp2_range_span;
  Real hb_sp2_BAH180_rise;
  Real hb_sp2_outer_width;
  Real hb_sp3_softmax_fade;
  Real threshold_distance;
};

template <typename Real, typename Int>
auto hbond_pair_score(
    TView<Vec<Real, 3>, 1> D,
    TView<Vec<Real, 3>, 1> H,
    TView<Int, 1> donor_type_index,

    TView<Vec<Real, 3>, 1> A,
    TView<Vec<Real, 3>, 1> B,
    TView<Vec<Real, 3>, 1> B0,
    TView<Int, 1> acceptor_type_index,

    HBondPairParams<Real, 2> type_pair_params,
    HBondGlobalParams<Real> global_params)
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

  using tmol::new_tensor;

  AT_ASSERTM(
      donor_type_index.size(0) == D.size(0),
      "Invalid donor coordinate shapes.");
  AT_ASSERTM(
      donor_type_index.size(0) == H.size(0),
      "Invalid donor coordinate shapes.");

  AT_ASSERTM(
      acceptor_type_index.size(0) == A.size(0),
      "Invalid acceptor coordinate shapes.");
  AT_ASSERTM(
      acceptor_type_index.size(0) == B.size(0),
      "Invalid acceptor coordinate shapes.");
  AT_ASSERTM(
      acceptor_type_index.size(0) == B0.size(0),
      "Invalid acceptor coordinate shapes.");

  auto [ind_t, ind] = new_tensor<int64_t, 2>({D.size(0) * A.size(0), 2});
  int nresult = 0;
  Real squared_threshold =
      global_params.threshold_distance * global_params.threshold_distance;

  for (auto [di, ai] : product(range(D.size(0)), range(A.size(0)))) {
    if ((H[di] - A[ai]).squaredNorm() < squared_threshold) {
      ind[nresult][0] = di;
      ind[nresult][1] = ai;
      nresult++;
    }
  }

  ind_t = ind_t.slice(0, 0, nresult).clone();
  ind = view_tensor<int64_t, 2>(ind_t);

  auto [E_t, E] = new_tensor<Real, 1>({nresult});
  auto [dE_dD_t, dE_dD] = new_tensor<Vec<Real, 3>, 1>({nresult});
  auto [dE_dH_t, dE_dH] = new_tensor<Vec<Real, 3>, 1>({nresult});
  auto [dE_dA_t, dE_dA] = new_tensor<Vec<Real, 3>, 1>({nresult});
  auto [dE_dB_t, dE_dB] = new_tensor<Vec<Real, 3>, 1>({nresult});
  auto [dE_dB0_t, dE_dB0] = new_tensor<Vec<Real, 3>, 1>({nresult});

  for (auto r : range(nresult)) {
    int di = ind[r][0];
    int ai = ind[r][1];

    int dt = donor_type_index[di];
    int at = acceptor_type_index[ai];

    tie(E[r], dE_dD[r], dE_dH[r], dE_dA[r], dE_dB[r], dE_dB0[r]) =
        hbond_score_V_dV(
            D[di],
            H[di],

            A[ai],
            B[ai],
            B0[ai],

            type_pair_params.acceptor_class[dt][at],
            type_pair_params.acceptor_weight[dt][at],
            type_pair_params.donor_weight[dt][at],

            type_pair_params.AHdist.coeffs[dt][at],
            type_pair_params.AHdist.range[dt][at],
            type_pair_params.AHdist.bound[dt][at],

            type_pair_params.cosBAH.coeffs[dt][at],
            type_pair_params.cosBAH.range[dt][at],
            type_pair_params.cosBAH.bound[dt][at],

            type_pair_params.cosAHD.coeffs[dt][at],
            type_pair_params.cosAHD.range[dt][at],
            type_pair_params.cosAHD.bound[dt][at],

            global_params.hb_sp2_range_span,
            global_params.hb_sp2_BAH180_rise,
            global_params.hb_sp2_outer_width,
            global_params.hb_sp3_softmax_fade);
  }

  return {ind_t, E_t, dE_dD_t, dE_dH_t, dE_dA_t, dE_dB_t, dE_dB0_t};
}

}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol
