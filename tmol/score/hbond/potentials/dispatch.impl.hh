#pragma once

#include <cmath>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <tuple>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/score/common/tuple.hh>

#include <tmol/score/hbond/potentials/potentials.hh>

#include <tmol/utility/nvtx.hh>

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
struct HBondDispatch {
  static auto f(
      TView<Vec<Real, 3>, 1, Dev> D,
      TView<Vec<Real, 3>, 1, Dev> H,
      TView<Int, 1, Dev> donor_type,

      TView<Vec<Real, 3>, 1, Dev> A,
      TView<Vec<Real, 3>, 1, Dev> B,
      TView<Vec<Real, 3>, 1, Dev> B0,
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

      Real hb_sp2_range_span,
      Real hb_sp2_BAH180_rise,
      Real hb_sp2_outer_width,
      Real hb_sp3_softmax_fade,
      Real threshold_distance)
      -> std::tuple<
          TPack<int64_t, 2, Dev>,
          TPack<Real, 1, Dev>,
          TPack<Vec<Real, 3>, 1, Dev>,
          TPack<Vec<Real, 3>, 1, Dev>,
          TPack<Vec<Real, 3>, 1, Dev>,
          TPack<Vec<Real, 3>, 1, Dev>,
          TPack<Vec<Real, 3>, 1, Dev>> {
    nvtxRangePush("hbond dispatch f");
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

    Dispatch<Dev> dispatcher(H.size(0), A.size(0));
    nvtxRangePush("dispatcher.scan");
    auto nresult = dispatcher.scan(threshold_distance, H, A);
    nvtxRangePop();

    nvtxRangePush("allocate empty tensors");
    auto ind_t = TPack<int64_t, 2, Dev>::empty({nresult, 2});
    auto E_t = TPack<Real, 1, Dev>::empty({nresult});
    auto dE_dD_t = TPack<Vec<Real, 3>, 1, Dev>::empty({nresult});
    auto dE_dH_t = TPack<Vec<Real, 3>, 1, Dev>::empty({nresult});
    auto dE_dA_t = TPack<Vec<Real, 3>, 1, Dev>::empty({nresult});
    auto dE_dB_t = TPack<Vec<Real, 3>, 1, Dev>::empty({nresult});
    auto dE_dB0_t = TPack<Vec<Real, 3>, 1, Dev>::empty({nresult});
    nvtxRangePop();

    auto ind = ind_t.view;
    auto E = E_t.view;
    auto dE_dD = dE_dD_t.view;
    auto dE_dH = dE_dH_t.view;
    auto dE_dA = dE_dA_t.view;
    auto dE_dB = dE_dB_t.view;
    auto dE_dB0 = dE_dB0_t.view;

    nvtxRangePush("dispatcher.score");
    dispatcher.score([=] EIGEN_DEVICE_FUNC(int o, int di, int ai) {
      ind[o][0] = di;
      ind[o][1] = ai;

      int dt = donor_type[di];
      int at = acceptor_type[ai];

      tie(E[o], dE_dD[o], dE_dH[o], dE_dA[o], dE_dB[o], dE_dB0[o]) =
          hbond_score_V_dV(
              D[di],
              H[di],

              A[ai],
              B[ai],
              B0[ai],

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

              hb_sp2_range_span,
              hb_sp2_BAH180_rise,
              hb_sp2_outer_width,
              hb_sp3_softmax_fade);
    });
    nvtxRangePop();

    nvtxRangePop();
    return {ind_t, E_t, dE_dD_t, dE_dH_t, dE_dA_t, dE_dB_t, dE_dB0_t};
  }
};

}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol
