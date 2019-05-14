#pragma once

#include <Eigen/Core>
#include <tuple>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>

#include "params.hh"

#undef B0

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

      TView<HBondPoly<double>, 2, Dev> AHdist_polys,
      // TView<Vec<double, 11>, 2, Dev> AHdist_coeffs,
      // TView<Vec<double, 2>, 2, Dev> AHdist_range,
      // TView<Vec<double, 2>, 2, Dev> AHdist_bound,

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
          TPack<Vec<Real, 3>, 1, Dev>>;

  //  static auto backward(
  //    TView<Real, 1, Dev> dTdV,
  //    TView<Real, 2, Dev> dV_d_don,
  //    TView<Real, 2, Dev> dV_d_acc)
  //    -> std::tuple<
  //        TPack<Real, 2, Dev>,
  //        TPack<Real, 2, Dev>>;
};

}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol
