#pragma once

#include <Eigen/Core>

namespace tmol {
namespace score {
namespace hbond {
namespace potentials {

template <typename Real>
struct HBondPoly {
  Eigen::Matrix<Real, 11, 1> coeffs;
  Real zero_pad;
  Eigen::Matrix<Real, 2, 1> range;
  Eigen::Matrix<Real, 2, 1> bound;
};

template <typename Real>
struct HBondPolynomials {
  HBondPoly<Real> AHdist_poly;
  HBondPoly<Real> cosBAH_poly;
  HBondPoly<Real> cosAHD_poly;
};

template <typename Real>
struct HBondGlobalParams {
  Real hb_sp2_range_span;
  Real hb_sp2_BAH180_rise;
  Real hb_sp2_outer_width;
  Real hb_sp3_softmax_fade;
  Real threshold_distance;
  Real max_ha_dis;
};

template <typename Real>
struct HBondPairParams {
  Real acceptor_hybridization;
  Real acceptor_weight;
  Real donor_weight;
};

}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol
