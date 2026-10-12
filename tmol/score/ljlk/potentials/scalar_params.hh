#pragma once

#include <cmath>

#include <Eigen/Core>
namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

template <typename Real>
struct LJTypeParams {
  Real lj_radius;
  Real lj_sqrt_wdepth;
  Real is_donor;
  Real is_hydroxyl;
  Real is_polarh;
  Real is_acceptor;
};

template <typename Real>
struct LKTypeParams {
  Real lj_radius;
  Real lk_dgfree;
  Real lk_lambda;
  Real lk_volume;
  Real is_donor;
  Real is_hydroxyl;
  Real is_polarh;
  Real is_acceptor;
  Real is_carbon_lk;
  Real lk_coeff;
  Real lk_inv_lambda2;
};

template <typename Real>
struct LJLKTypeParams {
  Real lj_radius;
  Real lj_sqrt_wdepth;
  Real lk_dgfree;
  Real lk_lambda;
  Real lk_volume;
  Real is_donor;
  Real is_hydroxyl;
  Real is_polarh;
  Real is_acceptor;
  Real is_carbon_lk;
  Real is_hydrogen;
  Real lk_coeff;
  Real lk_inv_lambda2;

  LJTypeParams<Real> EIGEN_DEVICE_FUNC lj_params() const {
    return LJTypeParams<Real>(
        {lj_radius,
         lj_sqrt_wdepth,
         is_donor,
         is_hydroxyl,
         is_polarh,
         is_acceptor});
  }

  LKTypeParams<Real> EIGEN_DEVICE_FUNC lk_params() const {
    return LKTypeParams<Real>(
        {lj_radius,
         lk_dgfree,
         lk_lambda,
         lk_volume,
         is_donor,
         is_hydroxyl,
         is_polarh,
         is_acceptor,
         is_carbon_lk,
         lk_coeff,
         lk_inv_lambda2});
  }
};

template <typename Real>
struct LJGlobalParams {
  Real max_dis;
  Real lj_dlin_sigma_factor;
  Real lj_hbond_dis;
  Real lj_hbond_OH_donor_dis;
  Real lj_hbond_hdis;
};
}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
