#pragma once

#include <cmath>

#include <Eigen/Core>

#include <tmol/score/common/cubic_hermite_polynomial.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/tuple_operators.hh>

#include "common.hh"

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

#define def auto EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE

using namespace tmol::score::common;

template <typename Real>
struct f_desolv {
  struct V_dV_t {
    Real V;
    Real dV_ddist;

    def astuple() { return tmol::score::common::make_tuple(V, dV_ddist); }
  };

  static def V_precomputed(
      Real dist,
      Real lj_radius_i,
      Real lk_coeff_i,
      Real lk_inv_lambda2_i,
      Real lk_volume_j) -> Real {
    Real const delta = dist - lj_radius_i;
    Real const inv_dist = 1 / dist;
    return lk_volume_j * lk_coeff_i * inv_dist * inv_dist
           * std::exp(-delta * delta * lk_inv_lambda2_i);
  }

  static def V_dV_precomputed(
      Real dist,
      Real lj_radius_i,
      Real lk_coeff_i,
      Real lk_inv_lambda2_i,
      Real lk_volume_j) -> V_dV_t {
    Real const delta = dist - lj_radius_i;
    Real const inv_dist = 1 / dist;
    Real const desolv = lk_volume_j * lk_coeff_i * inv_dist * inv_dist
                        * std::exp(-delta * delta * lk_inv_lambda2_i);
    Real const d_desolv_d_dist =
        desolv * (-2 * inv_dist - 2 * delta * lk_inv_lambda2_i);
    return {desolv, d_desolv_d_dist};
  }

  static def V(
      Real dist,
      Real lj_radius_i,
      Real lk_dgfree_i,
      Real lk_lambda_i,
      Real lk_volume_j) -> Real {
    static const Real pi_pow1p5 = 5.56832799683f;
    return V_precomputed(
        dist,
        lj_radius_i,
        -lk_dgfree_i / (2 * pi_pow1p5 * lk_lambda_i),
        1 / (lk_lambda_i * lk_lambda_i),
        lk_volume_j);
  }

  static def V_dV(
      Real dist,
      Real lj_radius_i,
      Real lk_dgfree_i,
      Real lk_lambda_i,
      Real lk_volume_j) -> V_dV_t {
    static const Real pi_pow1p5 = 5.56832799683f;
    return V_dV_precomputed(
        dist,
        lj_radius_i,
        -lk_dgfree_i / (2 * pi_pow1p5 * lk_lambda_i),
        1 / (lk_lambda_i * lk_lambda_i),
        lk_volume_j);
  }
};

template <typename Real>
struct lk_isotropic_pair {
  struct V_dV_t {
    Real V;
    Real dV_ddist;

    def astuple() { return tmol::score::common::make_tuple(V, dV_ddist); }
  };

  static def V(
      Real dist,
      Real bonded_path_length,
      Real lj_sigma_ij,
      Real lj_radius_i,
      Real lk_volume_j,
      Real lk_coeff_i,
      Real lk_inv_lambda2_i,
      Real max_dis,
      bool is_cc_pair) -> Real {
    Real d_min = lj_sigma_ij * .89;
    if (is_cc_pair)
      d_min = std::max(d_min, Real(4.2));  // C-C modifypot flatten

    // close-spline knots lie on etable bins spaced 1/20 A^2 apart
    Real const n = std::floor(Real(20) * d_min * d_min);
    Real cpoly_close_dmin = std::sqrt(std::max(Real(0), n - 29) / 20);
    Real cpoly_close_dmax = std::sqrt(std::min(n + 21, Real(405)) / 20);

    Real cpoly_far_dmax = max_dis;
    Real cpoly_far_dmin = max_dis - Real(1.5);

    Real weight = connectivity_weight<Real>(bonded_path_length);

    Real lk;

    if (dist > cpoly_far_dmax) {
      lk = 0.0;
    } else if (dist > cpoly_far_dmin) {
      auto f_desolv_at_dmin = f_desolv<Real>::V_dV_precomputed(
          cpoly_far_dmin,
          lj_radius_i,
          lk_coeff_i,
          lk_inv_lambda2_i,
          lk_volume_j);

      lk = interpolate_to_zero(
          dist,
          cpoly_far_dmin,
          f_desolv_at_dmin.V,
          f_desolv_at_dmin.dV_ddist,
          cpoly_far_dmax);
    } else if (dist > cpoly_close_dmax) {
      lk = f_desolv<Real>::V_precomputed(
          dist, lj_radius_i, lk_coeff_i, lk_inv_lambda2_i, lk_volume_j);
    } else if (dist > cpoly_close_dmin) {
      auto f_desolv_at_dmax = f_desolv<Real>::V_dV_precomputed(
          cpoly_close_dmax,
          lj_radius_i,
          lk_coeff_i,
          lk_inv_lambda2_i,
          lk_volume_j);

      lk = interpolate<Real>(
          dist,
          cpoly_close_dmin,
          f_desolv<Real>::V_precomputed(
              d_min, lj_radius_i, lk_coeff_i, lk_inv_lambda2_i, lk_volume_j),
          0.0,
          cpoly_close_dmax,
          f_desolv_at_dmax.V,
          f_desolv_at_dmax.dV_ddist);
    } else {
      lk = f_desolv<Real>::V_precomputed(
          d_min, lj_radius_i, lk_coeff_i, lk_inv_lambda2_i, lk_volume_j);
    }

    return weight * lk;
  }

  static def V_dV(
      Real dist,
      Real bonded_path_length,
      Real lj_sigma_ij,
      Real lj_radius_i,
      Real lk_volume_j,
      Real lk_coeff_i,
      Real lk_inv_lambda2_i,
      Real max_dis,
      bool is_cc_pair) -> V_dV_t {
    Real d_min = lj_sigma_ij * .89;
    if (is_cc_pair)
      d_min = std::max(d_min, Real(4.2));  // C-C modifypot flatten

    // close-spline knots lie on etable bins spaced 1/20 A^2 apart
    Real const n = std::floor(Real(20) * d_min * d_min);
    Real cpoly_close_dmin = std::sqrt(std::max(Real(0), n - 29) / 20);
    Real cpoly_close_dmax = std::sqrt(std::min(n + 21, Real(405)) / 20);

    Real cpoly_far_dmax = max_dis;
    Real cpoly_far_dmin = max_dis - Real(1.5);

    Real weight = connectivity_weight<Real>(bonded_path_length);

    Real lk, d_lk_d_dist;

    if (dist < cpoly_close_dmin) {
      lk = f_desolv<Real>::V_precomputed(
          d_min, lj_radius_i, lk_coeff_i, lk_inv_lambda2_i, lk_volume_j);
      d_lk_d_dist = 0;

    } else if (dist < cpoly_close_dmax) {
      auto f_desolv_at_dmax = f_desolv<Real>::V_dV_precomputed(
          cpoly_close_dmax,
          lj_radius_i,
          lk_coeff_i,
          lk_inv_lambda2_i,
          lk_volume_j);

      tie(lk, d_lk_d_dist) = interpolate_V_dV<Real>(
          dist,
          cpoly_close_dmin,
          f_desolv<Real>::V_precomputed(
              d_min, lj_radius_i, lk_coeff_i, lk_inv_lambda2_i, lk_volume_j),
          0.0,
          cpoly_close_dmax,
          f_desolv_at_dmax.V,
          f_desolv_at_dmax.dV_ddist);

    } else if (dist < cpoly_far_dmin) {
      auto f_desolv_at_dist = f_desolv<Real>::V_dV_precomputed(
          dist, lj_radius_i, lk_coeff_i, lk_inv_lambda2_i, lk_volume_j);

      lk = f_desolv_at_dist.V;
      d_lk_d_dist = f_desolv_at_dist.dV_ddist;
    } else if (dist < cpoly_far_dmax) {
      auto f_desolv_at_dmin = f_desolv<Real>::V_dV_precomputed(
          cpoly_far_dmin,
          lj_radius_i,
          lk_coeff_i,
          lk_inv_lambda2_i,
          lk_volume_j);

      tie(lk, d_lk_d_dist) = interpolate_to_zero_V_dV(
          dist,
          cpoly_far_dmin,
          f_desolv_at_dmin.V,
          f_desolv_at_dmin.dV_ddist,
          cpoly_far_dmax);

    } else {
      lk = 0.0;
      d_lk_d_dist = 0.0;
    }

    return {weight * lk, weight * d_lk_d_dist};
  }
};

template <typename Real>
struct lk_isotropic_score {
  struct V_dV_t {
    Real V;
    Real dV_ddist;

    def astuple() { return tmol::score::common::make_tuple(V, dV_ddist); }
  };

  static def V(
      Real dist,
      Real bonded_path_length,
      LKTypeParams<Real> i,
      LKTypeParams<Real> j,
      LJGlobalParams<Real> global) -> Real {
    if (dist > global.max_dis) {
      return 0.0;
    }

    Real lj_sigma_ij = lj_sigma<Real>(i, j, global);

    bool is_cc_pair = i.is_carbon_lk && j.is_carbon_lk;
    Real d_min = lj_sigma_ij * .89;
    if (is_cc_pair)
      d_min = std::max(d_min, Real(4.2));  // C-C modifypot flatten

    // close-spline knots lie on etable bins spaced 1/20 A^2 apart
    Real const n = std::floor(Real(20) * d_min * d_min);
    Real cpoly_close_dmin = std::sqrt(std::max(Real(0), n - 29) / 20);
    Real cpoly_close_dmax = std::sqrt(std::min(n + 21, Real(405)) / 20);

    Real cpoly_far_dmax = global.max_dis;
    Real cpoly_far_dmin = global.max_dis - Real(1.5);

    Real weight = connectivity_weight<Real>(bonded_path_length);

    Real lk;
    if (dist > cpoly_far_dmin) {
      auto f_desolv_at_dmin = f_desolv<Real>::V_dV_precomputed(
          cpoly_far_dmin,
          i.lj_radius,
          i.lk_coeff,
          i.lk_inv_lambda2,
          j.lk_volume);
      lk = interpolate_to_zero(
          dist,
          cpoly_far_dmin,
          f_desolv_at_dmin.V,
          f_desolv_at_dmin.dV_ddist,
          cpoly_far_dmax);
      f_desolv_at_dmin = f_desolv<Real>::V_dV_precomputed(
          cpoly_far_dmin,
          j.lj_radius,
          j.lk_coeff,
          j.lk_inv_lambda2,
          i.lk_volume);
      lk += interpolate_to_zero(
          dist,
          cpoly_far_dmin,
          f_desolv_at_dmin.V,
          f_desolv_at_dmin.dV_ddist,
          cpoly_far_dmax);
    } else if (dist > cpoly_close_dmax) {
      lk = f_desolv<Real>::V_precomputed(
               dist, i.lj_radius, i.lk_coeff, i.lk_inv_lambda2, j.lk_volume)
           + f_desolv<Real>::V_precomputed(
               dist, j.lj_radius, j.lk_coeff, j.lk_inv_lambda2, i.lk_volume);
    } else if (dist > cpoly_close_dmin) {
      auto f_desolv_at_dmax = f_desolv<Real>::V_dV_precomputed(
          cpoly_close_dmax,
          i.lj_radius,
          i.lk_coeff,
          i.lk_inv_lambda2,
          j.lk_volume);
      lk = interpolate<Real>(
          dist,
          cpoly_close_dmin,
          f_desolv<Real>::V_precomputed(
              d_min, i.lj_radius, i.lk_coeff, i.lk_inv_lambda2, j.lk_volume),
          0.0,
          cpoly_close_dmax,
          f_desolv_at_dmax.V,
          f_desolv_at_dmax.dV_ddist);
      f_desolv_at_dmax = f_desolv<Real>::V_dV_precomputed(
          cpoly_close_dmax,
          j.lj_radius,
          j.lk_coeff,
          j.lk_inv_lambda2,
          i.lk_volume);
      lk += interpolate<Real>(
          dist,
          cpoly_close_dmin,
          f_desolv<Real>::V_precomputed(
              d_min, j.lj_radius, j.lk_coeff, j.lk_inv_lambda2, i.lk_volume),
          0.0,
          cpoly_close_dmax,
          f_desolv_at_dmax.V,
          f_desolv_at_dmax.dV_ddist);

    } else {
      lk = f_desolv<Real>::V_precomputed(
               d_min, i.lj_radius, i.lk_coeff, i.lk_inv_lambda2, j.lk_volume)
           + f_desolv<Real>::V_precomputed(
               d_min, j.lj_radius, j.lk_coeff, j.lk_inv_lambda2, i.lk_volume);
    }
    return weight * lk;
  }

  static def V_dV(
      Real dist,
      Real bonded_path_length,
      LKTypeParams<Real> i,
      LKTypeParams<Real> j,
      LJGlobalParams<Real> global) -> V_dV_t {
    Real sigma = lj_sigma<Real>(i, j, global);
    bool is_cc_pair = i.is_carbon_lk && j.is_carbon_lk;

    auto ij = lk_isotropic_pair<Real>::V_dV(
        dist,
        bonded_path_length,
        sigma,
        i.lj_radius,
        j.lk_volume,
        i.lk_coeff,
        i.lk_inv_lambda2,
        global.max_dis,
        is_cc_pair);

    auto ji = lk_isotropic_pair<Real>::V_dV(
        dist,
        bonded_path_length,
        sigma,
        j.lj_radius,
        i.lk_volume,
        j.lk_coeff,
        j.lk_inv_lambda2,
        global.max_dis,
        is_cc_pair);

    return {ij.V + ji.V, ij.dV_ddist + ji.dV_ddist};
  }
};

#undef def

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
