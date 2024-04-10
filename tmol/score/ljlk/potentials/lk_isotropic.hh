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

  static def V(
      Real dist,
      Real lj_radius_i,
      Real lk_dgfree_i,
      Real lk_lambda_i,
      Real lk_volume_j)
      ->Real {
    using std::exp;
    using std::pow;
    const Real pi = EIGEN_PI;
    static const Real pi_pow1p5 = 5.56832799683f;

    // clang-format off
    return (
      -lk_volume_j
      * lk_dgfree_i
      / (2 * pi_pow1p5 * lk_lambda_i)
      / (dist * dist)
      * exp(-(dist - lj_radius_i)*(dist - lj_radius_i) / (lk_lambda_i*lk_lambda_i))
    );
    // clang-format on
  }

  static def V_dV(
      Real dist,
      Real lj_radius_i,
      Real lk_dgfree_i,
      Real lk_lambda_i,
      Real lk_volume_j)
      ->V_dV_t {
    using std::exp;
    using std::pow;
    const Real pi = EIGEN_PI;
    static const Real pi_pow1p5 = 5.56832799683f;

    Real const exp_val =
        exp(-(dist - lj_radius_i) * (dist - lj_radius_i)
            / (lk_lambda_i * lk_lambda_i));
    // clang-format off
    Real desolv = (
      -lk_volume_j
      * lk_dgfree_i
      / (2 * pi_pow1p5 * lk_lambda_i)
      / (dist * dist)
      * exp_val
    );

    Real d_desolv_d_dist = (
      -lk_volume_j
      * lk_dgfree_i
      / (2 * pi_pow1p5 * lk_lambda_i)
      * exp_val
      * ((  // (f * exp(g))' = f' * exp(g) + f g' exp(g)
          -2 / (dist * dist * dist)
        ) + (
          1 / (dist * dist)
          * -(2 * dist - 2 * lj_radius_i)
          / (lk_lambda_i * lk_lambda_i)
        )
      )
    );
    // clang-format on

    return {desolv, d_desolv_d_dist};
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
      Real lk_dgfree_i,
      Real lk_lambda_i,
      Real lk_volume_j)
      ->Real {
    Real d_min = lj_sigma_ij * .89;

    Real cpoly_close_dmin = d_min * d_min - 1.45;
    if (cpoly_close_dmin < 0.01) cpoly_close_dmin = 0.01;
    cpoly_close_dmin = std::sqrt(cpoly_close_dmin);

    Real cpoly_close_dmax = std::sqrt(d_min * d_min + 1.05);

    Real cpoly_far_dmin = 4.5;
    Real cpoly_far_dmax = 6.0;

    Real weight = connectivity_weight<Real>(bonded_path_length);

    Real lk;

    if (dist > cpoly_far_dmax) {
      lk = 0.0;
    } else if (dist > cpoly_far_dmin) {
      auto f_desolv_at_dmin = f_desolv<Real>::V_dV(
          cpoly_far_dmin, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j);

      lk = interpolate_to_zero(
          dist,
          cpoly_far_dmin,
          f_desolv_at_dmin.V,
          f_desolv_at_dmin.dV_ddist,
          cpoly_far_dmax);
    } else if (dist > cpoly_close_dmax) {
      lk = f_desolv<Real>::V(
          dist, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j);
    } else if (dist > cpoly_close_dmin) {
      auto f_desolv_at_dmax = f_desolv<Real>::V_dV(
          cpoly_close_dmax, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j);

      lk = interpolate<Real>(
          dist,
          cpoly_close_dmin,
          f_desolv<Real>::V(
              d_min, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j),
          0.0,
          cpoly_close_dmax,
          f_desolv_at_dmax.V,
          f_desolv_at_dmax.dV_ddist);
    } else {
      lk = f_desolv<Real>::V(
          d_min, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j);
    }

    return weight * lk;
  }

  static def V_dV(
      Real dist,
      Real bonded_path_length,
      Real lj_sigma_ij,
      Real lj_radius_i,
      Real lk_dgfree_i,
      Real lk_lambda_i,
      Real lk_volume_j)
      ->V_dV_t {
    Real d_min = lj_sigma_ij * .89;

    Real cpoly_close_dmin = d_min * d_min - 1.45;
    if (cpoly_close_dmin < 0.01) cpoly_close_dmin = 0.01;
    cpoly_close_dmin = std::sqrt(cpoly_close_dmin);

    Real cpoly_close_dmax = std::sqrt(d_min * d_min + 1.05);

    Real cpoly_far_dmin = 4.5;
    Real cpoly_far_dmax = 6.0;

    Real weight = connectivity_weight<Real>(bonded_path_length);

    Real lk, d_lk_d_dist;

    if (dist < cpoly_close_dmin) {
      lk = f_desolv<Real>::V(
          d_min, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j);
      d_lk_d_dist = 0;

    } else if (dist < cpoly_close_dmax) {
      auto f_desolv_at_dmax = f_desolv<Real>::V_dV(
          cpoly_close_dmax, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j);

      tie(lk, d_lk_d_dist) = interpolate_V_dV<Real>(
          dist,
          cpoly_close_dmin,
          f_desolv<Real>::V(
              d_min, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j),
          0.0,
          cpoly_close_dmax,
          f_desolv_at_dmax.V,
          f_desolv_at_dmax.dV_ddist);

    } else if (dist < cpoly_far_dmin) {
      auto f_desolv_at_dist = f_desolv<Real>::V_dV(
          dist, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j);

      lk = f_desolv_at_dist.V;
      d_lk_d_dist = f_desolv_at_dist.dV_ddist;
    } else if (dist < cpoly_far_dmax) {
      auto f_desolv_at_dmin = f_desolv<Real>::V_dV(
          cpoly_far_dmin, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j);

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
      LJGlobalParams<Real> global)
      ->Real {
    Real lj_sigma_ij = lj_sigma<Real>(i, j, global);

    Real d_min = lj_sigma_ij * .89;

    Real cpoly_close_dmin = d_min * d_min - 1.45;
    cpoly_close_dmin = cpoly_close_dmin < 0.01 ? 0.01 : cpoly_close_dmin;
    cpoly_close_dmin = std::sqrt(cpoly_close_dmin);
    Real cpoly_close_dmax = std::sqrt(d_min * d_min + 1.05);

    Real cpoly_far_dmin = 4.5;
    Real cpoly_far_dmax = 6.0;

    Real weight = connectivity_weight<Real>(bonded_path_length);

    Real lk;
    if (dist > cpoly_far_dmax) {
      lk = 0.0;
    } else if (dist > cpoly_far_dmin) {
      auto f_desolv_at_dmin = f_desolv<Real>::V_dV(
          cpoly_far_dmin, i.lj_radius, i.lk_dgfree, i.lk_lambda, j.lk_volume);
      lk = interpolate_to_zero(
          dist,
          cpoly_far_dmin,
          f_desolv_at_dmin.V,
          f_desolv_at_dmin.dV_ddist,
          cpoly_far_dmax);
      f_desolv_at_dmin = f_desolv<Real>::V_dV(
          cpoly_far_dmin, j.lj_radius, j.lk_dgfree, j.lk_lambda, i.lk_volume);
      lk += interpolate_to_zero(
          dist,
          cpoly_far_dmin,
          f_desolv_at_dmin.V,
          f_desolv_at_dmin.dV_ddist,
          cpoly_far_dmax);
    } else if (dist > cpoly_close_dmax) {
      lk = f_desolv<Real>::V(
               dist, i.lj_radius, i.lk_dgfree, i.lk_lambda, j.lk_volume)
           + f_desolv<Real>::V(
               dist, j.lj_radius, j.lk_dgfree, j.lk_lambda, i.lk_volume);
    } else if (dist > cpoly_close_dmin) {
      auto f_desolv_at_dmax = f_desolv<Real>::V_dV(
          cpoly_close_dmax, i.lj_radius, i.lk_dgfree, i.lk_lambda, j.lk_volume);
      lk = interpolate<Real>(
          dist,
          cpoly_close_dmin,
          f_desolv<Real>::V(
              d_min, i.lj_radius, i.lk_dgfree, i.lk_lambda, j.lk_volume),
          0.0,
          cpoly_close_dmax,
          f_desolv_at_dmax.V,
          f_desolv_at_dmax.dV_ddist);
      f_desolv_at_dmax = f_desolv<Real>::V_dV(
          cpoly_close_dmax, j.lj_radius, j.lk_dgfree, j.lk_lambda, i.lk_volume);
      lk += interpolate<Real>(
          dist,
          cpoly_close_dmin,
          f_desolv<Real>::V(
              d_min, j.lj_radius, j.lk_dgfree, j.lk_lambda, i.lk_volume),
          0.0,
          cpoly_close_dmax,
          f_desolv_at_dmax.V,
          f_desolv_at_dmax.dV_ddist);

    } else {
      lk = f_desolv<Real>::V(
               d_min, i.lj_radius, i.lk_dgfree, i.lk_lambda, j.lk_volume)
           + f_desolv<Real>::V(
               d_min, j.lj_radius, j.lk_dgfree, j.lk_lambda, i.lk_volume);
    }
    return weight * lk;
  }

  static def V_dV(
      Real dist,
      Real bonded_path_length,
      LKTypeParams<Real> i,
      LKTypeParams<Real> j,
      LJGlobalParams<Real> global)
      ->V_dV_t {
    Real sigma = lj_sigma<Real>(i, j, global);

    auto ij = lk_isotropic_pair<Real>::V_dV(
        dist,
        bonded_path_length,
        sigma,
        i.lj_radius,
        i.lk_dgfree,
        i.lk_lambda,
        j.lk_volume);

    auto ji = lk_isotropic_pair<Real>::V_dV(
        dist,
        bonded_path_length,
        sigma,
        j.lj_radius,
        j.lk_dgfree,
        j.lk_lambda,
        i.lk_volume);

    return {ij.V + ji.V, ij.dV_ddist + ji.dV_ddist};
  }
};

#undef def

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
