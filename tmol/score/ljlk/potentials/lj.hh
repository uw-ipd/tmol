#pragma once

#include <Eigen/Core>
#include <cmath>

#include <tmol/score/common/cubic_hermite_polynomial.hh>
#include <tmol/score/common/tuple.hh>

#include "common.hh"

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

#define def auto EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE

using namespace tmol::score::common;

template <typename Real>
struct vdw {
  struct dV_t {
    Real V;
    Real dV_ddist;

    def astuple() { return tmol::score::common::make_tuple(V, dV_ddist); }
  };

  static def V(Real dist, Real sigma, Real epsilon)->Real {
    Real sd = (sigma / dist);
    Real sd2 = sd * sd;
    Real sd6 = sd2 * sd2 * sd2;
    Real sd12 = sd6 * sd6;
    return epsilon * (sd12 - 2 * sd6);
  }

  static def V_dV(Real dist, Real sigma, Real epsilon)->dV_t {
    Real sd = (sigma / dist);
    Real sd2 = sd * sd;
    Real sd6 = sd2 * sd2 * sd2;
    Real sd12 = sd6 * sd6;

    return {epsilon * (sd12 - 2.0 * sd6),
            epsilon * ((-12.0 * sd12 / dist) - (2.0 * -6.0 * sd6 / dist))};
  }
};

template <typename Real>
struct lj_score {
  struct V_dV_t {
    Real V;
    Real dV_ddist;

    def astuple() { return tmol::score::common::make_tuple(V, dV_ddist); }
  };

  static def V(
      Real dist,
      Real bonded_path_length,
      LJTypeParams<Real> i,
      LJTypeParams<Real> j,
      LJGlobalParams<Real> global)
      ->Real {
    Real sigma = lj_sigma<Real>(i, j, global);
    Real weight = connectivity_weight<Real, Real>(bonded_path_length);
    Real epsilon = std::sqrt(i.lj_wdepth * j.lj_wdepth);

    Real d_lin = sigma * 0.6;
    Real cpoly_dmin = 4.5;
    Real cpoly_dmax = 6.0;

    if (dist < d_lin) {
      auto vdw_at_d_lin = vdw<Real>::V_dV(d_lin, sigma, epsilon);
      return weight * (vdw_at_d_lin.V + vdw_at_d_lin.dV_ddist * (dist - d_lin));
    } else if (dist < cpoly_dmin) {
      return weight * vdw<Real>::V(dist, sigma, epsilon);
    } else if (dist < cpoly_dmax) {
      auto vdw_at_cpoly_dmin = vdw<Real>::V_dV(cpoly_dmin, sigma, epsilon);
      return weight
             * interpolate_to_zero(
                   dist,
                   cpoly_dmin,
                   vdw_at_cpoly_dmin.V,
                   vdw_at_cpoly_dmin.dV_ddist,
                   cpoly_dmax);

    } else {
      return 0.0;
    }
  }

  static def V_dV(
      Real dist,
      Real bonded_path_length,
      LJTypeParams<Real> i,
      LJTypeParams<Real> j,
      LJGlobalParams<Real> global)
      ->V_dV_t {
    Real sigma = lj_sigma<Real>(i, j, global);
    Real weight = connectivity_weight<Real, Real>(bonded_path_length);
    Real epsilon = std::sqrt(i.lj_wdepth * j.lj_wdepth);

    Real d_lin = sigma * 0.6;
    Real cpoly_dmin = 4.5;
    Real cpoly_dmax = 6.0;

    Real lj, d_lj_d_dist;

    if (dist < d_lin) {
      auto vdw_at_d_lin = vdw<Real>::V_dV(d_lin, sigma, epsilon);

      lj = vdw_at_d_lin.V + vdw_at_d_lin.dV_ddist * (dist - d_lin);
      d_lj_d_dist = vdw_at_d_lin.dV_ddist;
    } else if (dist < cpoly_dmin) {
      auto vdw_at_dist = vdw<Real>::V_dV(dist, sigma, epsilon);

      lj = vdw_at_dist.V;
      d_lj_d_dist = vdw_at_dist.dV_ddist;

    } else if (dist < cpoly_dmax) {
      auto vdw_at_cpoly_dmin = vdw<Real>::V_dV(cpoly_dmin, sigma, epsilon);
      tie(lj, d_lj_d_dist) = interpolate_to_zero_V_dV(
          dist,
          cpoly_dmin,
          vdw_at_cpoly_dmin.V,
          vdw_at_cpoly_dmin.dV_ddist,
          cpoly_dmax);

    } else {
      lj = 0.0;
      d_lj_d_dist = 0.0;
    }

    return {weight * lj, weight * d_lj_d_dist};
  }
};

#undef def

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
