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

// the base vdw (no shift, linearizing, or fading applied)
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

    return {
        epsilon * (sd12 - Real(2.0) * sd6),
        epsilon * ((Real(-12.0) * sd12 / dist) - (Real(-12.0) * sd6 / dist))};
  }
};

// LJ energies, with shift, linearizing, and fading
template <typename Real>
struct lj_score {
  struct V_dV_t {
    Real Vatr;
    Real Vrep;
    Real dVatr_ddist;
    Real dVrep_ddist;

    def astuple() {
      return tmol::score::common::make_tuple(
          Vatr, Vrep, dVatr_ddist, dVrep_ddist);
    }
  };

  static def V(
      Real dist,
      Real bonded_path_length,
      LJTypeParams<Real> i,
      LJTypeParams<Real> j,
      LJGlobalParams<Real> global)
      ->std::array<Real, 2> {
    Real cpoly_dmax = 6.0;

    Real weight;
    Real Vatr, Vrep;
    if (dist > cpoly_dmax) {
      Vatr = 0.0;
      Vrep = 0.0;
      weight = 0.0;
    } else {
      Real sigma = lj_sigma<Real>(i, j, global);
      Real epsilon = std::sqrt(i.lj_wdepth * j.lj_wdepth);
      Real d_lin = sigma * global.lj_dlin_sigma_factor;
      Real cpoly_dmin =
          sigma > 4.5 ? (sigma > cpoly_dmax - 0.1 ? cpoly_dmax - 0.1 : sigma)
                      : 4.5;

      weight = connectivity_weight<Real, Real>(bonded_path_length);
      if (dist > cpoly_dmin) {
        auto vdw_at_cpoly_dmin = vdw<Real>::V_dV(cpoly_dmin, sigma, epsilon);
        Vatr = interpolate_to_zero(
            dist,
            cpoly_dmin,
            vdw_at_cpoly_dmin.V,
            vdw_at_cpoly_dmin.dV_ddist,
            cpoly_dmax);
      } else if (dist > d_lin) {
        Vatr = vdw<Real>::V(dist, sigma, epsilon);
      } else {
        auto vdw_at_d_lin = vdw<Real>::V_dV(d_lin, sigma, epsilon);
        Vatr = (vdw_at_d_lin.V + vdw_at_d_lin.dV_ddist * (dist - d_lin));
      }

      if (dist < sigma) {
        Vrep = Vatr + epsilon;
        Vatr = -epsilon;
      } else {
        Vrep = 0.0;
      }
    }
    return {weight * Vatr, weight * Vrep};
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

    // Real d_lin = sigma * 0.6;
    Real d_lin = sigma * global.lj_dlin_sigma_factor;

    Real cpoly_dmin = 4.5;
    if (sigma > cpoly_dmin) cpoly_dmin = sigma;

    Real cpoly_dmax = 6.0;
    if (cpoly_dmin > cpoly_dmax - 0.1) cpoly_dmin = cpoly_dmax - 0.1;

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

    Real Vatr, Vrep, d_Vatr_dd, d_Vrep_dd;
    if (dist < sigma) {
      Vatr = -epsilon;
      Vrep = lj + epsilon;
      d_Vatr_dd = 0.0;
      d_Vrep_dd = d_lj_d_dist;
    } else {
      Vatr = lj;
      Vrep = 0.0;
      d_Vatr_dd = d_lj_d_dist;
      d_Vrep_dd = 0.0;
    }

    return {
        weight * Vatr, weight * Vrep, weight * d_Vatr_dd, weight * d_Vrep_dd};
  }
};

#undef def

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
