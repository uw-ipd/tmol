#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>

#include <tmol/score/common/geom.hh>
#include <tmol/score/ljlk/potentials/lk_isotropic.hh>

namespace tmol {
namespace score {
namespace lk_ball {
namespace potentials {

#define def auto EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE

using tmol::score::ljlk::potentials::LJGlobalParams;
using tmol::score::ljlk::potentials::LKTypeParams;

template <typename Real>
struct lkball_globals {
  static constexpr Real overlap_gap_A2 = 0.5;
  static constexpr Real overlap_width_A2 = 2.6;
  static constexpr Real angle_overlap_A2 = 2.8 * overlap_width_A2;
  static constexpr Real ramp_width_A2 = 3.709;
};

template <typename Real, int MAX_WATER>
struct lk_fraction {
  typedef Eigen::Matrix<Real, 3, 1> Real3;
  typedef Eigen::Matrix<Real, MAX_WATER, 3> WatersMat;

  struct dV_t {
    WatersMat dWI;
    Real3 dJ;

    def astuple() { return tmol::score::common::make_tuple(dWI, dJ); }

    static def Zero()->dV_t { return {WatersMat::Zero(), Real3::Zero()}; }
  };

  static constexpr Real ramp_width_A2 = lkball_globals<Real>::ramp_width_A2;

  static def square(Real v)->Real { return v * v; }

  static def V(WatersMat WI, Real3 J, Real lj_radius_j)->Real {
    Real d2_low = std::max(0.0, square(1.4 + lj_radius_j) - ramp_width_A2);

    Real wted_d2_delta = 0;
    for (int w = 0; w < MAX_WATER; ++w) {
      Real d2_delta = (J - WI.row(w).transpose()).squaredNorm() - d2_low;
      if (!std::isnan(d2_delta)) {
        wted_d2_delta += std::exp(-d2_delta);
      }
    }

    wted_d2_delta = -std::log(wted_d2_delta);

    Real frac = 0;
    if (wted_d2_delta < 0) {
      frac = 1;
    } else if (wted_d2_delta < ramp_width_A2) {
      frac = square(1 - square(wted_d2_delta / ramp_width_A2));
    }

    return frac;
  }

  static def dV(WatersMat WI, Real3 J, Real lj_radius_j)->dV_t {
    Real d2_low = std::max(0.0, square(1.4 + lj_radius_j) - ramp_width_A2);

    Real wted_d2_delta = 0.0;
    Real3 d_wted_d2_delta_d_J = Real3::Zero();
    WatersMat d_wted_d2_delta_d_WI = WatersMat::Zero();

    for (int w = 0; w < MAX_WATER; ++w) {
      Real3 delta_Jw = J - WI.row(w).transpose();
      Real d2_delta = delta_Jw.squaredNorm() - d2_low;

      if (!std::isnan(d2_delta)) {
        Real exp_d2_delta = std::exp(-d2_delta);

        wted_d2_delta += exp_d2_delta;
        d_wted_d2_delta_d_J += 2 * exp_d2_delta * delta_Jw;
        d_wted_d2_delta_d_WI.row(w).transpose() = -2 * exp_d2_delta * delta_Jw;
      }
    }

    d_wted_d2_delta_d_J /= wted_d2_delta;
    d_wted_d2_delta_d_WI /= wted_d2_delta;

    wted_d2_delta = -std::log(wted_d2_delta);

    Real dfrac_dwted_d2 = 0;
    if (wted_d2_delta > 0 && wted_d2_delta < ramp_width_A2) {
      dfrac_dwted_d2 = -4.0 * wted_d2_delta
                       * (square(ramp_width_A2) - square(wted_d2_delta))
                       / square(square(ramp_width_A2));
    }

    return dV_t{d_wted_d2_delta_d_WI * dfrac_dwted_d2,
                d_wted_d2_delta_d_J * dfrac_dwted_d2};
  }
};

template <typename Real, int MAX_WATER>
struct lk_bridge_fraction {
  typedef Eigen::Matrix<Real, 3, 1> Real3;
  typedef Eigen::Matrix<Real, MAX_WATER, 3> WatersMat;

  struct dV_t {
    Real3 dI;
    Real3 dJ;
    WatersMat dWI;
    WatersMat dWJ;

    def astuple() { return tmol::score::common::make_tuple(dI, dJ, dWI, dWJ); }

    static def Zero()->dV_t {
      return dV_t{
          Real3::Zero(), Real3::Zero(), WatersMat::Zero(), WatersMat::Zero()};
    }
  };

  static constexpr Real ramp_width_A2 = lkball_globals<Real>::ramp_width_A2;
  static constexpr Real overlap_gap_A2 = lkball_globals<Real>::overlap_gap_A2;
  static constexpr Real overlap_width_A2 =
      lkball_globals<Real>::overlap_width_A2;
  static constexpr Real angle_overlap_A2 =
      lkball_globals<Real>::angle_overlap_A2;

  static def square(Real v)->Real { return v * v; }

  static def V(
      Real3 I, Real3 J, WatersMat WI, WatersMat WJ, Real lkb_water_dist)
      ->Real {
    // water overlap
    Real wted_d2_delta = 0.0;
    for (int wi = 0; wi < MAX_WATER; wi++) {
      for (int wj = 0; wj < MAX_WATER; wj++) {
        Real d2_delta =
            (WI.row(wi) - WJ.row(wj)).squaredNorm() - overlap_gap_A2;
        if (!std::isnan(d2_delta)) {
          wted_d2_delta += std::exp(-d2_delta);
        }
      }
    }
    wted_d2_delta = -std::log(wted_d2_delta);

    Real overlapfrac;
    if (wted_d2_delta > overlap_width_A2) {
      overlapfrac = 0;
    } else {
      // square-square -> 1 as x -> 0
      overlapfrac = square(1 - square(wted_d2_delta / overlap_width_A2));
    }

    // base angle
    Real overlap_target_len2 = 8.0 / 3.0 * square(lkb_water_dist);
    Real overlap_len2 = (I - J).squaredNorm();
    Real base_delta = std::abs(overlap_len2 - overlap_target_len2);

    Real anglefrac;
    if (base_delta > angle_overlap_A2) {
      anglefrac = 0;
    } else {
      // square-square -> 1 as x -> 0
      anglefrac = square(1 - square(base_delta / angle_overlap_A2));
    }

    return overlapfrac * anglefrac;
  }

  static def dV(
      Real3 I, Real3 J, WatersMat WI, WatersMat WJ, Real lkb_water_dist)
      ->dV_t {
    Real wted_d2_delta = 0.0;

    WatersMat d_wted_d2_delta_d_WI = WatersMat::Zero();
    WatersMat d_wted_d2_delta_d_WJ = WatersMat::Zero();

    for (int wi = 0; wi < MAX_WATER; wi++) {
      for (int wj = 0; wj < MAX_WATER; wj++) {
        Real3 delta_ij = WI.row(wi) - WJ.row(wj);
        Real d2_delta = delta_ij.squaredNorm() - overlap_gap_A2;

        if (!std::isnan(d2_delta)) {
          Real exp_d2_delta = std::exp(-d2_delta);

          d_wted_d2_delta_d_WI.row(wi).transpose() +=
              2 * exp_d2_delta * delta_ij;
          d_wted_d2_delta_d_WJ.row(wj).transpose() -=
              2 * exp_d2_delta * delta_ij;

          wted_d2_delta += std::exp(-d2_delta);
        }
      }
    }

    if (wted_d2_delta != 0) {
      d_wted_d2_delta_d_WI /= wted_d2_delta;
      d_wted_d2_delta_d_WJ /= wted_d2_delta;
    }

    wted_d2_delta = -std::log(wted_d2_delta);

    Real overlapfrac;
    Real d_overlapfrac_d_wted_d2;

    if (wted_d2_delta > overlap_width_A2) {
      overlapfrac = 0.0;
      d_overlapfrac_d_wted_d2 = 0.0;
    } else if (wted_d2_delta > 0) {
      overlapfrac = square(1 - square(wted_d2_delta / overlap_width_A2));
      d_overlapfrac_d_wted_d2 =
          -4.0 * wted_d2_delta
          * (square(overlap_width_A2) - square(wted_d2_delta))
          / square(square(overlap_width_A2));
    } else {
      overlapfrac = 1.0;
      d_overlapfrac_d_wted_d2 = 0.0;
    }

    // base angle
    Real overlap_target_len2 = 8.0 / 3.0 * square(lkb_water_dist);
    Real3 delta_ij = I - J;
    Real overlap_len2 = delta_ij.squaredNorm();
    Real base_delta = overlap_len2 - overlap_target_len2;
    Real3 d_wted_d2_delta_d_I = 2.0 * delta_ij;
    Real3 d_wted_d2_delta_d_J = -2.0 * delta_ij;

    Real anglefrac;
    Real d_anglefrac_d_base_delta;
    if (std::abs(base_delta) > angle_overlap_A2) {
      anglefrac = 0.0;
      d_anglefrac_d_base_delta = 0.0;
    } else if (std::abs(base_delta) > 0.0) {
      anglefrac = square(1 - square(base_delta / angle_overlap_A2));
      d_anglefrac_d_base_delta =
          -4.0 * base_delta * (square(angle_overlap_A2) - square(base_delta))
          / square(square(angle_overlap_A2));
    } else {
      anglefrac = 1.0;
      d_anglefrac_d_base_delta = 0.0;
    }

    // final scaling
    return dV_t{
        overlapfrac * d_anglefrac_d_base_delta * d_wted_d2_delta_d_I,
        overlapfrac * d_anglefrac_d_base_delta * d_wted_d2_delta_d_J,
        anglefrac * d_overlapfrac_d_wted_d2 * d_wted_d2_delta_d_WI,
        anglefrac * d_overlapfrac_d_wted_d2 * d_wted_d2_delta_d_WJ,
    };
  }
};

template <typename Real, int MAX_WATER>
struct lk_ball_score {
  typedef Eigen::Matrix<Real, 3, 1> Real3;
  typedef Eigen::Matrix<Real, MAX_WATER, 3> WatersMat;

  struct V_t {
    Real lkball_iso;
    Real lkball;
    Real lkbridge;
    Real lkbridge_uncpl;

    auto astuple() {
      return tmol::score::common::make_tuple(
          lkball_iso, lkball, lkbridge, lkbridge_uncpl);
    }
  };

  struct dV_dReal3 {
    Real3 d_lkball_iso;
    Real3 d_lkball;
    Real3 d_lkbridge;
    Real3 d_lkbridge_uncpl;

    auto astuple() {
      return tmol::score::common::make_tuple(
          d_lkball_iso, d_lkball, d_lkbridge, d_lkbridge_uncpl);
    }

    static def Zero()->dV_dReal3 {
      return {Real3::Zero(), Real3::Zero(), Real3::Zero(), Real3::Zero()};
    }
  };

  struct dV_dWatersMat {
    WatersMat d_lkball_iso;
    WatersMat d_lkball;
    WatersMat d_lkbridge;
    WatersMat d_lkbridge_uncpl;

    auto astuple() {
      return tmol::score::common::make_tuple(
          d_lkball_iso, d_lkball, d_lkbridge, d_lkbridge_uncpl);
    }

    static def Zero()->dV_dWatersMat {
      return {WatersMat::Zero(),
              WatersMat::Zero(),
              WatersMat::Zero(),
              WatersMat::Zero()};
    }
  };

  struct dV_t {
    dV_dReal3 dI;
    dV_dReal3 dJ;
    dV_dWatersMat dWI;
    dV_dWatersMat dWJ;

    auto astuple() {
      return tmol::score::common::make_tuple(
          dI.astuple(), dJ.astuple(), dWI.astuple(), dWJ.astuple());
    }

    static def Zero()->dV_t {
      return {dV_dReal3::Zero(),
              dV_dReal3::Zero(),
              dV_dWatersMat::Zero(),
              dV_dWatersMat::Zero()};
    }
  };

  static def V(
      Real3 I,
      Real3 J,
      WatersMat WI,
      WatersMat WJ,
      Real bonded_path_length,
      Real lkb_water_dist,
      LKTypeParams<Real> i,
      LKTypeParams<Real> j,
      LJGlobalParams<Real> global)
      ->V_t {
    using tmol::score::common::distance;
    using tmol::score::ljlk::potentials::lj_sigma;
    using tmol::score::ljlk::potentials::lk_isotropic_pair_V;

    Real sigma = lj_sigma<Real>(i, j, global);

    Real dist = distance<Real>::V(I, J);

    Real lk_iso_IJ = lk_isotropic_pair_V<Real>(
        dist,
        bonded_path_length,
        sigma,
        i.lj_radius,
        i.lk_dgfree,
        i.lk_lambda,
        j.lk_volume);

    Real frac_IJ_desolv = lk_fraction<Real, MAX_WATER>::V(WI, J, j.lj_radius);

    Real frac_IJ_water_overlap;
    if (j.is_donor || j.is_acceptor) {
      frac_IJ_water_overlap =
          lk_bridge_fraction<Real, MAX_WATER>::V(I, J, WI, WJ, lkb_water_dist);
    } else {
      frac_IJ_water_overlap = 0.0;
    }

    return V_t{
        lk_iso_IJ,
        lk_iso_IJ * frac_IJ_desolv,
        lk_iso_IJ * frac_IJ_water_overlap,
        frac_IJ_water_overlap / 2,
    };
  }

  static def dV(
      Real3 I,
      Real3 J,
      WatersMat WI,
      WatersMat WJ,
      Real bonded_path_length,
      Real lkb_water_dist,
      LKTypeParams<Real> i,
      LKTypeParams<Real> j,
      LJGlobalParams<Real> global)
      ->dV_t {
    using tmol::score::common::distance;
    using tmol::score::common::get;
    using tmol::score::ljlk::potentials::lj_sigma;
    using tmol::score::ljlk::potentials::lk_isotropic_pair_V_dV;

    Real sigma = lj_sigma<Real>(i, j, global);

    auto _dist = distance<Real>::V_dV(I, J);
    Real dist = _dist.V;
    Real3 d_dist_dI = _dist.dV_dA;
    Real3 d_dist_dJ = _dist.dV_dB;

    auto _lk_iso_IJ = lk_isotropic_pair_V_dV<Real>(
        dist,
        bonded_path_length,
        sigma,
        i.lj_radius,
        i.lk_dgfree,
        i.lk_lambda,
        j.lk_volume);
    Real lk_iso_IJ = get<0>(_lk_iso_IJ);
    Real d_lk_iso_IJ_d_dist = get<1>(_lk_iso_IJ);

    Real frac_IJ_desolv = lk_fraction<Real, MAX_WATER>::V(WI, J, j.lj_radius);
    auto d_frac_IJ_desolv =
        lk_fraction<Real, MAX_WATER>::dV(WI, J, j.lj_radius);

    Real frac_IJ_water_overlap;
    typename lk_bridge_fraction<Real, MAX_WATER>::dV_t d_frac_IJ_water_overlap;
    if (j.is_donor || j.is_acceptor) {
      frac_IJ_water_overlap =
          lk_bridge_fraction<Real, MAX_WATER>::V(I, J, WI, WJ, lkb_water_dist);
      d_frac_IJ_water_overlap =
          lk_bridge_fraction<Real, MAX_WATER>::dV(I, J, WI, WJ, lkb_water_dist);
    } else {
      frac_IJ_water_overlap = 0.0;
      d_frac_IJ_water_overlap = decltype(d_frac_IJ_water_overlap)::Zero();
    }

    Real3 d_lk_iso_IJ_dI = d_lk_iso_IJ_d_dist * d_dist_dI;
    Real3 d_lk_iso_IJ_dJ = d_lk_iso_IJ_d_dist * d_dist_dJ;

    return dV_t{
        // dI
        dV_dReal3{
            d_lk_iso_IJ_dI,
            // d(lk_iso_IJ * frac_IJ_desolv)
            // d(frac_IJ_desolv)/dI == 0
            d_lk_iso_IJ_dI * frac_IJ_desolv,
            // d(lk_iso_IJ * frac_IJ_water_overlap)
            d_lk_iso_IJ_dI * frac_IJ_water_overlap
                + lk_iso_IJ * d_frac_IJ_water_overlap.dI,
            // d(frac_IJ_water_overlap / 2)
            d_frac_IJ_water_overlap.dI / 2,
        },
        // dJ
        dV_dReal3{
            // d(lk_iso_IJ)
            d_lk_iso_IJ_dJ,
            // d(lk_iso_IJ * frac_IJ_desolv)
            d_lk_iso_IJ_dJ * frac_IJ_desolv + lk_iso_IJ * d_frac_IJ_desolv.dJ,
            // d(lk_iso_IJ * frac_IJ_water_overlap)
            d_lk_iso_IJ_dJ * frac_IJ_water_overlap
                + lk_iso_IJ * d_frac_IJ_water_overlap.dJ,
            // d(frac_IJ_water_overlap / 2)
            d_frac_IJ_water_overlap.dJ / 2,
        },
        // dWI
        dV_dWatersMat{// d(lk_iso_IJ)
                      WatersMat::Zero(),

                      // d(lk_iso_IJ * frac_IJ_desolv)
                      // d(lk_iso_IJ)/dWI == 0
                      lk_iso_IJ * d_frac_IJ_desolv.dWI,

                      // d(lk_iso_IJ * frac_IJ_water_overlap)
                      // d(lk_iso_IJ)/dWI == 0
                      lk_iso_IJ * d_frac_IJ_water_overlap.dWI,

                      // d(frac_IJ_water_overlap / 2)
                      d_frac_IJ_water_overlap.dWI / 2},
        // dWJ
        dV_dWatersMat{// d(lk_iso_IJ)
                      WatersMat::Zero(),

                      // d(lk_iso_IJ * frac_IJ_desolv)
                      WatersMat::Zero(),

                      // d(lk_iso_IJ * frac_IJ_water_overlap)
                      // d(lk_iso_IJ)/dWJ == 0
                      lk_iso_IJ * d_frac_IJ_water_overlap.dWJ,

                      // d(frac_IJ_water_overlap / 2)
                      d_frac_IJ_water_overlap.dWJ / 2}};
  }
};

#undef def

}  // namespace potentials
}  // namespace lk_ball
}  // namespace score
}  // namespace tmol
