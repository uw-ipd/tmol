#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>

#include <tmol/score/common/geom.hh>
#include <tmol/score/ljlk/potentials/lk_isotropic.hh>

#include "scalar_params.hh"

namespace tmol {
namespace score {
namespace lk_ball {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

enum lk_ball_score_type {
  w_lk_ball_iso = 0,
  w_lk_ball,
  w_lk_bridge,
  w_lk_bridge_uncpl,
  n_lk_ball_score_types  // keep this last
};

#define def auto EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE

// fd  these could be exposed as global parameters
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

    static def Zero() -> dV_t { return {WatersMat::Zero(), Real3::Zero()}; }
  };

  struct V_dV_t {
    Real V;
    dV_t dV;
  };

  static constexpr Real ramp_width_A2 = lkball_globals<Real>::ramp_width_A2;

  static def square(Real v) -> Real { return v * v; }

  static def V(WatersMat WI, Real3 J, Real lj_radius_j) -> Real {
    Real d2_low = square(Real(1.4) + lj_radius_j) - ramp_width_A2;
    if (d2_low < 0.0) d2_low = 0.0;

    Real wted_d2_delta = 0;
    for (int w = 0; w < MAX_WATER; ++w) {
      if (std::isnan(WI(w, 0))) continue;
      Real d2_delta = (J - WI.row(w).transpose()).squaredNorm() - d2_low;
      wted_d2_delta += std::exp(-d2_delta);
    }

    // The piecewise ramp is already clamped when the exponential sum lies
    // outside (exp(-ramp_width_A2), 1). Avoid a logarithm in those common
    // cases.
    if (wted_d2_delta >= Real(1)) return Real(1);
    if (wted_d2_delta <= std::exp(-ramp_width_A2)) return Real(0);
    wted_d2_delta = -std::log(wted_d2_delta);

    // The clamps above imply 0 < wted_d2_delta < ramp_width_A2.
    return square(1 - square(wted_d2_delta / ramp_width_A2));
  }

  static def V_dV(WatersMat WI, Real3 J, Real lj_radius_j) -> V_dV_t {
    Real d2_low = square(Real(1.4) + lj_radius_j) - ramp_width_A2;
    if (d2_low < 0.0) d2_low = 0.0;

    Real wted_d2_delta = 0.0;
    Real3 d_wted_d2_delta_d_J = Real3::Zero();
    WatersMat d_wted_d2_delta_d_WI = WatersMat::Zero();

    for (int w = 0; w < MAX_WATER; ++w) {
      if (std::isnan(WI(w, 0))) continue;
      Real3 delta_Jw = J - WI.row(w).transpose();
      Real d2_delta = delta_Jw.squaredNorm() - d2_low;
      Real exp_d2_delta = std::exp(-d2_delta);

      wted_d2_delta += exp_d2_delta;
      d_wted_d2_delta_d_J += exp_d2_delta * delta_Jw;
      d_wted_d2_delta_d_WI.row(w).transpose() = -exp_d2_delta * delta_Jw;
    }

    Real const exp_sum = wted_d2_delta;
    if (exp_sum >= Real(1)) {
      return V_dV_t{Real(1), dV_t::Zero()};
    }
    if (exp_sum <= std::exp(-ramp_width_A2)) {
      return V_dV_t{Real(0), dV_t::Zero()};
    }
    wted_d2_delta = -std::log(wted_d2_delta);

    // The clamps above imply 0 < wted_d2_delta < ramp_width_A2.
    Real frac = square(1 - square(wted_d2_delta / ramp_width_A2));
    Real dfrac_dwted_d2 = Real(-4) * wted_d2_delta
                          * (square(ramp_width_A2) - square(wted_d2_delta))
                          / square(square(ramp_width_A2));
    Real const derivative_scale = 2 * dfrac_dwted_d2 / exp_sum;
    return V_dV_t{
        frac,
        dV_t{
            d_wted_d2_delta_d_WI * derivative_scale,
            d_wted_d2_delta_d_J * derivative_scale}};
  }

  static def dV(WatersMat WI, Real3 J, Real lj_radius_j) -> dV_t {
    return V_dV(WI, J, lj_radius_j).dV;
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

    static def Zero() -> dV_t {
      return dV_t{
          Real3::Zero(), Real3::Zero(), WatersMat::Zero(), WatersMat::Zero()};
    }
  };

  struct V_dV_t {
    Real V;
    dV_t dV;
  };

  static constexpr Real ramp_width_A2 = lkball_globals<Real>::ramp_width_A2;
  static constexpr Real overlap_gap_A2 = lkball_globals<Real>::overlap_gap_A2;
  static constexpr Real overlap_width_A2 =
      lkball_globals<Real>::overlap_width_A2;
  static constexpr Real angle_overlap_A2 =
      lkball_globals<Real>::angle_overlap_A2;

  static def square(Real v) -> Real { return v * v; }

  static def V(
      Real3 I, Real3 J, WatersMat WI, WatersMat WJ, Real lkb_water_dist)
      -> Real {
    // The bridge score has compact support in the base-atom separation. Test
    // that inexpensive condition before evaluating any water-pair exponentials.
    Real overlap_target_len2 = Real(8) / Real(3) * square(lkb_water_dist);
    Real overlap_len2 = (I - J).squaredNorm();
    Real base_delta = fabs(overlap_len2 - overlap_target_len2);
    if (base_delta > angle_overlap_A2) return Real(0);
    Real anglefrac = square(1 - square(base_delta / angle_overlap_A2));

    // water overlap
    Real wted_d2_delta = 0.0;
    for (int wi = 0; wi < MAX_WATER; wi++) {
      if (std::isnan(WI(wi, 0))) continue;
      for (int wj = 0; wj < MAX_WATER; wj++) {
        if (std::isnan(WJ(wj, 0))) continue;
        Real d2_delta =
            (WI.row(wi) - WJ.row(wj)).squaredNorm() - overlap_gap_A2;
        wted_d2_delta += std::exp(-d2_delta);
      }
    }
    Real overlapfrac;
    if (wted_d2_delta <= std::exp(-overlap_width_A2)) {
      overlapfrac = 0;
    } else if (wted_d2_delta >= Real(1)) {
      overlapfrac = 1;
    } else {
      wted_d2_delta = -std::log(wted_d2_delta);
      // square-square -> 1 as x -> 0
      overlapfrac = square(1 - square(wted_d2_delta / overlap_width_A2));
    }
    return overlapfrac * anglefrac;
  }

  static def V_dV(
      Real3 I, Real3 J, WatersMat WI, WatersMat WJ, Real lkb_water_dist)
      -> V_dV_t {
    // Reject geometrically impossible bridges before the water-overlap loop.
    Real overlap_target_len2 = Real(8) / Real(3) * square(lkb_water_dist);
    Real3 delta_ij = I - J;
    Real overlap_len2 = delta_ij.squaredNorm();
    Real base_delta = overlap_len2 - overlap_target_len2;
    if (std::abs(base_delta) > angle_overlap_A2) {
      return V_dV_t{Real(0), dV_t::Zero()};
    }
    Real3 d_wted_d2_delta_d_I = Real(2) * delta_ij;
    Real3 d_wted_d2_delta_d_J = Real(-2) * delta_ij;
    Real anglefrac = square(1 - square(base_delta / angle_overlap_A2));
    Real d_anglefrac_d_base_delta =
        std::abs(base_delta) > 0.0
            ? Real(-4) * base_delta
                  * (square(angle_overlap_A2) - square(base_delta))
                  / square(square(angle_overlap_A2))
            : Real(0);

    Real wted_d2_delta = 0.0;

    WatersMat d_wted_d2_delta_d_WI = WatersMat::Zero();
    WatersMat d_wted_d2_delta_d_WJ = WatersMat::Zero();

    for (int wi = 0; wi < MAX_WATER; wi++) {
      if (std::isnan(WI(wi, 0))) continue;
      for (int wj = 0; wj < MAX_WATER; wj++) {
        if (std::isnan(WJ(wj, 0))) continue;
        Real3 delta_ij = WI.row(wi) - WJ.row(wj);
        Real d2_delta = delta_ij.squaredNorm() - overlap_gap_A2;
        Real exp_d2_delta = std::exp(-d2_delta);

        d_wted_d2_delta_d_WI.row(wi).transpose() += exp_d2_delta * delta_ij;
        d_wted_d2_delta_d_WJ.row(wj).transpose() -= exp_d2_delta * delta_ij;

        wted_d2_delta += exp_d2_delta;
      }
    }

    Real const exp_sum = wted_d2_delta;

    Real overlapfrac;
    Real d_overlapfrac_d_wted_d2;

    if (exp_sum <= std::exp(-overlap_width_A2)) {
      return V_dV_t{Real(0), dV_t::Zero()};
    } else if (exp_sum >= Real(1)) {
      overlapfrac = 1;
      d_overlapfrac_d_wted_d2 = 0;
    } else {
      wted_d2_delta = -std::log(wted_d2_delta);
      overlapfrac = square(1 - square(wted_d2_delta / overlap_width_A2));
      d_overlapfrac_d_wted_d2 =
          Real(-4) * wted_d2_delta
          * (square(overlap_width_A2) - square(wted_d2_delta))
          / square(square(overlap_width_A2));
    }

    Real const water_derivative_scale =
        exp_sum != 0 ? 2 * anglefrac * d_overlapfrac_d_wted_d2 / exp_sum
                     : Real(0);
    return V_dV_t{
        overlapfrac * anglefrac,
        dV_t{
            overlapfrac * d_anglefrac_d_base_delta * d_wted_d2_delta_d_I,
            overlapfrac * d_anglefrac_d_base_delta * d_wted_d2_delta_d_J,
            water_derivative_scale * d_wted_d2_delta_d_WI,
            water_derivative_scale * d_wted_d2_delta_d_WJ}};
  }

  static def dV(
      Real3 I, Real3 J, WatersMat WI, WatersMat WJ, Real lkb_water_dist)
      -> dV_t {
    return V_dV(I, J, WI, WJ, lkb_water_dist).dV;
  }
};

template <typename Real>
struct lk_ball_Vt {
  Real lkball_iso;
  Real lkball;
  Real lkbridge;
  Real lkbridge_uncpl;
};

template <typename Real>
struct lk_ball_dV_dReal3 {
  typedef Eigen::Matrix<Real, 3, 1> Real3;

  Real3 d_lkball_iso;
  Real3 d_lkball;
  Real3 d_lkbridge;
  Real3 d_lkbridge_uncpl;

  static def Zero() -> lk_ball_dV_dReal3 {
    return {Real3::Zero(), Real3::Zero(), Real3::Zero(), Real3::Zero()};
  }
};

template <typename Real, int MAX_WATER>
struct lk_ball_dV_dWater {
  typedef Eigen::Matrix<Real, MAX_WATER, 3> WatersMat;

  WatersMat d_lkball_iso;
  WatersMat d_lkball;
  WatersMat d_lkbridge;
  WatersMat d_lkbridge_uncpl;

  static def Zero() -> lk_ball_dV_dWater {
    return {
        WatersMat::Zero(),
        WatersMat::Zero(),
        WatersMat::Zero(),
        WatersMat::Zero()};
  }
};

template <typename Real, int MAX_WATER>
struct lk_ball_dVt {
  lk_ball_dV_dReal3<Real> dI;
  lk_ball_dV_dReal3<Real> dJ;
  lk_ball_dV_dWater<Real, MAX_WATER> dWI;
  lk_ball_dV_dWater<Real, MAX_WATER> dWJ;

  static def Zero() -> lk_ball_dVt {
    return {
        lk_ball_dV_dReal3<Real>::Zero(),
        lk_ball_dV_dReal3<Real>::Zero(),
        lk_ball_dV_dWater<Real, MAX_WATER>::Zero(),
        lk_ball_dV_dWater<Real, MAX_WATER>::Zero()};
  }
};

template <typename Real, int MAX_WATER>
struct lk_ball_weighted_dVt {
  typedef Eigen::Matrix<Real, 3, 1> Real3;
  typedef Eigen::Matrix<Real, MAX_WATER, 3> WatersMat;

  Real3 dI;
  Real3 dJ;
  WatersMat dWI;
  WatersMat dWJ;

  static def Zero() -> lk_ball_weighted_dVt {
    return {Real3::Zero(), Real3::Zero(), WatersMat::Zero(), WatersMat::Zero()};
  }
};

template <typename Real, int MAX_WATER>
struct lk_ball_score {
  typedef Eigen::Matrix<Real, 3, 1> Real3;
  typedef Eigen::Matrix<Real, MAX_WATER, 3> WatersMat;

  static def has_water(WatersMat const& waters) -> bool {
    for (int w = 0; w < MAX_WATER; ++w) {
      if (!std::isnan(waters(w, 0))) return true;
    }
    return false;
  }

  static def V(
      Real3 I,
      Real3 J,
      WatersMat WI,
      WatersMat WJ,
      Real bonded_path_length,
      LKBallTypeParams<Real> i,
      LKBallTypeParams<Real> j,
      LKBallGlobalParams<Real> global) -> lk_ball_Vt<Real> {
    using tmol::score::common::distance;
    using tmol::score::ljlk::potentials::lj_sigma;
    using tmol::score::ljlk::potentials::lk_isotropic_pair;

    // No j-against-i score if I has no attached waters.
    if (!has_water(WI)) {
      return {0, 0, 0, 0};
    }

    Real sigma = lj_sigma<Real>(i, j, global);

    Real dist = distance<Real>::V(I, J);

    Real lk_iso_IJ = lk_isotropic_pair<Real>::V(
        dist,
        bonded_path_length,
        sigma,
        i.lj_radius,
        j.lk_volume,
        i.lk_coeff,
        i.lk_inv_lambda2,
        global.distance_threshold,
        i.is_carbon_lk && j.is_carbon_lk);
    Real frac_IJ_desolv = lk_fraction<Real, MAX_WATER>::V(WI, J, j.lj_radius);

    Real frac_IJ_water_overlap;
    if (j.is_donor || j.is_acceptor) {
      frac_IJ_water_overlap = lk_bridge_fraction<Real, MAX_WATER>::V(
          I, J, WI, WJ, global.lkb_water_dist);
    } else {
      frac_IJ_water_overlap = 0.0;
    }

    return lk_ball_Vt<Real>{
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
      LKBallTypeParams<Real> i,
      LKBallTypeParams<Real> j,
      LKBallGlobalParams<Real> global) -> lk_ball_dVt<Real, MAX_WATER> {
    using tmol::score::common::distance;
    using tmol::score::common::get;
    using tmol::score::ljlk::potentials::lj_sigma;
    using tmol::score::ljlk::potentials::lk_isotropic_pair;

    // No j-against-i score if I has no attached waters.
    if (!has_water(WI)) {
      return lk_ball_dVt<Real, MAX_WATER>::Zero();
    }

    Real sigma = lj_sigma<Real>(i, j, global);

    auto _dist = distance<Real>::V_dV(I, J);
    Real dist = _dist.V;
    Real3 d_dist_dI = _dist.dV_dA;
    Real3 d_dist_dJ = _dist.dV_dB;

    auto lk_iso_IJ = lk_isotropic_pair<Real>::V_dV(
        dist,
        bonded_path_length,
        sigma,
        i.lj_radius,
        j.lk_volume,
        i.lk_coeff,
        i.lk_inv_lambda2,
        global.distance_threshold,
        i.is_carbon_lk && j.is_carbon_lk);

    auto const frac_IJ_desolv =
        lk_fraction<Real, MAX_WATER>::V_dV(WI, J, j.lj_radius);

    typename lk_bridge_fraction<Real, MAX_WATER>::V_dV_t frac_IJ_water_overlap;
    if (j.is_donor || j.is_acceptor) {
      frac_IJ_water_overlap = lk_bridge_fraction<Real, MAX_WATER>::V_dV(
          I, J, WI, WJ, global.lkb_water_dist);
    } else {
      frac_IJ_water_overlap = {0.0, decltype(frac_IJ_water_overlap.dV)::Zero()};
    }

    Real3 d_lk_iso_IJ_dI = lk_iso_IJ.dV_ddist * d_dist_dI;
    Real3 d_lk_iso_IJ_dJ = lk_iso_IJ.dV_ddist * d_dist_dJ;

    return lk_ball_dVt<Real, MAX_WATER>{
        // dI
        lk_ball_dV_dReal3<Real>{
            d_lk_iso_IJ_dI,
            // d(lk_iso_IJ.V * frac_IJ_desolv)
            // d(frac_IJ_desolv)/dI == 0
            d_lk_iso_IJ_dI * frac_IJ_desolv.V,
            // d(lk_iso_IJ.V * frac_IJ_water_overlap)
            d_lk_iso_IJ_dI * frac_IJ_water_overlap.V
                + lk_iso_IJ.V * frac_IJ_water_overlap.dV.dI,
            // d(frac_IJ_water_overlap / 2)
            frac_IJ_water_overlap.dV.dI / 2,
        },
        // dJ
        lk_ball_dV_dReal3<Real>{
            // d(lk_iso_IJ.V)
            d_lk_iso_IJ_dJ,
            // d(lk_iso_IJ.V * frac_IJ_desolv)
            d_lk_iso_IJ_dJ * frac_IJ_desolv.V
                + lk_iso_IJ.V * frac_IJ_desolv.dV.dJ,
            // d(lk_iso_IJ.V * frac_IJ_water_overlap)
            d_lk_iso_IJ_dJ * frac_IJ_water_overlap.V
                + lk_iso_IJ.V * frac_IJ_water_overlap.dV.dJ,
            // d(frac_IJ_water_overlap / 2)
            frac_IJ_water_overlap.dV.dJ / 2,
        },
        // dWI
        lk_ball_dV_dWater<Real, MAX_WATER>{
            // d(lk_iso_IJ.V)
            WatersMat::Zero(),

            // d(lk_iso_IJ.V * frac_IJ_desolv)
            // d(lk_iso_IJ.V)/dWI == 0
            lk_iso_IJ.V * frac_IJ_desolv.dV.dWI,

            // d(lk_iso_IJ.V * frac_IJ_water_overlap)
            // d(lk_iso_IJ.V)/dWI == 0
            lk_iso_IJ.V * frac_IJ_water_overlap.dV.dWI,

            // d(frac_IJ_water_overlap / 2)
            frac_IJ_water_overlap.dV.dWI / 2},
        // dWJ
        lk_ball_dV_dWater<Real, MAX_WATER>{
            // d(lk_iso_IJ.V)
            WatersMat::Zero(),

            // d(lk_iso_IJ.V * frac_IJ_desolv)
            WatersMat::Zero(),

            // d(lk_iso_IJ.V * frac_IJ_water_overlap)
            // d(lk_iso_IJ.V)/dWJ == 0
            lk_iso_IJ.V * frac_IJ_water_overlap.dV.dWJ,

            // d(frac_IJ_water_overlap / 2)
            frac_IJ_water_overlap.dV.dWJ / 2}};
  }

  template <bool UseSharedScratch = false>
  static def weighted_dV(
      Real3 I,
      Real3 J,
      WatersMat WI,
      WatersMat WJ,
      Real bonded_path_length,
      LKBallTypeParams<Real> i,
      LKBallTypeParams<Real> j,
      LKBallGlobalParams<Real> global,
      Vec<Real, 4> weights) -> lk_ball_weighted_dVt<Real, MAX_WATER> {
    using tmol::score::common::distance;
    using tmol::score::ljlk::potentials::lj_sigma;
    using tmol::score::ljlk::potentials::lk_isotropic_pair;

    if (!has_water(WI)) {
      return lk_ball_weighted_dVt<Real, MAX_WATER>::Zero();
    }

    Real const sigma = lj_sigma<Real>(i, j, global);
    auto const dist = distance<Real>::V_dV(I, J);
    auto const lk_iso = lk_isotropic_pair<Real>::V_dV(
        dist.V,
        bonded_path_length,
        sigma,
        i.lj_radius,
        j.lk_volume,
        i.lk_coeff,
        i.lk_inv_lambda2,
        global.distance_threshold,
        i.is_carbon_lk && j.is_carbon_lk);

    auto frac_desolv = lk_fraction<Real, MAX_WATER>::V_dV(WI, J, j.lj_radius);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
    // Hopper has enough shared memory to keep occupancy while these values
    // leave registers during the bridge calculation. Each lane owns its
    // column; volatile stores and reloads shorten the register lifetimes.
    __shared__ volatile Real desolv_scratch[4 + MAX_WATER * 3][32];
    if constexpr (UseSharedScratch) {
      desolv_scratch[0][threadIdx.x] = frac_desolv.V;
#pragma unroll
      for (int d = 0; d < 3; ++d) {
        desolv_scratch[1 + d][threadIdx.x] = frac_desolv.dV.dJ[d];
#pragma unroll
        for (int w = 0; w < MAX_WATER; ++w) {
          desolv_scratch[4 + d * MAX_WATER + w][threadIdx.x] =
              frac_desolv.dV.dWI(w, d);
        }
      }
    }
#endif

    typename lk_bridge_fraction<Real, MAX_WATER>::V_dV_t frac_bridge;
    if (j.is_donor || j.is_acceptor) {
      frac_bridge = lk_bridge_fraction<Real, MAX_WATER>::V_dV(
          I, J, WI, WJ, global.lkb_water_dist);
    } else {
      frac_bridge = {0.0, decltype(frac_bridge.dV)::Zero()};
    }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
    if constexpr (UseSharedScratch) {
      frac_desolv.V = desolv_scratch[0][threadIdx.x];
#pragma unroll
      for (int d = 0; d < 3; ++d) {
        frac_desolv.dV.dJ[d] = desolv_scratch[1 + d][threadIdx.x];
#pragma unroll
        for (int w = 0; w < MAX_WATER; ++w) {
          frac_desolv.dV.dWI(w, d) =
              desolv_scratch[4 + d * MAX_WATER + w][threadIdx.x];
        }
      }
    }
#endif

    Real const iso_scale = weights[w_lk_ball_iso]
                           + weights[w_lk_ball] * frac_desolv.V
                           + weights[w_lk_bridge] * frac_bridge.V;
    Real const bridge_scale = weights[w_lk_bridge] * lk_iso.V
                              + weights[w_lk_bridge_uncpl] * Real(0.5);
    Real3 const d_iso_dI = lk_iso.dV_ddist * dist.dV_dA;

    return lk_ball_weighted_dVt<Real, MAX_WATER>{
        d_iso_dI * iso_scale + frac_bridge.dV.dI * bridge_scale,
        -d_iso_dI * iso_scale
            + frac_desolv.dV.dJ * (weights[w_lk_ball] * lk_iso.V)
            + frac_bridge.dV.dJ * bridge_scale,
        frac_desolv.dV.dWI * (weights[w_lk_ball] * lk_iso.V)
            + frac_bridge.dV.dWI * bridge_scale,
        frac_bridge.dV.dWJ * bridge_scale};
  }
};

#undef def

}  // namespace potentials
}  // namespace lk_ball
}  // namespace score
}  // namespace tmol
