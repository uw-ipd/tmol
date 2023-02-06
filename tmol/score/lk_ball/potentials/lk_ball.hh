#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>

#include <tmol/score/common/geom.hh>
#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/count_pair.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/data_loading.hh>
#include <tmol/score/ljlk/potentials/lk_isotropic.hh>

// Operator definitions; safe for CPU compilation
#include <moderngpu/operators.hxx>

#include "params.hh"

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

    static def Zero()->dV_t { return {WatersMat::Zero(), Real3::Zero()}; }
  };

  static constexpr Real ramp_width_A2 = lkball_globals<Real>::ramp_width_A2;

  static def square(Real v)->Real { return v * v; }

  static def V(WatersMat WI, Real3 J, Real lj_radius_j)->Real {
    Real d2_low = square(1.4 + lj_radius_j) - ramp_width_A2;
    if (d2_low < 0.0) d2_low = 0.0;

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
    Real d2_low = square(1.4 + lj_radius_j) - ramp_width_A2;
    if (d2_low < 0.0) d2_low = 0.0;

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

    if (wted_d2_delta != 0) {
      d_wted_d2_delta_d_J /= wted_d2_delta;
      d_wted_d2_delta_d_WI /= wted_d2_delta;
    }

    wted_d2_delta = -std::log(wted_d2_delta);

    Real dfrac_dwted_d2 = 0;
    if (wted_d2_delta > 0 && wted_d2_delta < ramp_width_A2) {
      dfrac_dwted_d2 = -4.0 * wted_d2_delta
                       * (square(ramp_width_A2) - square(wted_d2_delta))
                       / square(square(ramp_width_A2));
    }

    // TODO nan @ 0
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
    Real base_delta = fabs(overlap_len2 - overlap_target_len2);

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

  static def Zero()->lk_ball_dV_dReal3 {
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

  static def Zero()->lk_ball_dV_dWater {
    return {WatersMat::Zero(),
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

  static def Zero()->lk_ball_dVt {
    return {lk_ball_dV_dReal3<Real>::Zero(),
            lk_ball_dV_dReal3<Real>::Zero(),
            lk_ball_dV_dWater<Real, MAX_WATER>::Zero(),
            lk_ball_dV_dWater<Real, MAX_WATER>::Zero()};
  }
};

template <typename Real, int MAX_WATER>
struct lk_ball_score {
  typedef Eigen::Matrix<Real, 3, 1> Real3;
  typedef Eigen::Matrix<Real, MAX_WATER, 3> WatersMat;

  static def V(
      Real3 I,
      Real3 J,
      WatersMat WI,
      WatersMat WJ,
      Real bonded_path_length,
      LKBallTypeParams<Real> i,
      LKBallTypeParams<Real> j,
      LKBallGlobalParams<Real> global)
      ->lk_ball_Vt<Real> {
    using tmol::score::common::distance;
    using tmol::score::ljlk::potentials::lj_sigma;
    using tmol::score::ljlk::potentials::lk_isotropic_pair;

    // No j-against-i score if I has no attached waters.
    if (std::isnan(WI(0, 0))) {
      return {0, 0, 0, 0};
    }

    Real sigma = lj_sigma<Real>(i, j, global);

    Real dist = distance<Real>::V(I, J);

    Real lk_iso_IJ = lk_isotropic_pair<Real>::V(
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
      LKBallGlobalParams<Real> global)
      ->lk_ball_dVt<Real, MAX_WATER> {
    using tmol::score::common::distance;
    using tmol::score::common::get;
    using tmol::score::ljlk::potentials::lj_sigma;
    using tmol::score::ljlk::potentials::lk_isotropic_pair;

    // No j-against-i score if I has no attached waters.
    if (std::isnan(WI(0, 0))) {
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
        i.lk_dgfree,
        i.lk_lambda,
        j.lk_volume);

    Real frac_IJ_desolv = lk_fraction<Real, MAX_WATER>::V(WI, J, j.lj_radius);
    auto d_frac_IJ_desolv =
        lk_fraction<Real, MAX_WATER>::dV(WI, J, j.lj_radius);

    Real frac_IJ_water_overlap;
    typename lk_bridge_fraction<Real, MAX_WATER>::dV_t d_frac_IJ_water_overlap;
    if (j.is_donor || j.is_acceptor) {
      frac_IJ_water_overlap = lk_bridge_fraction<Real, MAX_WATER>::V(
          I, J, WI, WJ, global.lkb_water_dist);
      d_frac_IJ_water_overlap = lk_bridge_fraction<Real, MAX_WATER>::dV(
          I, J, WI, WJ, global.lkb_water_dist);
    } else {
      frac_IJ_water_overlap = 0.0;
      d_frac_IJ_water_overlap = decltype(d_frac_IJ_water_overlap)::Zero();
    }

    Real3 d_lk_iso_IJ_dI = lk_iso_IJ.dV_ddist * d_dist_dI;
    Real3 d_lk_iso_IJ_dJ = lk_iso_IJ.dV_ddist * d_dist_dJ;

    return lk_ball_dVt<Real, MAX_WATER>{
        // dI
        lk_ball_dV_dReal3<Real>{
            d_lk_iso_IJ_dI,
            // d(lk_iso_IJ.V * frac_IJ_desolv)
            // d(frac_IJ_desolv)/dI == 0
            d_lk_iso_IJ_dI * frac_IJ_desolv,
            // d(lk_iso_IJ.V * frac_IJ_water_overlap)
            d_lk_iso_IJ_dI * frac_IJ_water_overlap
                + lk_iso_IJ.V * d_frac_IJ_water_overlap.dI,
            // d(frac_IJ_water_overlap / 2)
            d_frac_IJ_water_overlap.dI / 2,
        },
        // dJ
        lk_ball_dV_dReal3<Real>{
            // d(lk_iso_IJ.V)
            d_lk_iso_IJ_dJ,
            // d(lk_iso_IJ.V * frac_IJ_desolv)
            d_lk_iso_IJ_dJ * frac_IJ_desolv + lk_iso_IJ.V * d_frac_IJ_desolv.dJ,
            // d(lk_iso_IJ.V * frac_IJ_water_overlap)
            d_lk_iso_IJ_dJ * frac_IJ_water_overlap
                + lk_iso_IJ.V * d_frac_IJ_water_overlap.dJ,
            // d(frac_IJ_water_overlap / 2)
            d_frac_IJ_water_overlap.dJ / 2,
        },
        // dWI
        lk_ball_dV_dWater<Real, MAX_WATER>{
            // d(lk_iso_IJ.V)
            WatersMat::Zero(),

            // d(lk_iso_IJ.V * frac_IJ_desolv)
            // d(lk_iso_IJ.V)/dWI == 0
            lk_iso_IJ.V * d_frac_IJ_desolv.dWI,

            // d(lk_iso_IJ.V * frac_IJ_water_overlap)
            // d(lk_iso_IJ.V)/dWI == 0
            lk_iso_IJ.V * d_frac_IJ_water_overlap.dWI,

            // d(frac_IJ_water_overlap / 2)
            d_frac_IJ_water_overlap.dWI / 2},
        // dWJ
        lk_ball_dV_dWater<Real, MAX_WATER>{
            // d(lk_iso_IJ.V)
            WatersMat::Zero(),

            // d(lk_iso_IJ.V * frac_IJ_desolv)
            WatersMat::Zero(),

            // d(lk_iso_IJ.V * frac_IJ_water_overlap)
            // d(lk_iso_IJ.V)/dWJ == 0
            lk_iso_IJ.V * d_frac_IJ_water_overlap.dWJ,

            // d(frac_IJ_water_overlap / 2)
            d_frac_IJ_water_overlap.dWJ / 2}};
  }
};

template <typename Real>
class LKBallSingleResData {
 public:
  int block_ind;
  int block_type;
  int block_coord_offset;
  int n_atoms;
  int n_conn;
  Real *pose_coords;
  Real *water_coords;
  unsigned char n_polars;
  unsigned char n_occluders;
  unsigned char *pol_occ_tile_inds;
  LKBallTypeParams<Real> *lk_ball_params;
  unsigned char *path_dist;
};

template <typename Real>
class LKBallResPairData {
 public:
  int pose_ind;
  int max_important_bond_separation;
  int min_separation;
  bool in_count_pair_striking_dist;
  unsigned char *conn_seps;

  // load global params once; store totalE's
  LKBallGlobalParams<Real> global_params;
  Real total_lk_ball_iso;
  Real total_lk_ball;
  Real total_lk_bridge;
  Real total_lk_bridge_uncpl;
};

template <typename Real>
class LKBallScoringData {
 public:
  LKBallSingleResData<Real> r1;
  LKBallSingleResData<Real> r2;
  LKBallResPairData<Real> pair_data;
};

template <typename Real, int TILE_SIZE, int MAX_N_WATER, int MAX_N_CONN>
struct LKBallBlockPairSharedData {
  Real pose_coords1[TILE_SIZE * 3];  // 768 bytes for coords
  Real pose_coords2[TILE_SIZE * 3];
  Real water_coords1[TILE_SIZE * MAX_N_WATER * 3];  // 3072 bytes for coords
  Real water_coords2[TILE_SIZE * MAX_N_WATER * 3];

  unsigned char n_polars1;  // 4 bytes for counts
  unsigned char n_polars2;
  unsigned char n_occluders1;
  unsigned char n_occluders2;
  unsigned char pol_occ_tile_inds1[TILE_SIZE];  // 64 bytes for indices
  unsigned char pol_occ_tile_inds2[TILE_SIZE];
  LKBallTypeParams<Real> lk_ball_params1[TILE_SIZE];  // 192 bytes for params
  LKBallTypeParams<Real> lk_ball_params2[TILE_SIZE];

  unsigned char conn_ats1[MAX_N_CONN];  // 8 bytes
  unsigned char conn_ats2[MAX_N_CONN];
  unsigned char path_dist1[MAX_N_CONN * TILE_SIZE];  // 256 bytes
  unsigned char path_dist2[MAX_N_CONN * TILE_SIZE];
  unsigned char conn_seps[MAX_N_CONN * MAX_N_CONN];  // 64 bytes
};

template <typename Real>
class LKBallSingleResData2 {
 public:
  int block_ind;
  int block_type;
  int block_coord_offset;
  int n_atoms;
  int n_conn;
  Real *pose_coords;
  Real *water_coords;
  unsigned char n_polars;
  unsigned char n_occluders;
  unsigned char *pol_occ_tile_inds;
  unsigned char *n_waters;
  LKBallTypeParams<Real> *lk_ball_params;
  unsigned char *path_dist;
};

template <typename Real>
class LKBallResPairData2 {
 public:
  int pose_ind;
  int max_important_bond_separation;
  int min_separation;
  bool in_count_pair_striking_dist;
  unsigned char *conn_seps;

  Real *lk_iso;
  Real *lk_fraction;
  Real *lk_bridge_fraction;
  Real *lk_fraction_exp_d2_delta_sum;
  Real *lk_bridge_fraction_exp_d2_delta_sum;

  int *n_tasks_per_line_element;
  int *task_offset_for_line_element;
  int *line_element_for_task;
  int *rank_for_task;
  int *task_is_first_task_for_line_element;

  Real *exp_d2_delta_for_task;

  // load global params once; store totalE's
  LKBallGlobalParams<Real> global_params;
  Real total_lk_ball_iso;
  Real total_lk_ball;
  Real total_lk_bridge;
  Real total_lk_bridge_uncpl;
};

template <typename Real>
class LKBallScoringData2 {
 public:
  LKBallSingleResData2<Real> r1;
  LKBallSingleResData2<Real> r2;
  LKBallResPairData2<Real> pair_data;
};

// A version of the shared data where we store a row's worth of partial
// results of the lk-ball calculation so that we can divide the work of
// processing water/occluder and water/water interactions across multiple
// threads instead of putting all the work onto a small handful of
// threads
template <typename Real, int TILE_SIZE, int MAX_N_WATER, int MAX_N_CONN>
struct LKBallBlockPairSharedData2 {
  Real pose_coords1[TILE_SIZE * 3];  // 768 bytes for coords
  Real pose_coords2[TILE_SIZE * 3];
  Real water_coords1[TILE_SIZE * MAX_N_WATER * 3];  // 3072 bytes for coords
  Real water_coords2[TILE_SIZE * MAX_N_WATER * 3];

  unsigned char n_polars1;  // 4 bytes for counts
  unsigned char n_polars2;
  unsigned char n_occluders1;
  unsigned char n_occluders2;
  unsigned char pol_occ_tile_inds1[TILE_SIZE];  // 64 bytes for indices
  unsigned char pol_occ_tile_inds2[TILE_SIZE];
  unsigned char n_waters1[TILE_SIZE];
  unsigned char n_waters2[TILE_SIZE];
  LKBallTypeParams<Real> lk_ball_params1[TILE_SIZE];  // 192 bytes for params
  LKBallTypeParams<Real> lk_ball_params2[TILE_SIZE];

  unsigned char conn_ats1[MAX_N_CONN];  // 8 bytes
  unsigned char conn_ats2[MAX_N_CONN];
  unsigned char path_dist1[MAX_N_CONN * TILE_SIZE];  // 256 bytes
  unsigned char path_dist2[MAX_N_CONN * TILE_SIZE];
  unsigned char conn_seps[MAX_N_CONN * MAX_N_CONN];  // 64 bytes

  // Real lk_iso[TILE_SIZE];
  // int line_n_waters[ITLE_SIZE];
  // int line_ind_for_water[TILE_SIZE];

  Real lk_iso[TILE_SIZE];
  Real lk_fraction[TILE_SIZE];
  Real lk_bridge_fraction[TILE_SIZE];
  Real lk_fraction_exp_d2_delta_sum[TILE_SIZE];
  Real lk_bridge_fraction_exp_d2_delta_sum[TILE_SIZE];

  Real exp_d2_delta_for_task[TILE_SIZE];

  // This data feels like something that could be a struct
  // that's used in multiple parallelize-subtasks-within-
  // a-workgroup contexts
  int n_tasks_per_line_element[TILE_SIZE];
  int task_offset_for_line_element[TILE_SIZE];
  int line_element_for_task[TILE_SIZE];
  int rank_for_task[TILE_SIZE];
  int task_is_first_task_for_line_element[TILE_SIZE];
};

#undef def

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    int MAX_N_WATER,
    typename Real,
    typename Int>
void TMOL_DEVICE_FUNC lk_ball_load_block_coords_and_params_into_shared(
    TView<Vec<Real, 3>, 2, Dev> pose_coords,
    TView<Vec<Real, 3>, 3, Dev> water_coords,
    TView<Int, 3, Dev> block_type_tile_pol_occ_inds,
    TView<LKBallTypeParams<Real>, 3, Dev> block_type_tile_lk_ball_params,
    int pose_ind,
    int tile_ind,
    LKBallSingleResData<Real> &r_dat,
    int n_atoms_to_load,
    int start_atom) {
  // pre-condition: n_atoms_to_load < TILE_SIZE
  // Note that TILE_SIZE is not explicitly passed in, but is "present"
  // in r_dat.coords allocation

  DeviceDispatch<Dev>::template copy_contiguous_data<nt, 3>(
      r_dat.pose_coords,
      reinterpret_cast<Real *>(
          &pose_coords[pose_ind][r_dat.block_coord_offset + start_atom]),
      n_atoms_to_load * 3);
  DeviceDispatch<Dev>::template copy_contiguous_data<nt, MAX_N_WATER * 3>(
      r_dat.water_coords,
      reinterpret_cast<Real *>(
          &water_coords[pose_ind][r_dat.block_coord_offset + start_atom][0]),
      n_atoms_to_load * MAX_N_WATER * 3);
  DeviceDispatch<Dev>::template copy_contiguous_data_and_cast<nt, 1>(
      r_dat.pol_occ_tile_inds,
      &block_type_tile_pol_occ_inds[r_dat.block_type][tile_ind][0],
      r_dat.n_occluders);
  int const N_PARAMS = sizeof(LKBallTypeParams<Real>) / sizeof(Real);
  DeviceDispatch<Dev>::template copy_contiguous_data<nt, N_PARAMS>(
      reinterpret_cast<Real *>(r_dat.lk_ball_params),
      reinterpret_cast<Real *>(
          &block_type_tile_lk_ball_params[r_dat.block_type][tile_ind][0]),
      r_dat.n_occluders * N_PARAMS);
}

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    int TILE_SIZE,
    int MAX_N_WATER,
    typename Real,
    typename Int>
void TMOL_DEVICE_FUNC lk_ball_load_block_into_shared(
    TView<Vec<Real, 3>, 2, Dev> pose_coords,
    TView<Vec<Real, 3>, 3, Dev> water_coords,
    TView<Int, 3, Dev> block_type_tile_pol_occ_inds,
    TView<LKBallTypeParams<Real>, 3, Dev> block_type_tile_lk_ball_params,
    TView<Int, 3, Dev> block_type_path_distance,
    int pose_ind,
    int tile_ind,
    LKBallSingleResData<Real> &r_dat,
    int n_atoms_to_load,
    int start_atom,
    bool count_pair_striking_dist,
    unsigned char *__restrict__ conn_ats) {
  lk_ball_load_block_coords_and_params_into_shared<
      DeviceDispatch,
      Dev,
      nt,
      MAX_N_WATER>(
      pose_coords,
      water_coords,
      block_type_tile_pol_occ_inds,
      block_type_tile_lk_ball_params,
      pose_ind,
      tile_ind,
      r_dat,
      n_atoms_to_load,
      start_atom);

  auto copy_path_dists = ([=](int tid) {
    for (int count = tid; count < n_atoms_to_load; count += nt) {
      int const atid = start_atom + count;
      for (int j = 0; j < r_dat.n_conn; ++j) {
        unsigned char ij_path_dist =
            block_type_path_distance[r_dat.block_type][conn_ats[j]][atid];
        r_dat.path_dist[j * TILE_SIZE + count] = ij_path_dist;
      }
    }
  });
  if (count_pair_striking_dist) {
    DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(copy_path_dists);
  }
}

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    typename Int,
    typename Real,
    int TILE_SIZE,
    int MAX_N_WATER,
    int MAX_N_CONN>
void TMOL_DEVICE_FUNC lk_ball_load_tile_invariant_interres_data(
    TView<Int, 2, Dev> pose_stack_block_coord_offset,
    TView<Int, 2, Dev> pose_stack_block_type,
    TView<Vec<Int, 2>, 3, Dev> pose_stack_inter_residue_connections,
    TView<Int, 3, Dev> pose_stack_min_bond_separation,
    TView<Int, 5, Dev> pose_stack_inter_block_bondsep,
    TView<Int, 1, Dev> block_type_n_interblock_bonds,
    TView<Int, 2, Dev> block_type_atoms_forming_chemical_bonds,
    TView<LKBallGlobalParams<Real>, 1, Dev> global_params,
    int const max_important_bond_separation,
    int pose_ind,
    int block_ind1,
    int block_ind2,
    int block_type1,
    int block_type2,
    int n_atoms1,
    int n_atoms2,
    LKBallScoringData<Real> &inter_dat,
    LKBallBlockPairSharedData<Real, TILE_SIZE, MAX_N_WATER, MAX_N_CONN>
        &shared_m) {
  inter_dat.pair_data.pose_ind = pose_ind;
  inter_dat.r1.block_ind = block_ind1;
  inter_dat.r2.block_ind = block_ind2;
  inter_dat.r1.block_type = block_type1;
  inter_dat.r2.block_type = block_type2;
  inter_dat.r1.block_coord_offset =
      pose_stack_block_coord_offset[pose_ind][block_ind1];
  inter_dat.r2.block_coord_offset =
      pose_stack_block_coord_offset[pose_ind][block_ind2];
  inter_dat.pair_data.max_important_bond_separation =
      max_important_bond_separation;
  inter_dat.pair_data.min_separation =
      pose_stack_min_bond_separation[pose_ind][block_ind1][block_ind2];
  inter_dat.pair_data.in_count_pair_striking_dist =
      inter_dat.pair_data.min_separation <= max_important_bond_separation;
  inter_dat.r1.n_atoms = n_atoms1;
  inter_dat.r2.n_atoms = n_atoms2;
  inter_dat.r1.n_conn = block_type_n_interblock_bonds[block_type1];
  inter_dat.r2.n_conn = block_type_n_interblock_bonds[block_type2];

  // set the pointers in inter_dat to point at the shared-memory arrays
  inter_dat.r1.pose_coords = shared_m.pose_coords1;
  inter_dat.r2.pose_coords = shared_m.pose_coords2;
  inter_dat.r1.water_coords = shared_m.water_coords1;
  inter_dat.r2.water_coords = shared_m.water_coords2;
  inter_dat.r1.pol_occ_tile_inds = shared_m.pol_occ_tile_inds1;
  inter_dat.r2.pol_occ_tile_inds = shared_m.pol_occ_tile_inds2;
  inter_dat.r1.lk_ball_params = shared_m.lk_ball_params1;
  inter_dat.r2.lk_ball_params = shared_m.lk_ball_params2;

  inter_dat.r1.path_dist = shared_m.path_dist1;
  inter_dat.r2.path_dist = shared_m.path_dist2;
  inter_dat.pair_data.conn_seps = shared_m.conn_seps;

  // Count pair setup that does not depend on which tile we are
  // operating on; only necessary if r1 and r2 are within
  // a minimum number of chemical bonds separation
  if (inter_dat.pair_data.in_count_pair_striking_dist) {
    // Load data into shared arrays
    auto load_count_pair_conn_at_data = ([&](int tid) {
      int n_conn_tot = inter_dat.r1.n_conn + inter_dat.r2.n_conn
                       + inter_dat.r1.n_conn * inter_dat.r2.n_conn;
      for (int count = tid; count < n_conn_tot; count += nt) {
        if (count < inter_dat.r1.n_conn) {
          int const conn_ind = count;
          shared_m.conn_ats1[conn_ind] =
              block_type_atoms_forming_chemical_bonds[block_type1][conn_ind];
        } else if (count < inter_dat.r1.n_conn + inter_dat.r2.n_conn) {
          int const conn_ind = count - inter_dat.r1.n_conn;
          shared_m.conn_ats2[conn_ind] =
              block_type_atoms_forming_chemical_bonds[block_type2][conn_ind];
        } else {
          int const conn_ind =
              count - inter_dat.r1.n_conn - inter_dat.r2.n_conn;
          int conn1 = conn_ind / inter_dat.r2.n_conn;
          int conn2 = conn_ind % inter_dat.r2.n_conn;
          shared_m.conn_seps[conn_ind] =
              pose_stack_inter_block_bondsep[pose_ind][block_ind1][block_ind2]
                                            [conn1][conn2];
        }
      }
    });
    DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(
        load_count_pair_conn_at_data);
  }

  // Final data members
  inter_dat.pair_data.global_params = global_params[0];

  // Set initial energy totals to 0
  inter_dat.pair_data.total_lk_ball_iso = 0;
  inter_dat.pair_data.total_lk_ball = 0;
  inter_dat.pair_data.total_lk_bridge = 0;
  inter_dat.pair_data.total_lk_bridge_uncpl = 0;
}

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    int TILE_SIZE,
    int MAX_N_WATER,
    int MAX_N_CONN,
    typename Real,
    typename Int>
void TMOL_DEVICE_FUNC lk_ball_load_interres1_tile_data_to_shared(
    TView<Vec<Real, 3>, 2, Dev> pose_coords,
    TView<Vec<Real, 3>, 3, Dev> water_coords,
    TView<Int, 2, Dev> block_type_tile_n_polar_atoms,
    TView<Int, 2, Dev> block_type_tile_n_occluder_atoms,
    TView<Int, 3, Dev> block_type_tile_pol_occ_inds,
    TView<LKBallTypeParams<Real>, 3, Dev> block_type_tile_lk_ball_params,
    TView<Int, 3, Dev> block_type_path_distance,
    int tile_ind,
    int start_atom1,
    int n_atoms_to_load1,
    LKBallScoringData<Real> &inter_dat,
    LKBallBlockPairSharedData<Real, TILE_SIZE, MAX_N_WATER, MAX_N_CONN>
        &shared_m) {
  auto store_n_pol_n_occ1 = ([&](int tid) {
    int n_pol =
        block_type_tile_n_polar_atoms[inter_dat.r1.block_type][tile_ind];
    int n_occ =
        block_type_tile_n_occluder_atoms[inter_dat.r1.block_type][tile_ind];
    inter_dat.r1.n_polars = n_pol;
    inter_dat.r1.n_occluders = n_occ;
    if (tid == 0) {
      shared_m.n_polars1 = n_pol;
      shared_m.n_occluders1 = n_occ;
    }
  });
  DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(store_n_pol_n_occ1);

  lk_ball_load_block_into_shared<
      DeviceDispatch,
      Dev,
      nt,
      TILE_SIZE,
      MAX_N_WATER>(
      pose_coords,
      water_coords,
      block_type_tile_pol_occ_inds,
      block_type_tile_lk_ball_params,
      block_type_path_distance,
      inter_dat.pair_data.pose_ind,
      tile_ind,
      inter_dat.r1,
      n_atoms_to_load1,
      start_atom1,
      inter_dat.pair_data.in_count_pair_striking_dist,
      shared_m.conn_ats1);
}

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    int TILE_SIZE,
    int MAX_N_WATER,
    int MAX_N_CONN,
    typename Real,
    typename Int>
void TMOL_DEVICE_FUNC lk_ball_load_interres2_tile_data_to_shared(
    TView<Vec<Real, 3>, 2, Dev> pose_coords,
    TView<Vec<Real, 3>, 3, Dev> water_coords,
    TView<Int, 2, Dev> block_type_tile_n_polar_atoms,
    TView<Int, 2, Dev> block_type_tile_n_occluder_atoms,
    TView<Int, 3, Dev> block_type_tile_pol_occ_inds,
    TView<LKBallTypeParams<Real>, 3, Dev> block_type_tile_lk_ball_params,
    TView<Int, 3, Dev> block_type_path_distance,
    int tile_ind,
    int start_atom2,
    int n_atoms_to_load2,
    LKBallScoringData<Real> &inter_dat,
    LKBallBlockPairSharedData<Real, TILE_SIZE, MAX_N_WATER, MAX_N_CONN>
        &shared_m) {
  auto store_n_pol_n_occ2 = ([&](int tid) {
    int n_pol =
        block_type_tile_n_polar_atoms[inter_dat.r2.block_type][tile_ind];
    int n_occ =
        block_type_tile_n_occluder_atoms[inter_dat.r2.block_type][tile_ind];
    inter_dat.r2.n_polars = n_pol;
    inter_dat.r2.n_occluders = n_occ;
    if (tid == 0) {
      shared_m.n_polars2 = n_pol;
      shared_m.n_occluders2 = n_occ;
    }
  });
  DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(store_n_pol_n_occ2);

  lk_ball_load_block_into_shared<
      DeviceDispatch,
      Dev,
      nt,
      TILE_SIZE,
      MAX_N_WATER>(
      pose_coords,
      water_coords,
      block_type_tile_pol_occ_inds,
      block_type_tile_lk_ball_params,
      block_type_path_distance,
      inter_dat.pair_data.pose_ind,
      tile_ind,
      inter_dat.r2,
      n_atoms_to_load2,
      start_atom2,
      inter_dat.pair_data.in_count_pair_striking_dist,
      shared_m.conn_ats2);
}

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    typename Int,
    typename Real,
    int TILE_SIZE,
    int MAX_N_WATER,
    int MAX_N_CONN>
void TMOL_DEVICE_FUNC lk_ball_load_tile_invariant_intrares_data(
    TView<Int, 2, Dev> pose_stack_block_coord_offset,
    TView<Int, 2, Dev> pose_stack_block_type,
    TView<LKBallGlobalParams<Real>, 1, Dev> global_params,
    int const max_important_bond_separation,
    int pose_ind,
    int block_ind1,
    int block_type1,
    int n_atoms1,
    LKBallScoringData<Real> &intra_dat,
    LKBallBlockPairSharedData<Real, TILE_SIZE, MAX_N_WATER, MAX_N_CONN>
        &shared_m) {
  intra_dat.pair_data.pose_ind = pose_ind;
  intra_dat.r1.block_ind = block_ind1;
  intra_dat.r2.block_ind = block_ind1;
  intra_dat.r1.block_type = block_type1;
  intra_dat.r2.block_type = block_type1;
  intra_dat.r1.block_coord_offset =
      pose_stack_block_coord_offset[pose_ind][block_ind1];
  intra_dat.r2.block_coord_offset = intra_dat.r1.block_coord_offset;
  intra_dat.pair_data.max_important_bond_separation =
      max_important_bond_separation;

  // we are not going to load count pair data into shared memory because
  // we are not going to use that data from shared memory; setting
  // in_count_pair_striking_distance to false prevents downstream loading
  // of count-pair data, which waste memory bandwidth
  intra_dat.pair_data.min_separation = 0;
  intra_dat.pair_data.in_count_pair_striking_dist = false;

  intra_dat.r1.n_atoms = n_atoms1;
  intra_dat.r2.n_atoms = n_atoms1;
  intra_dat.r1.n_conn = 0;
  intra_dat.r2.n_conn = 0;

  // set the pointers in intra_dat to point at the
  // shared-memory arrays. Note that these arrays will be reset
  // later because which shared memory arrays we will use depends on
  // which tile pair we are evaluating!
  intra_dat.r1.pose_coords = shared_m.pose_coords1;
  intra_dat.r2.pose_coords = shared_m.pose_coords2;
  intra_dat.r1.water_coords = shared_m.water_coords1;
  intra_dat.r2.water_coords = shared_m.water_coords2;
  intra_dat.r1.pol_occ_tile_inds = shared_m.pol_occ_tile_inds1;
  intra_dat.r2.pol_occ_tile_inds = shared_m.pol_occ_tile_inds2;
  intra_dat.r1.lk_ball_params = shared_m.lk_ball_params1;
  intra_dat.r2.lk_ball_params = shared_m.lk_ball_params2;

  // these count pair arrays are not going to be used
  intra_dat.r1.path_dist = 0;
  intra_dat.r2.path_dist = 0;
  intra_dat.pair_data.conn_seps = 0;

  // Final data members
  intra_dat.pair_data.global_params = global_params[0];
  intra_dat.pair_data.total_lk_ball_iso = 0;
  intra_dat.pair_data.total_lk_ball = 0;
  intra_dat.pair_data.total_lk_bridge = 0;
  intra_dat.pair_data.total_lk_bridge_uncpl = 0;
}

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    int TILE_SIZE,
    int MAX_N_WATER,
    int MAX_N_CONN,
    typename Real,
    typename Int>
void TMOL_DEVICE_FUNC lk_ball_load_intrares1_tile_data_to_shared(
    TView<Vec<Real, 3>, 2, Dev> pose_coords,
    TView<Vec<Real, 3>, 3, Dev> water_coords,
    TView<Int, 2, Dev> block_type_tile_n_polar_atoms,
    TView<Int, 2, Dev> block_type_tile_n_occluder_atoms,
    TView<Int, 3, Dev> block_type_tile_pol_occ_inds,
    TView<LKBallTypeParams<Real>, 3, Dev> block_type_tile_lk_ball_params,
    int tile_ind,
    int start_atom1,
    int n_atoms_to_load1,
    LKBallScoringData<Real> &intra_dat,
    LKBallBlockPairSharedData<Real, TILE_SIZE, MAX_N_WATER, MAX_N_CONN>
        &shared_m) {
  auto store_n_pol_n_occ1 = ([&](int tid) {
    if (tid == 0) {
      int n_pol =
          block_type_tile_n_polar_atoms[intra_dat.r1.block_type][tile_ind];
      int n_occ =
          block_type_tile_n_occluder_atoms[intra_dat.r1.block_type][tile_ind];
      shared_m.n_polars1 = n_pol;
      shared_m.n_occluders1 = n_occ;
    }
  });
  DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(store_n_pol_n_occ1);
  DeviceDispatch<Dev>::synchronize_workgroup();

  intra_dat.r1.n_polars = shared_m.n_polars1;
  intra_dat.r1.n_occluders = shared_m.n_occluders1;

  lk_ball_load_block_coords_and_params_into_shared<
      DeviceDispatch,
      Dev,
      nt,
      MAX_N_WATER>(
      pose_coords,
      water_coords,
      block_type_tile_pol_occ_inds,
      block_type_tile_lk_ball_params,
      intra_dat.pair_data.pose_ind,
      tile_ind,
      intra_dat.r1,
      n_atoms_to_load1,
      start_atom1);
}

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    int TILE_SIZE,
    int MAX_N_WATER,
    int MAX_N_CONN,
    typename Real,
    typename Int>
void TMOL_DEVICE_FUNC lk_ball_load_intrares2_tile_data_to_shared(
    TView<Vec<Real, 3>, 2, Dev> pose_coords,
    TView<Vec<Real, 3>, 3, Dev> water_coords,
    TView<Int, 2, Dev> block_type_tile_n_polar_atoms,
    TView<Int, 2, Dev> block_type_tile_n_occluder_atoms,
    TView<Int, 3, Dev> block_type_tile_pol_occ_inds,
    TView<LKBallTypeParams<Real>, 3, Dev> block_type_tile_lk_ball_params,
    int tile_ind,
    int start_atom2,
    int n_atoms_to_load2,
    LKBallScoringData<Real> &intra_dat,
    LKBallBlockPairSharedData<Real, TILE_SIZE, MAX_N_WATER, MAX_N_CONN>
        &shared_m) {
  auto store_n_pol_n_occ2 = ([&](int tid) {
    if (tid == 0) {
      int n_pol =
          block_type_tile_n_polar_atoms[intra_dat.r2.block_type][tile_ind];
      int n_occ =
          block_type_tile_n_occluder_atoms[intra_dat.r2.block_type][tile_ind];
      shared_m.n_polars2 = n_pol;
      shared_m.n_occluders2 = n_occ;
    }
  });
  DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(store_n_pol_n_occ2);
  DeviceDispatch<Dev>::synchronize_workgroup();

  intra_dat.r2.n_polars = shared_m.n_polars2;
  intra_dat.r2.n_occluders = shared_m.n_occluders2;

  lk_ball_load_block_coords_and_params_into_shared<
      DeviceDispatch,
      Dev,
      nt,
      MAX_N_WATER>(
      pose_coords,
      water_coords,
      block_type_tile_pol_occ_inds,
      block_type_tile_lk_ball_params,
      intra_dat.pair_data.pose_ind,
      tile_ind,
      intra_dat.r2,
      n_atoms_to_load2,
      start_atom2);
}

template <int TILE_SIZE, int MAX_N_WATER, int MAX_N_CONN, typename Real>
void TMOL_DEVICE_FUNC lk_ball_load_intrares_data_from_shared(
    int tile_ind1,
    int tile_ind2,
    LKBallBlockPairSharedData<Real, TILE_SIZE, MAX_N_WATER, MAX_N_CONN>
        &shared_m,
    LKBallScoringData<Real> &intra_dat) {
  // set the pointers in intra_dat to point at the shared-memory arrays
  // If we are evaluating the energies between atoms in the same tile
  // then only the "1" shared-memory arrays will be loaded with data;
  // we will point the "2" memory pointers at the "1" arrays
  bool same_tile = tile_ind1 == tile_ind2;
  intra_dat.r1.pose_coords = shared_m.pose_coords1;
  intra_dat.r2.pose_coords =
      (same_tile ? shared_m.pose_coords1 : shared_m.pose_coords2);
  intra_dat.r1.water_coords = shared_m.water_coords1;
  intra_dat.r2.water_coords =
      (same_tile ? shared_m.water_coords1 : shared_m.water_coords2);
  intra_dat.r1.pol_occ_tile_inds = shared_m.pol_occ_tile_inds1;
  intra_dat.r2.pol_occ_tile_inds =
      (same_tile ? shared_m.pol_occ_tile_inds1 : shared_m.pol_occ_tile_inds2);
  intra_dat.r1.lk_ball_params = shared_m.lk_ball_params1;
  intra_dat.r2.lk_ball_params =
      (same_tile ? shared_m.lk_ball_params1 : shared_m.lk_ball_params2);
  if (same_tile) {
    intra_dat.r2.n_polars = intra_dat.r1.n_polars;
    intra_dat.r2.n_occluders = intra_dat.r1.n_occluders;
  }
}

template <int MAX_N_WATER, typename Real>
TMOL_DEVICE_FUNC lk_ball_Vt<Real> lk_ball_atom_energy_full(
    int polar_ind,               // in [0:n_polar)
    int occluder_ind,            // in [0:n_occluders)
    int polar_atom_tile_ind,     // in [0:TILE_SIZE)
    int occluder_atom_tile_ind,  // in [0:TILE_SIZE)
    int polar_start,
    int occluder_start,
    LKBallSingleResData<Real> const &polar_block_dat,
    LKBallSingleResData<Real> const &occluder_block_dat,
    LKBallResPairData<Real> const &block_pair_dat,
    int cp_separation) {
  using tmol::score::common::coord_from_shared;
  using tmol::score::common::distance;
  using Real3 = Eigen::Matrix<Real, 3, 1>;

  if (cp_separation <= 3) {
    return {0, 0, 0, 0};
  }

  Real3 polar_xyz =
      coord_from_shared(polar_block_dat.pose_coords, polar_atom_tile_ind);
  Real3 occluder_xyz =
      coord_from_shared(occluder_block_dat.pose_coords, occluder_atom_tile_ind);

  Real const dist = distance<Real>::V(polar_xyz, occluder_xyz);

  if (dist >= block_pair_dat.global_params.distance_threshold) {
    return {0, 0, 0, 0};
    ;
  }

  Eigen::Matrix<Real, 4, 3> wmat_polar;
  Eigen::Matrix<Real, 4, 3> wmat_occluder;

  for (int wi = 0; wi < MAX_N_WATER; wi++) {
    wmat_polar.row(wi) = coord_from_shared(
        polar_block_dat.water_coords, MAX_N_WATER * polar_atom_tile_ind + wi);
    wmat_occluder.row(wi) = coord_from_shared(
        occluder_block_dat.water_coords,
        MAX_N_WATER * occluder_atom_tile_ind + wi);
  }

  // DEBUG: load dummy coords into shared memory
  // for (int wi = 0; wi < MAX_N_WATER; wi++) {
  //   if (wi < 2) {
  //     wmat_polar.row(wi) =
  //         coord_from_shared(polar_block_dat.pose_coords, polar_atom_tile_ind)
  //         + Real3{1, 1, 1};
  //     wmat_occluder.row(wi) =
  //         coord_from_shared(occluder_block_dat.pose_coords,
  //         polar_atom_tile_ind)
  //         + Real3{1, 1, 1};
  //   } else {
  //     wmat_polar.row(wi) = Real3{NAN, NAN, NAN};
  //     wmat_occluder.row(wi) = Real3{NAN, NAN, NAN};
  //   }
  // }

  return lk_ball_score<Real, MAX_N_WATER>::V(
      polar_xyz,
      occluder_xyz,
      wmat_polar,
      wmat_occluder,
      cp_separation,
      polar_block_dat.lk_ball_params[polar_ind],
      occluder_block_dat.lk_ball_params[occluder_ind],
      block_pair_dat.global_params);
}

// Calculate and write to global memory only the derivatives for the two
// indicated atoms Does not return  the score
template <int TILE_SIZE, int MAX_N_WATER, typename Real, tmol::Device Dev>
TMOL_DEVICE_FUNC void lk_ball_atom_derivs_full(
    int polar_ind,               // in [0:n_polar)
    int occluder_ind,            // in [0:n_occluders)
    int polar_atom_tile_ind,     // in [0:TILE_SIZE)
    int occluder_atom_tile_ind,  // in [0:TILE_SIZE)
    int polar_start,
    int occluder_start,
    LKBallSingleResData<Real> const &polar_block_dat,
    LKBallSingleResData<Real> const &occluder_block_dat,
    LKBallResPairData<Real> const &block_pair_dat,
    int cp_separation,
    TView<Real, 2, Dev> dTdV,
    TView<Eigen::Matrix<Real, 3, 1>, 2, Dev> dV_d_pose_coords,
    TView<Eigen::Matrix<Real, 3, 1>, 3, Dev> dV_d_water_coords) {
  using WatersMat = Eigen::Matrix<Real, MAX_N_WATER, 3>;
  using Real3 = Eigen::Matrix<Real, 3, 1>;
  using tmol::score::common::accumulate;
  using tmol::score::common::coord_from_shared;
  using tmol::score::common::distance;

  if (cp_separation <= 3) {
    return;
  }

  Real3 polar_xyz =
      coord_from_shared(polar_block_dat.pose_coords, polar_atom_tile_ind);
  Real3 occluder_xyz =
      coord_from_shared(occluder_block_dat.pose_coords, occluder_atom_tile_ind);

  auto const dist_r = distance<Real>::V_dV(polar_xyz, occluder_xyz);
  if (dist_r.V >= block_pair_dat.global_params.distance_threshold) return;

  Eigen::Matrix<Real, MAX_N_WATER, 3> wmat_polar;
  Eigen::Matrix<Real, MAX_N_WATER, 3> wmat_occluder;
  Eigen::Matrix<Real, n_lk_ball_score_types, 1> dTdV_local;

  for (int i = 0; i < n_lk_ball_score_types; ++i) {
    dTdV_local[i] = dTdV[block_pair_dat.pose_ind][i];
  }
  for (int wi = 0; wi < MAX_N_WATER; wi++) {
    wmat_polar.row(wi) = coord_from_shared(
        polar_block_dat.water_coords, MAX_N_WATER * polar_atom_tile_ind + wi);
    wmat_occluder.row(wi) = coord_from_shared(
        occluder_block_dat.water_coords,
        MAX_N_WATER * occluder_atom_tile_ind + wi);
  }

  // DEBUG: load dummy coords into shared memory
  // for (int wi = 0; wi < MAX_N_WATER; wi++) {
  //   if (wi < 2) {
  //     wmat_polar.row(wi) =
  //         coord_from_shared(polar_block_dat.pose_coords, polar_atom_tile_ind)
  //         + Real3{1, 1, 1};
  //     wmat_occluder.row(wi) =
  //         coord_from_shared(occluder_block_dat.pose_coords,
  //         polar_atom_tile_ind)
  //         + Real3{1, 1, 1};
  //   } else {
  //     wmat_polar.row(wi) = Real3{NAN, NAN, NAN};
  //     wmat_occluder.row(wi) = Real3{NAN, NAN, NAN};
  //   }
  // }

  auto dV = lk_ball_score<Real, 4>::dV(
      polar_xyz,
      occluder_xyz,
      wmat_polar,
      wmat_occluder,
      cp_separation,
      polar_block_dat.lk_ball_params[polar_ind],
      occluder_block_dat.lk_ball_params[occluder_ind],
      block_pair_dat.global_params);

  auto accum_pose_derivs = ([&] TMOL_DEVICE_FUNC(
                                LKBallSingleResData<Real> const &block_dat,
                                int atom_ind,
                                lk_ball_dV_dReal3<Real> dV) {
    for (int j = 0; j < 3; ++j) {
      Real dV_dj = dTdV_local[w_lk_ball_iso] * dV.d_lkball_iso[j]
                   + dTdV_local[w_lk_ball] * dV.d_lkball[j]
                   + dTdV_local[w_lk_bridge] * dV.d_lkbridge[j]
                   + dTdV_local[w_lk_bridge_uncpl] * dV.d_lkbridge_uncpl[j];

      if (dV_dj != 0) {
        accumulate<Dev, Real>::add(
            dV_d_pose_coords[block_pair_dat.pose_ind]
                            [block_dat.block_coord_offset + atom_ind][j],
            dV_dj);
      }
    }
  });

  auto accum_water_derivs = ([&] TMOL_DEVICE_FUNC(
                                 LKBallSingleResData<Real> const &block_dat,
                                 int atom_ind,
                                 int water_ind,
                                 lk_ball_dV_dWater<Real, MAX_N_WATER> dV) {
    for (int j = 0; j < 3; ++j) {
      Real dV_dj =
          dTdV_local[w_lk_ball_iso] * dV.d_lkball_iso(water_ind, j)
          + dTdV_local[w_lk_ball] * dV.d_lkball(water_ind, j)
          + dTdV_local[w_lk_bridge] * dV.d_lkbridge(water_ind, j)
          + dTdV_local[w_lk_bridge_uncpl] * dV.d_lkbridge_uncpl(water_ind, j);
      if (dV_dj != 0) {
        accumulate<Dev, Real>::add(
            dV_d_water_coords[block_pair_dat.pose_ind]
                             [block_dat.block_coord_offset + atom_ind]
                             [water_ind][j],
            dV_dj);
      }
    }
  });

  accum_pose_derivs(polar_block_dat, polar_start + polar_atom_tile_ind, dV.dI);

  accum_pose_derivs(
      occluder_block_dat, occluder_start + occluder_atom_tile_ind, dV.dJ);

  for (int i = 0; i < MAX_N_WATER; ++i) {
    accum_water_derivs(
        polar_block_dat, polar_start + polar_atom_tile_ind, i, dV.dWI);
    accum_water_derivs(
        occluder_block_dat, occluder_start + occluder_atom_tile_ind, i, dV.dWJ);
  }
}

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    typename Func,
    typename Real>
void TMOL_DEVICE_FUNC eval_interres_pol_occ_pair_energies(
    LKBallScoringData<Real> &inter_dat,
    int start_atom1,
    int start_atom2,
    Func f) {
  auto eval_scores_for_pol_occ_pairs = ([&](int tid) {
    int const n_pol_occ_pairs =
        inter_dat.r1.n_polars * inter_dat.r2.n_occluders
        + inter_dat.r1.n_occluders * inter_dat.r2.n_polars;
    for (int i = tid; i < n_pol_occ_pairs; i += nt) {
      bool r1_polar = i < inter_dat.r1.n_polars * inter_dat.r2.n_occluders;
      int pair_ind =
          r1_polar ? i : i - inter_dat.r1.n_polars * inter_dat.r2.n_occluders;
      LKBallSingleResData<Real> const &pol_dat =
          r1_polar ? inter_dat.r1 : inter_dat.r2;
      LKBallSingleResData<Real> const &occ_dat =
          r1_polar ? inter_dat.r2 : inter_dat.r1;
      int pol_ind = pair_ind / occ_dat.n_occluders;
      int occ_ind = pair_ind % occ_dat.n_occluders;
      int pol_start = r1_polar ? start_atom1 : start_atom2;
      int occ_start = r1_polar ? start_atom2 : start_atom1;

      // Do the work!
      f(pol_start, occ_start, pol_ind, occ_ind, inter_dat, r1_polar);
    }
  });
  DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(
      eval_scores_for_pol_occ_pairs);
}

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    typename Func,
    typename Real>
void TMOL_DEVICE_FUNC eval_intrares_pol_occ_pair_energies(
    LKBallScoringData<Real> &intra_dat,
    int start_atom1,
    int start_atom2,
    Func f) {
  auto eval_scores_for_pol_occ_pairs = ([&](int tid) {
    if (start_atom1 == start_atom2) {
      int const n_pol_occ_pairs =
          intra_dat.r1.n_polars * intra_dat.r1.n_occluders;
      for (int i = tid; i < n_pol_occ_pairs; i += nt) {
        int pol_ind = i / intra_dat.r1.n_occluders;
        int occ_ind = i % intra_dat.r1.n_occluders;

        // An atom canot occlude itself
        if (pol_ind == occ_ind) continue;

        // Do the work!
        f(start_atom1, start_atom1, pol_ind, occ_ind, intra_dat, true);
      }
    } else {
      int const n_pol_occ_pairs =
          intra_dat.r1.n_polars * intra_dat.r2.n_occluders
          + intra_dat.r1.n_occluders * intra_dat.r2.n_polars;
      for (int i = tid; i < n_pol_occ_pairs; i += nt) {
        bool r1_polar = i < intra_dat.r1.n_polars * intra_dat.r2.n_occluders;
        int pair_ind =
            r1_polar ? i : i - intra_dat.r1.n_polars * intra_dat.r2.n_occluders;
        LKBallSingleResData<Real> const &occ_dat =
            r1_polar ? intra_dat.r2 : intra_dat.r1;
        int pol_ind = pair_ind / occ_dat.n_occluders;
        int occ_ind = pair_ind % occ_dat.n_occluders;
        int pol_start = r1_polar ? start_atom1 : start_atom2;
        int occ_start = r1_polar ? start_atom2 : start_atom1;

        // Do the work!
        f(pol_start, occ_start, pol_ind, occ_ind, intra_dat, r1_polar);
      }
    }
  });

  DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(
      eval_scores_for_pol_occ_pairs);
}

///////////////////////////////////////////
///// BEGIN MASSIVE DUPLICATION ///////////
///////////////////////////////////////////

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    int MAX_N_WATER,
    typename Real,
    typename Int>
void TMOL_DEVICE_FUNC lk_ball_load_block_coords_and_params_into_shared2(
    TView<Vec<Real, 3>, 2, Dev> pose_coords,
    TView<Vec<Real, 3>, 3, Dev> water_coords,
    TView<Int, 3, Dev> block_type_tile_pol_occ_inds,
    TView<Int, 3, Dev> block_type_tile_pol_occ_n_waters,
    TView<LKBallTypeParams<Real>, 3, Dev> block_type_tile_lk_ball_params,
    int pose_ind,
    int tile_ind,
    LKBallSingleResData2<Real> &r_dat,
    int n_atoms_to_load,
    int start_atom) {
  // pre-condition: n_atoms_to_load < TILE_SIZE
  // Note that TILE_SIZE is not explicitly passed in, but is "present"
  // in r_dat.coords allocation

  DeviceDispatch<Dev>::template copy_contiguous_data<nt, 3>(
      r_dat.pose_coords,
      reinterpret_cast<Real *>(
          &pose_coords[pose_ind][r_dat.block_coord_offset + start_atom]),
      n_atoms_to_load * 3);
  DeviceDispatch<Dev>::template copy_contiguous_data<nt, MAX_N_WATER * 3>(
      r_dat.water_coords,
      reinterpret_cast<Real *>(
          &water_coords[pose_ind][r_dat.block_coord_offset + start_atom][0]),
      n_atoms_to_load * MAX_N_WATER * 3);
  DeviceDispatch<Dev>::template copy_contiguous_data_and_cast<nt, 1>(
      r_dat.pol_occ_tile_inds,
      &block_type_tile_pol_occ_inds[r_dat.block_type][tile_ind][0],
      r_dat.n_occluders);
  DeviceDispatch<Dev>::template copy_contiguous_data_and_cast<nt, 1>(
      r_dat.n_waters,
      &block_type_tile_pol_occ_n_waters[r_dat.block_type][tile_ind][0],
      r_dat.n_occluders);
  int const N_PARAMS = sizeof(LKBallTypeParams<Real>) / sizeof(Real);
  DeviceDispatch<Dev>::template copy_contiguous_data<nt, N_PARAMS>(
      reinterpret_cast<Real *>(r_dat.lk_ball_params),
      reinterpret_cast<Real *>(
          &block_type_tile_lk_ball_params[r_dat.block_type][tile_ind][0]),
      r_dat.n_occluders * N_PARAMS);
}

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    int TILE_SIZE,
    int MAX_N_WATER,
    typename Real,
    typename Int>
void TMOL_DEVICE_FUNC lk_ball_load_block_into_shared2(
    TView<Vec<Real, 3>, 2, Dev> pose_coords,
    TView<Vec<Real, 3>, 3, Dev> water_coords,
    TView<Int, 3, Dev> block_type_tile_pol_occ_inds,
    TView<Int, 3, Dev> block_type_tile_pol_occ_n_waters,
    TView<LKBallTypeParams<Real>, 3, Dev> block_type_tile_lk_ball_params,
    TView<Int, 3, Dev> block_type_path_distance,
    int pose_ind,
    int tile_ind,
    LKBallSingleResData2<Real> &r_dat,
    int n_atoms_to_load,
    int start_atom,
    bool count_pair_striking_dist,
    unsigned char *__restrict__ conn_ats) {
  lk_ball_load_block_coords_and_params_into_shared2<
      DeviceDispatch,
      Dev,
      nt,
      MAX_N_WATER>(
      pose_coords,
      water_coords,
      block_type_tile_pol_occ_inds,
      block_type_tile_pol_occ_n_waters,
      block_type_tile_lk_ball_params,
      pose_ind,
      tile_ind,
      r_dat,
      n_atoms_to_load,
      start_atom);

  auto copy_path_dists = ([=](int tid) {
    for (int count = tid; count < n_atoms_to_load; count += nt) {
      int const atid = start_atom + count;
      for (int j = 0; j < r_dat.n_conn; ++j) {
        unsigned char ij_path_dist =
            block_type_path_distance[r_dat.block_type][conn_ats[j]][atid];
        r_dat.path_dist[j * TILE_SIZE + count] = ij_path_dist;
      }
    }
  });
  if (count_pair_striking_dist) {
    DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(copy_path_dists);
  }
}

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    typename Int,
    typename Real,
    int TILE_SIZE,
    int MAX_N_WATER,
    int MAX_N_CONN>
void TMOL_DEVICE_FUNC lk_ball_load_tile_invariant_interres_data2(
    TView<Int, 2, Dev> pose_stack_block_coord_offset,
    TView<Int, 2, Dev> pose_stack_block_type,
    TView<Vec<Int, 2>, 3, Dev> pose_stack_inter_residue_connections,
    TView<Int, 3, Dev> pose_stack_min_bond_separation,
    TView<Int, 5, Dev> pose_stack_inter_block_bondsep,
    TView<Int, 1, Dev> block_type_n_interblock_bonds,
    TView<Int, 2, Dev> block_type_atoms_forming_chemical_bonds,
    TView<LKBallGlobalParams<Real>, 1, Dev> global_params,
    int const max_important_bond_separation,
    int pose_ind,
    int block_ind1,
    int block_ind2,
    int block_type1,
    int block_type2,
    int n_atoms1,
    int n_atoms2,
    LKBallScoringData2<Real> &inter_dat,
    LKBallBlockPairSharedData2<Real, TILE_SIZE, MAX_N_WATER, MAX_N_CONN>
        &shared_m) {
  inter_dat.pair_data.pose_ind = pose_ind;
  inter_dat.r1.block_ind = block_ind1;
  inter_dat.r2.block_ind = block_ind2;
  inter_dat.r1.block_type = block_type1;
  inter_dat.r2.block_type = block_type2;
  inter_dat.r1.block_coord_offset =
      pose_stack_block_coord_offset[pose_ind][block_ind1];
  inter_dat.r2.block_coord_offset =
      pose_stack_block_coord_offset[pose_ind][block_ind2];
  inter_dat.pair_data.max_important_bond_separation =
      max_important_bond_separation;
  inter_dat.pair_data.min_separation =
      pose_stack_min_bond_separation[pose_ind][block_ind1][block_ind2];
  inter_dat.pair_data.in_count_pair_striking_dist =
      inter_dat.pair_data.min_separation <= max_important_bond_separation;
  inter_dat.r1.n_atoms = n_atoms1;
  inter_dat.r2.n_atoms = n_atoms2;
  inter_dat.r1.n_conn = block_type_n_interblock_bonds[block_type1];
  inter_dat.r2.n_conn = block_type_n_interblock_bonds[block_type2];

  // set the pointers in inter_dat to point at the shared-memory arrays
  inter_dat.r1.pose_coords = shared_m.pose_coords1;
  inter_dat.r2.pose_coords = shared_m.pose_coords2;
  inter_dat.r1.water_coords = shared_m.water_coords1;
  inter_dat.r2.water_coords = shared_m.water_coords2;
  inter_dat.r1.pol_occ_tile_inds = shared_m.pol_occ_tile_inds1;
  inter_dat.r2.pol_occ_tile_inds = shared_m.pol_occ_tile_inds2;
  inter_dat.r1.n_waters = shared_m.n_waters1;
  inter_dat.r2.n_waters = shared_m.n_waters2;
  inter_dat.r1.lk_ball_params = shared_m.lk_ball_params1;
  inter_dat.r2.lk_ball_params = shared_m.lk_ball_params2;

  inter_dat.r1.path_dist = shared_m.path_dist1;
  inter_dat.r2.path_dist = shared_m.path_dist2;
  inter_dat.pair_data.conn_seps = shared_m.conn_seps;

  // Count pair setup that does not depend on which tile we are
  // operating on; only necessary if r1 and r2 are within
  // a minimum number of chemical bonds separation
  if (inter_dat.pair_data.in_count_pair_striking_dist) {
    // Load data into shared arrays
    auto load_count_pair_conn_at_data = ([&](int tid) {
      int n_conn_tot = inter_dat.r1.n_conn + inter_dat.r2.n_conn
                       + inter_dat.r1.n_conn * inter_dat.r2.n_conn;
      for (int count = tid; count < n_conn_tot; count += nt) {
        if (count < inter_dat.r1.n_conn) {
          int const conn_ind = count;
          shared_m.conn_ats1[conn_ind] =
              block_type_atoms_forming_chemical_bonds[block_type1][conn_ind];
        } else if (count < inter_dat.r1.n_conn + inter_dat.r2.n_conn) {
          int const conn_ind = count - inter_dat.r1.n_conn;
          shared_m.conn_ats2[conn_ind] =
              block_type_atoms_forming_chemical_bonds[block_type2][conn_ind];
        } else {
          int const conn_ind =
              count - inter_dat.r1.n_conn - inter_dat.r2.n_conn;
          int conn1 = conn_ind / inter_dat.r2.n_conn;
          int conn2 = conn_ind % inter_dat.r2.n_conn;
          shared_m.conn_seps[conn_ind] =
              pose_stack_inter_block_bondsep[pose_ind][block_ind1][block_ind2]
                                            [conn1][conn2];
        }
      }
    });
    DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(
        load_count_pair_conn_at_data);
  }

  // Final data members
  inter_dat.pair_data.global_params = global_params[0];

  // Set initial energy totals to 0
  inter_dat.pair_data.total_lk_ball_iso = 0;
  inter_dat.pair_data.total_lk_ball = 0;
  inter_dat.pair_data.total_lk_bridge = 0;
  inter_dat.pair_data.total_lk_bridge_uncpl = 0;

  // locations in shared mem where we will keep our temporary results
  inter_dat.pair_data.lk_iso = shared_m.lk_iso;
  inter_dat.pair_data.lk_fraction = shared_m.lk_fraction;
  inter_dat.pair_data.lk_bridge_fraction = shared_m.lk_bridge_fraction;
  inter_dat.pair_data.lk_fraction_exp_d2_delta_sum =
      shared_m.lk_fraction_exp_d2_delta_sum;
  inter_dat.pair_data.lk_bridge_fraction_exp_d2_delta_sum =
      shared_m.lk_bridge_fraction_exp_d2_delta_sum;
  inter_dat.pair_data.n_tasks_per_line_element =
      shared_m.n_tasks_per_line_element;
  inter_dat.pair_data.task_offset_for_line_element =
      shared_m.task_offset_for_line_element;
  inter_dat.pair_data.line_element_for_task = shared_m.line_element_for_task;
  inter_dat.pair_data.rank_for_task = shared_m.rank_for_task;
  inter_dat.pair_data.task_is_first_task_for_line_element =
      shared_m.task_is_first_task_for_line_element;
  inter_dat.pair_data.exp_d2_delta_for_task = shared_m.exp_d2_delta_for_task;
}

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    int TILE_SIZE,
    int MAX_N_WATER,
    int MAX_N_CONN,
    typename Real,
    typename Int>
void TMOL_DEVICE_FUNC lk_ball_load_interres1_tile_data_to_shared2(
    TView<Vec<Real, 3>, 2, Dev> pose_coords,
    TView<Vec<Real, 3>, 3, Dev> water_coords,
    TView<Int, 2, Dev> block_type_tile_n_polar_atoms,
    TView<Int, 2, Dev> block_type_tile_n_occluder_atoms,
    TView<Int, 3, Dev> block_type_tile_pol_occ_inds,
    TView<Int, 3, Dev> block_type_tile_pol_occ_n_waters,
    TView<LKBallTypeParams<Real>, 3, Dev> block_type_tile_lk_ball_params,
    TView<Int, 3, Dev> block_type_path_distance,
    int tile_ind,
    int start_atom1,
    int n_atoms_to_load1,
    LKBallScoringData2<Real> &inter_dat,
    LKBallBlockPairSharedData2<Real, TILE_SIZE, MAX_N_WATER, MAX_N_CONN>
        &shared_m) {
  auto store_n_pol_n_occ1 = ([&](int tid) {
    int n_pol =
        block_type_tile_n_polar_atoms[inter_dat.r1.block_type][tile_ind];
    int n_occ =
        block_type_tile_n_occluder_atoms[inter_dat.r1.block_type][tile_ind];
    inter_dat.r1.n_polars = n_pol;
    inter_dat.r1.n_occluders = n_occ;
    if (tid == 0) {
      shared_m.n_polars1 = n_pol;
      shared_m.n_occluders1 = n_occ;
    }
  });
  DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(store_n_pol_n_occ1);

  lk_ball_load_block_into_shared2<
      DeviceDispatch,
      Dev,
      nt,
      TILE_SIZE,
      MAX_N_WATER>(
      pose_coords,
      water_coords,
      block_type_tile_pol_occ_inds,
      block_type_tile_pol_occ_n_waters,
      block_type_tile_lk_ball_params,
      block_type_path_distance,
      inter_dat.pair_data.pose_ind,
      tile_ind,
      inter_dat.r1,
      n_atoms_to_load1,
      start_atom1,
      inter_dat.pair_data.in_count_pair_striking_dist,
      shared_m.conn_ats1);
}

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    int TILE_SIZE,
    int MAX_N_WATER,
    int MAX_N_CONN,
    typename Real,
    typename Int>
void TMOL_DEVICE_FUNC lk_ball_load_interres2_tile_data_to_shared2(
    TView<Vec<Real, 3>, 2, Dev> pose_coords,
    TView<Vec<Real, 3>, 3, Dev> water_coords,
    TView<Int, 2, Dev> block_type_tile_n_polar_atoms,
    TView<Int, 2, Dev> block_type_tile_n_occluder_atoms,
    TView<Int, 3, Dev> block_type_tile_pol_occ_inds,
    TView<Int, 3, Dev> block_type_tile_pol_occ_n_waters,
    TView<LKBallTypeParams<Real>, 3, Dev> block_type_tile_lk_ball_params,
    TView<Int, 3, Dev> block_type_path_distance,
    int tile_ind,
    int start_atom2,
    int n_atoms_to_load2,
    LKBallScoringData2<Real> &inter_dat,
    LKBallBlockPairSharedData2<Real, TILE_SIZE, MAX_N_WATER, MAX_N_CONN>
        &shared_m) {
  auto store_n_pol_n_occ2 = ([&](int tid) {
    int n_pol =
        block_type_tile_n_polar_atoms[inter_dat.r2.block_type][tile_ind];
    int n_occ =
        block_type_tile_n_occluder_atoms[inter_dat.r2.block_type][tile_ind];
    inter_dat.r2.n_polars = n_pol;
    inter_dat.r2.n_occluders = n_occ;
    if (tid == 0) {
      shared_m.n_polars2 = n_pol;
      shared_m.n_occluders2 = n_occ;
    }
  });
  DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(store_n_pol_n_occ2);

  lk_ball_load_block_into_shared2<
      DeviceDispatch,
      Dev,
      nt,
      TILE_SIZE,
      MAX_N_WATER>(
      pose_coords,
      water_coords,
      block_type_tile_pol_occ_inds,
      block_type_tile_pol_occ_n_waters,
      block_type_tile_lk_ball_params,
      block_type_path_distance,
      inter_dat.pair_data.pose_ind,
      tile_ind,
      inter_dat.r2,
      n_atoms_to_load2,
      start_atom2,
      inter_dat.pair_data.in_count_pair_striking_dist,
      shared_m.conn_ats2);
}

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    typename Int,
    typename Real,
    int TILE_SIZE,
    int MAX_N_WATER,
    int MAX_N_CONN>
void TMOL_DEVICE_FUNC lk_ball_load_tile_invariant_intrares_data2(
    TView<Int, 2, Dev> pose_stack_block_coord_offset,
    TView<Int, 2, Dev> pose_stack_block_type,
    TView<LKBallGlobalParams<Real>, 1, Dev> global_params,
    int const max_important_bond_separation,
    int pose_ind,
    int block_ind1,
    int block_type1,
    int n_atoms1,
    LKBallScoringData2<Real> &intra_dat,
    LKBallBlockPairSharedData2<Real, TILE_SIZE, MAX_N_WATER, MAX_N_CONN>
        &shared_m) {
  intra_dat.pair_data.pose_ind = pose_ind;
  intra_dat.r1.block_ind = block_ind1;
  intra_dat.r2.block_ind = block_ind1;
  intra_dat.r1.block_type = block_type1;
  intra_dat.r2.block_type = block_type1;
  intra_dat.r1.block_coord_offset =
      pose_stack_block_coord_offset[pose_ind][block_ind1];
  intra_dat.r2.block_coord_offset = intra_dat.r1.block_coord_offset;
  intra_dat.pair_data.max_important_bond_separation =
      max_important_bond_separation;

  // we are not going to load count pair data into shared memory because
  // we are not going to use that data from shared memory; setting
  // in_count_pair_striking_distance to false prevents downstream loading
  // of count-pair data, which waste memory bandwidth
  intra_dat.pair_data.min_separation = 0;
  intra_dat.pair_data.in_count_pair_striking_dist = false;

  intra_dat.r1.n_atoms = n_atoms1;
  intra_dat.r2.n_atoms = n_atoms1;
  intra_dat.r1.n_conn = 0;
  intra_dat.r2.n_conn = 0;

  // set the pointers in intra_dat to point at the
  // shared-memory arrays. Note that these arrays will be reset
  // later because which shared memory arrays we will use depends on
  // which tile pair we are evaluating!
  intra_dat.r1.pose_coords = shared_m.pose_coords1;
  intra_dat.r2.pose_coords = shared_m.pose_coords2;
  intra_dat.r1.water_coords = shared_m.water_coords1;
  intra_dat.r2.water_coords = shared_m.water_coords2;
  intra_dat.r1.pol_occ_tile_inds = shared_m.pol_occ_tile_inds1;
  intra_dat.r2.pol_occ_tile_inds = shared_m.pol_occ_tile_inds2;
  intra_dat.r1.n_waters = shared_m.n_waters1;
  intra_dat.r2.n_waters = shared_m.n_waters2;
  intra_dat.r1.lk_ball_params = shared_m.lk_ball_params1;
  intra_dat.r2.lk_ball_params = shared_m.lk_ball_params2;

  // these count pair arrays are not going to be used
  intra_dat.r1.path_dist = 0;
  intra_dat.r2.path_dist = 0;
  intra_dat.pair_data.conn_seps = 0;

  // Final data members
  intra_dat.pair_data.global_params = global_params[0];
  intra_dat.pair_data.total_lk_ball_iso = 0;
  intra_dat.pair_data.total_lk_ball = 0;
  intra_dat.pair_data.total_lk_bridge = 0;
  intra_dat.pair_data.total_lk_bridge_uncpl = 0;
}

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    int TILE_SIZE,
    int MAX_N_WATER,
    int MAX_N_CONN,
    typename Real,
    typename Int>
void TMOL_DEVICE_FUNC lk_ball_load_intrares1_tile_data_to_shared2(
    TView<Vec<Real, 3>, 2, Dev> pose_coords,
    TView<Vec<Real, 3>, 3, Dev> water_coords,
    TView<Int, 2, Dev> block_type_tile_n_polar_atoms,
    TView<Int, 2, Dev> block_type_tile_n_occluder_atoms,
    TView<Int, 3, Dev> block_type_tile_pol_occ_inds,
    TView<Int, 3, Dev> block_type_tile_pol_occ_n_waters,
    TView<LKBallTypeParams<Real>, 3, Dev> block_type_tile_lk_ball_params,
    int tile_ind,
    int start_atom1,
    int n_atoms_to_load1,
    LKBallScoringData2<Real> &intra_dat,
    LKBallBlockPairSharedData2<Real, TILE_SIZE, MAX_N_WATER, MAX_N_CONN>
        &shared_m) {
  auto store_n_pol_n_occ1 = ([&](int tid) {
    if (tid == 0) {
      int n_pol =
          block_type_tile_n_polar_atoms[intra_dat.r1.block_type][tile_ind];
      int n_occ =
          block_type_tile_n_occluder_atoms[intra_dat.r1.block_type][tile_ind];
      shared_m.n_polars1 = n_pol;
      shared_m.n_occluders1 = n_occ;
    }
  });
  DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(store_n_pol_n_occ1);
  DeviceDispatch<Dev>::synchronize_workgroup();

  intra_dat.r1.n_polars = shared_m.n_polars1;
  intra_dat.r1.n_occluders = shared_m.n_occluders1;

  lk_ball_load_block_coords_and_params_into_shared2<
      DeviceDispatch,
      Dev,
      nt,
      MAX_N_WATER>(
      pose_coords,
      water_coords,
      block_type_tile_pol_occ_inds,
      block_type_tile_pol_occ_n_waters,
      block_type_tile_lk_ball_params,
      intra_dat.pair_data.pose_ind,
      tile_ind,
      intra_dat.r1,
      n_atoms_to_load1,
      start_atom1);
}

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    int TILE_SIZE,
    int MAX_N_WATER,
    int MAX_N_CONN,
    typename Real,
    typename Int>
void TMOL_DEVICE_FUNC lk_ball_load_intrares2_tile_data_to_shared2(
    TView<Vec<Real, 3>, 2, Dev> pose_coords,
    TView<Vec<Real, 3>, 3, Dev> water_coords,
    TView<Int, 2, Dev> block_type_tile_n_polar_atoms,
    TView<Int, 2, Dev> block_type_tile_n_occluder_atoms,
    TView<Int, 3, Dev> block_type_tile_pol_occ_inds,
    TView<Int, 3, Dev> block_type_tile_pol_occ_n_waters,
    TView<LKBallTypeParams<Real>, 3, Dev> block_type_tile_lk_ball_params,
    int tile_ind,
    int start_atom2,
    int n_atoms_to_load2,
    LKBallScoringData2<Real> &intra_dat,
    LKBallBlockPairSharedData2<Real, TILE_SIZE, MAX_N_WATER, MAX_N_CONN>
        &shared_m) {
  auto store_n_pol_n_occ2 = ([&](int tid) {
    if (tid == 0) {
      int n_pol =
          block_type_tile_n_polar_atoms[intra_dat.r2.block_type][tile_ind];
      int n_occ =
          block_type_tile_n_occluder_atoms[intra_dat.r2.block_type][tile_ind];
      shared_m.n_polars2 = n_pol;
      shared_m.n_occluders2 = n_occ;
    }
  });
  DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(store_n_pol_n_occ2);
  DeviceDispatch<Dev>::synchronize_workgroup();

  intra_dat.r2.n_polars = shared_m.n_polars2;
  intra_dat.r2.n_occluders = shared_m.n_occluders2;

  lk_ball_load_block_coords_and_params_into_shared2<
      DeviceDispatch,
      Dev,
      nt,
      MAX_N_WATER>(
      pose_coords,
      water_coords,
      block_type_tile_pol_occ_inds,
      block_type_tile_pol_occ_n_waters,
      block_type_tile_lk_ball_params,
      intra_dat.pair_data.pose_ind,
      tile_ind,
      intra_dat.r2,
      n_atoms_to_load2,
      start_atom2);
}

template <int TILE_SIZE, int MAX_N_WATER, int MAX_N_CONN, typename Real>
void TMOL_DEVICE_FUNC lk_ball_load_intrares_data_from_shared2(
    int tile_ind1,
    int tile_ind2,
    LKBallBlockPairSharedData2<Real, TILE_SIZE, MAX_N_WATER, MAX_N_CONN>
        &shared_m,
    LKBallScoringData2<Real> &intra_dat) {
  // set the pointers in intra_dat to point at the shared-memory arrays
  // If we are evaluating the energies between atoms in the same tile
  // then only the "1" shared-memory arrays will be loaded with data;
  // we will point the "2" memory pointers at the "1" arrays
  bool same_tile = tile_ind1 == tile_ind2;
  intra_dat.r1.pose_coords = shared_m.pose_coords1;
  intra_dat.r2.pose_coords =
      (same_tile ? shared_m.pose_coords1 : shared_m.pose_coords2);
  intra_dat.r1.water_coords = shared_m.water_coords1;
  intra_dat.r2.water_coords =
      (same_tile ? shared_m.water_coords1 : shared_m.water_coords2);
  intra_dat.r1.pol_occ_tile_inds = shared_m.pol_occ_tile_inds1;
  intra_dat.r2.pol_occ_tile_inds =
      (same_tile ? shared_m.pol_occ_tile_inds1 : shared_m.pol_occ_tile_inds2);
  intra_dat.r1.n_waters = shared_m.n_waters1;
  intra_dat.r2.n_waters = (same_tile ? shared_m.n_waters1 : shared_m.n_waters2);
  intra_dat.r1.lk_ball_params = shared_m.lk_ball_params1;
  intra_dat.r2.lk_ball_params =
      (same_tile ? shared_m.lk_ball_params1 : shared_m.lk_ball_params2);
  if (same_tile) {
    intra_dat.r2.n_polars = intra_dat.r1.n_polars;
    intra_dat.r2.n_occluders = intra_dat.r1.n_occluders;
  }
}

template <int MAX_N_WATER, typename Real>
TMOL_DEVICE_FUNC lk_ball_Vt<Real> lk_ball_atom_energy_full2(
    int polar_ind,               // in [0:n_polar)
    int occluder_ind,            // in [0:n_occluders)
    int polar_atom_tile_ind,     // in [0:TILE_SIZE)
    int occluder_atom_tile_ind,  // in [0:TILE_SIZE)
    int polar_start,
    int occluder_start,
    LKBallSingleResData2<Real> const &polar_block_dat,
    LKBallSingleResData2<Real> const &occluder_block_dat,
    LKBallResPairData2<Real> const &block_pair_dat,
    int cp_separation) {
  using tmol::score::common::coord_from_shared;
  using tmol::score::common::distance;
  using Real3 = Eigen::Matrix<Real, 3, 1>;

  if (cp_separation <= 3) {
    return {0, 0, 0, 0};
  }

  Real3 polar_xyz =
      coord_from_shared(polar_block_dat.pose_coords, polar_atom_tile_ind);
  Real3 occluder_xyz =
      coord_from_shared(occluder_block_dat.pose_coords, occluder_atom_tile_ind);

  Real const dist = distance<Real>::V(polar_xyz, occluder_xyz);

  if (dist >= block_pair_dat.global_params.distance_threshold) {
    return {0, 0, 0, 0};
  }

  Eigen::Matrix<Real, 4, 3> wmat_polar;
  Eigen::Matrix<Real, 4, 3> wmat_occluder;

  for (int wi = 0; wi < MAX_N_WATER; wi++) {
    wmat_polar.row(wi) = coord_from_shared(
        polar_block_dat.water_coords, MAX_N_WATER * polar_atom_tile_ind + wi);
    wmat_occluder.row(wi) = coord_from_shared(
        occluder_block_dat.water_coords,
        MAX_N_WATER * occluder_atom_tile_ind + wi);
  }

  // DEBUG: load dummy coords into shared memory
  // for (int wi = 0; wi < MAX_N_WATER; wi++) {
  //   if (wi < 2) {
  //     wmat_polar.row(wi) =
  //         coord_from_shared(polar_block_dat.pose_coords, polar_atom_tile_ind)
  //         + Real3{1, 1, 1};
  //     wmat_occluder.row(wi) =
  //         coord_from_shared(occluder_block_dat.pose_coords,
  //         polar_atom_tile_ind)
  //         + Real3{1, 1, 1};
  //   } else {
  //     wmat_polar.row(wi) = Real3{NAN, NAN, NAN};
  //     wmat_occluder.row(wi) = Real3{NAN, NAN, NAN};
  //   }
  // }

  return lk_ball_score<Real, MAX_N_WATER>::V(
      polar_xyz,
      occluder_xyz,
      wmat_polar,
      wmat_occluder,
      cp_separation,
      polar_block_dat.lk_ball_params[polar_ind],
      occluder_block_dat.lk_ball_params[occluder_ind],
      block_pair_dat.global_params);
}

// Calculate and write to global memory only the derivatives for the two
// indicated atoms Does not return  the score
template <int TILE_SIZE, int MAX_N_WATER, typename Real, tmol::Device Dev>
TMOL_DEVICE_FUNC void lk_ball_atom_derivs_full2(
    int polar_ind,               // in [0:n_polar)
    int occluder_ind,            // in [0:n_occluders)
    int polar_atom_tile_ind,     // in [0:TILE_SIZE)
    int occluder_atom_tile_ind,  // in [0:TILE_SIZE)
    int polar_start,
    int occluder_start,
    LKBallSingleResData2<Real> const &polar_block_dat,
    LKBallSingleResData2<Real> const &occluder_block_dat,
    LKBallResPairData2<Real> const &block_pair_dat,
    int cp_separation,
    TView<Real, 2, Dev> dTdV,
    TView<Eigen::Matrix<Real, 3, 1>, 2, Dev> dV_d_pose_coords,
    TView<Eigen::Matrix<Real, 3, 1>, 3, Dev> dV_d_water_coords) {
  using WatersMat = Eigen::Matrix<Real, MAX_N_WATER, 3>;
  using Real3 = Eigen::Matrix<Real, 3, 1>;
  using tmol::score::common::accumulate;
  using tmol::score::common::coord_from_shared;
  using tmol::score::common::distance;

  if (cp_separation <= 3) {
    return;
  }

  Real3 polar_xyz =
      coord_from_shared(polar_block_dat.pose_coords, polar_atom_tile_ind);
  Real3 occluder_xyz =
      coord_from_shared(occluder_block_dat.pose_coords, occluder_atom_tile_ind);

  auto const dist_r = distance<Real>::V_dV(polar_xyz, occluder_xyz);
  if (dist_r.V >= block_pair_dat.global_params.distance_threshold) return;

  Eigen::Matrix<Real, MAX_N_WATER, 3> wmat_polar;
  Eigen::Matrix<Real, MAX_N_WATER, 3> wmat_occluder;
  Eigen::Matrix<Real, n_lk_ball_score_types, 1> dTdV_local;

  for (int i = 0; i < n_lk_ball_score_types; ++i) {
    dTdV_local[i] = dTdV[block_pair_dat.pose_ind][i];
  }
  for (int wi = 0; wi < MAX_N_WATER; wi++) {
    wmat_polar.row(wi) = coord_from_shared(
        polar_block_dat.water_coords, MAX_N_WATER * polar_atom_tile_ind + wi);
    wmat_occluder.row(wi) = coord_from_shared(
        occluder_block_dat.water_coords,
        MAX_N_WATER * occluder_atom_tile_ind + wi);
  }

  // DEBUG: load dummy coords into shared memory
  // for (int wi = 0; wi < MAX_N_WATER; wi++) {
  //   if (wi < 2) {
  //     wmat_polar.row(wi) =
  //         coord_from_shared(polar_block_dat.pose_coords, polar_atom_tile_ind)
  //         + Real3{1, 1, 1};
  //     wmat_occluder.row(wi) =
  //         coord_from_shared(occluder_block_dat.pose_coords,
  //         polar_atom_tile_ind)
  //         + Real3{1, 1, 1};
  //   } else {
  //     wmat_polar.row(wi) = Real3{NAN, NAN, NAN};
  //     wmat_occluder.row(wi) = Real3{NAN, NAN, NAN};
  //   }
  // }

  auto dV = lk_ball_score<Real, 4>::dV(
      polar_xyz,
      occluder_xyz,
      wmat_polar,
      wmat_occluder,
      cp_separation,
      polar_block_dat.lk_ball_params[polar_ind],
      occluder_block_dat.lk_ball_params[occluder_ind],
      block_pair_dat.global_params);

  auto accum_pose_derivs = ([&] TMOL_DEVICE_FUNC(
                                LKBallSingleResData2<Real> const &block_dat,
                                int atom_ind,
                                lk_ball_dV_dReal3<Real> dV) {
    for (int j = 0; j < 3; ++j) {
      Real dV_dj = dTdV_local[w_lk_ball_iso] * dV.d_lkball_iso[j]
                   + dTdV_local[w_lk_ball] * dV.d_lkball[j]
                   + dTdV_local[w_lk_bridge] * dV.d_lkbridge[j]
                   + dTdV_local[w_lk_bridge_uncpl] * dV.d_lkbridge_uncpl[j];

      if (dV_dj != 0) {
        accumulate<Dev, Real>::add(
            dV_d_pose_coords[block_pair_dat.pose_ind]
                            [block_dat.block_coord_offset + atom_ind][j],
            dV_dj);
      }
    }
  });

  auto accum_water_derivs = ([&] TMOL_DEVICE_FUNC(
                                 LKBallSingleResData2<Real> const &block_dat,
                                 int atom_ind,
                                 int water_ind,
                                 lk_ball_dV_dWater<Real, MAX_N_WATER> dV) {
    for (int j = 0; j < 3; ++j) {
      Real dV_dj =
          dTdV_local[w_lk_ball_iso] * dV.d_lkball_iso(water_ind, j)
          + dTdV_local[w_lk_ball] * dV.d_lkball(water_ind, j)
          + dTdV_local[w_lk_bridge] * dV.d_lkbridge(water_ind, j)
          + dTdV_local[w_lk_bridge_uncpl] * dV.d_lkbridge_uncpl(water_ind, j);
      if (dV_dj != 0) {
        accumulate<Dev, Real>::add(
            dV_d_water_coords[block_pair_dat.pose_ind]
                             [block_dat.block_coord_offset + atom_ind]
                             [water_ind][j],
            dV_dj);
      }
    }
  });

  accum_pose_derivs(polar_block_dat, polar_start + polar_atom_tile_ind, dV.dI);

  accum_pose_derivs(
      occluder_block_dat, occluder_start + occluder_atom_tile_ind, dV.dJ);

  for (int i = 0; i < MAX_N_WATER; ++i) {
    accum_water_derivs(
        polar_block_dat, polar_start + polar_atom_tile_ind, i, dV.dWI);
    accum_water_derivs(
        occluder_block_dat, occluder_start + occluder_atom_tile_ind, i, dV.dWJ);
  }
}

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    typename Func,
    typename Real>
void TMOL_DEVICE_FUNC eval_interres_pol_occ_pair_energies2(
    LKBallScoringData2<Real> &inter_dat,
    int start_atom1,
    int start_atom2,
    Func f) {
  auto eval_scores_for_pol_occ_pairs = ([&](int tid) {
    int const n_pol_occ_pairs =
        inter_dat.r1.n_polars * inter_dat.r2.n_occluders
        + inter_dat.r1.n_occluders * inter_dat.r2.n_polars;
    for (int i = tid; i < n_pol_occ_pairs; i += nt) {
      bool r1_polar = i < inter_dat.r1.n_polars * inter_dat.r2.n_occluders;
      int pair_ind =
          r1_polar ? i : i - inter_dat.r1.n_polars * inter_dat.r2.n_occluders;
      LKBallSingleResData2<Real> const &pol_dat =
          r1_polar ? inter_dat.r1 : inter_dat.r2;
      LKBallSingleResData2<Real> const &occ_dat =
          r1_polar ? inter_dat.r2 : inter_dat.r1;
      int pol_ind = pair_ind / occ_dat.n_occluders;
      int occ_ind = pair_ind % occ_dat.n_occluders;
      int pol_start = r1_polar ? start_atom1 : start_atom2;
      int occ_start = r1_polar ? start_atom2 : start_atom1;

      // Do the work!
      f(pol_start, occ_start, pol_ind, occ_ind, inter_dat, r1_polar);
    }
  });
  DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(
      eval_scores_for_pol_occ_pairs);
}

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    typename Func,
    typename Real>
void TMOL_DEVICE_FUNC eval_intrares_pol_occ_pair_energies2(
    LKBallScoringData2<Real> &intra_dat,
    int start_atom1,
    int start_atom2,
    Func f) {
  auto eval_scores_for_pol_occ_pairs = ([&](int tid) {
    if (start_atom1 == start_atom2) {
      int const n_pol_occ_pairs =
          intra_dat.r1.n_polars * intra_dat.r1.n_occluders;
      for (int i = tid; i < n_pol_occ_pairs; i += nt) {
        int pol_ind = i / intra_dat.r1.n_occluders;
        int occ_ind = i % intra_dat.r1.n_occluders;

        // An atom canot occlude itself
        if (pol_ind == occ_ind) continue;

        // Do the work!
        f(start_atom1, start_atom1, pol_ind, occ_ind, intra_dat, true);
      }
    } else {
      int const n_pol_occ_pairs =
          intra_dat.r1.n_polars * intra_dat.r2.n_occluders
          + intra_dat.r1.n_occluders * intra_dat.r2.n_polars;
      for (int i = tid; i < n_pol_occ_pairs; i += nt) {
        bool r1_polar = i < intra_dat.r1.n_polars * intra_dat.r2.n_occluders;
        int pair_ind =
            r1_polar ? i : i - intra_dat.r1.n_polars * intra_dat.r2.n_occluders;
        LKBallSingleResData2<Real> const &occ_dat =
            r1_polar ? intra_dat.r2 : intra_dat.r1;
        int pol_ind = pair_ind / occ_dat.n_occluders;
        int occ_ind = pair_ind % occ_dat.n_occluders;
        int pol_start = r1_polar ? start_atom1 : start_atom2;
        int occ_start = r1_polar ? start_atom2 : start_atom1;

        // Do the work!
        f(pol_start, occ_start, pol_ind, occ_ind, intra_dat, r1_polar);
      }
    }
  });

  DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(
      eval_scores_for_pol_occ_pairs);
}

template <int TILE, typename PairData, typename ResData>
TMOL_DEVICE_FUNC int interres_count_pair_separation2(
    PairData const &pair_data,
    ResData const &r1,
    ResData const &r2,
    int atom_tile_ind1,
    int atom_tile_ind2) {
  int separation = pair_data.min_separation;
  if (separation <= pair_data.max_important_bond_separation) {
    separation = common::count_pair::shared_mem_inter_block_separation<TILE>(
        pair_data.max_important_bond_separation,
        atom_tile_ind1,
        atom_tile_ind2,
        r1.n_conn,
        r2.n_conn,
        r1.path_dist,
        r2.path_dist,
        pair_data.conn_seps);
  }
  return separation;
}

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    int TILE_SIZE,
    int MAX_N_WATER,
    typename Real>
TMOL_DEVICE_FUNC void load_balanced_interres_polar_occluder_tile(
    LKBallSingleResData2<Real> &pol_dat,
    LKBallSingleResData2<Real> &occ_dat,
    LKBallResPairData2<Real> &pair_dat,
    bool pol_first) {
  using Real3 = Eigen::Matrix<Real, 3, 1>;
  using tmol::score::common::coord_from_shared;
  using tmol::score::common::distance;
  using tmol::score::ljlk::potentials::lj_sigma;
  using tmol::score::ljlk::potentials::lk_isotropic_pair;

  int n_interacting_pairs = pol_dat.n_polars * occ_dat.n_occluders;

  // We will process groups of TILE_SIZE pol/occs at a time -- call these
  // "rows" (because if the n_pol x n_occ tile were TILE_SIZE x TILE_SIZE,
  // then each group of TILE_SIZE of them would essentially be a row of
  // the tile.
  int const n_rows = (n_interacting_pairs - 1) / TILE_SIZE + 1;
  printf("n_rows %d n_interacting_pairs %d\n", n_rows, n_interacting_pairs);
  for (int count_row = 0; count_row < n_rows; ++count_row) {
    // step 0: initialize shared memory
    auto setup_for_row = ([&] TMOL_DEVICE_FUNC(int tid) {
      DeviceDispatch<Dev>::synchronize_workgroup();
      for (int i = tid; i < TILE_SIZE; i += nt) {
        pair_dat.lk_iso[i] = 0;
        pair_dat.lk_fraction[i] = 0;
        pair_dat.lk_bridge_fraction[i] = 0;
      }
    });
    DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(setup_for_row);

    // step1 evaluate lk_iso for the atoms in the row:
    auto eval_lk_iso_for_row = ([&] TMOL_DEVICE_FUNC(int tid) {
      DeviceDispatch<Dev>::synchronize_workgroup();
      for (int i = tid; i < TILE_SIZE; i += nt) {
        int const pair_ind = TILE_SIZE * count_row + i;
        int const pol_ind = pair_ind / occ_dat.n_occluders;
        int const occ_ind = pair_ind % occ_dat.n_occluders;
        if (pol_ind >= pol_dat.n_polars) {
          pair_dat.lk_iso[i] = 0;
          continue;
        }
        int const pol_tile_ind = pol_dat.pol_occ_tile_inds[pol_ind];
        int const occ_tile_ind = occ_dat.pol_occ_tile_inds[occ_ind];
        Real3 pol_xyz = coord_from_shared(pol_dat.pose_coords, pol_tile_ind);
        Real3 occ_xyz = coord_from_shared(occ_dat.pose_coords, occ_tile_ind);
        Real const dist = distance<Real>::V(pol_xyz, occ_xyz);
        if (dist >= pair_dat.global_params.distance_threshold) {
          pair_dat.lk_iso[i] = 0;
          continue;
        }
        LKBallTypeParams<Real> pol_params = pol_dat.lk_ball_params[pol_ind];
        LKBallTypeParams<Real> occ_params = occ_dat.lk_ball_params[occ_ind];
        Real sigma =
            lj_sigma<Real>(pol_params, occ_params, pair_dat.global_params);

        int const bonded_path_length =
            interres_count_pair_separation2<TILE_SIZE>(
                pair_dat,
                pol_first ? pol_dat : occ_dat,
                pol_first ? occ_dat : pol_dat,
                pol_first ? pol_tile_ind : occ_tile_ind,
                pol_first ? occ_tile_ind : pol_tile_ind);

        Real lk_iso = lk_isotropic_pair<Real>::V(
            dist,
            bonded_path_length,
            sigma,
            pol_params.lj_radius,
            pol_params.lk_dgfree,
            pol_params.lk_lambda,
            occ_params.lk_volume);
        pair_dat.lk_iso[i] = lk_iso;
        printf("%d lk_iso: %9.5f\n", i, lk_iso);
      }
    });

    DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(
        eval_lk_iso_for_row);

    // step2: now we need to compute the water/occluder partition
    // functions for lk-fraction
    auto eval_lk_fraction_d2_delta_for_row = ([&] TMOL_DEVICE_FUNC(int tid) {
      DeviceDispatch<Dev>::synchronize_workgroup();

      // first, we count the number of waters each polar of each polar/occluder
      // pair in this line has; we will use these counts to generate offsets
      for (int i = tid; i < TILE_SIZE; i += nt) {
        int const pair_ind = TILE_SIZE * count_row + i;
        int const pol_ind = pair_ind / occ_dat.n_occluders;
        int const i_n_waters =
            pol_ind < pol_dat.n_polars ? pol_dat.n_waters[pol_ind] : 0;
        pair_dat.n_tasks_per_line_element[i] = i_n_waters;
        pair_dat.task_offset_for_line_element[i] = i_n_waters;

        // Let's take care of some other initialization while we are here
        pair_dat.lk_fraction_exp_d2_delta_sum[i] = 0;
      }

      // Now perform an exclusive scan of the water counts and broadcast the
      // total number of water/occ pairs total to all threads
      int const n_water_occ_pairs_in_row = DeviceDispatch<Dev>::
          template exclusive_scan_in_workgroup<nt, TILE_SIZE>(
              pair_dat.task_offset_for_line_element,
              int(0.),
              mgpu::plus_t<int>(),
              true);

      printf("lk_fraction:\n");
      printf("pair_dat.n_tasks_per_line_element:");
      for (int i = 0; i < TILE_SIZE; ++i) {
        printf(" %d", pair_dat.n_tasks_per_line_element[i]);
      }
      printf("\n");
      printf("task offset:");
      for (int i = 0; i < TILE_SIZE; ++i) {
        printf(" %d", pair_dat.task_offset_for_line_element[i]);
      }
      printf("\n");

      // Ok, knowing how many water/occluder interactions we have, we will
      // process each chunk of TILE_SIZE water/occluder pairs. For each chunk,
      // we will need to figure out exactly which polar/occluder pair the
      // focused water/occluder pair comes from; essentially a "load balance
      // search"
      int const n_wat_occ_iterations =
          (n_water_occ_pairs_in_row - 1) / TILE_SIZE + 1;
      for (int wat_occ_count = 0; wat_occ_count < n_wat_occ_iterations;
           wat_occ_count++) {
        printf(
            "for row %d starting water/occluder iteration %d\n",
            count_row,
            wat_occ_count);
        // Pause to make sure everything in shared memory has been read from
        // before we write any more things
        DeviceDispatch<Dev>::synchronize_workgroup();

        // We need to figure out what set of waters we're going to operate on
        // in this iteration. One way to do this is to examine each
        // polar/occluder pair's offset and water count and if its waters fall
        // in the range between TILE_SIZE * (wat_occ_count) and TILE_SIZE *
        // (wat_occ_count + 1) then we have a pol/occ pair that we will examine
        // in this itration. Start by resetting some shared-memory arrays
        for (int j = tid; j < TILE_SIZE; j += nt) {
          pair_dat.line_element_for_task[j] = -1;
          pair_dat.task_is_first_task_for_line_element[j] = 0;
          pair_dat.rank_for_task[j] = -1;
          pair_dat.exp_d2_delta_for_task[j] = 0;
        }
        DeviceDispatch<Dev>::synchronize_workgroup();

        // For each pol/occ pair in this row, figure out whether they are active
        // in this iteration if so, mark a "this position belongs to me" entry
        // in the line_element_for_task shared-memory array; we will use
        // scan(max) to fill in this array completely in the next step.
        for (int j = tid; j < TILE_SIZE; j += nt) {
          int const j_wat_offset = pair_dat.task_offset_for_line_element[j];
          int const j_n_waters = pair_dat.n_tasks_per_line_element[j];
          if (j_n_waters == 0) {
            continue;
          }
          if (wat_occ_count * TILE_SIZE <= j_wat_offset
              && j_wat_offset < (wat_occ_count + 1) * TILE_SIZE) {
            // ok, then we are operating on some or all of the waters for this
            // pol/occ pair and the first water for this polar is in the range
            int const j_tile_offset = j_wat_offset - wat_occ_count * TILE_SIZE;
            pair_dat.line_element_for_task[j_tile_offset] = j;
            pair_dat.task_is_first_task_for_line_element[j_tile_offset] = 1;
          } else if (
              j_wat_offset < wat_occ_count * TILE_SIZE
              && j_wat_offset + j_n_waters >= wat_occ_count * TILE_SIZE) {
            // ok, then some of the waters for this pose pair are within this
            // tile, but it won't be the first water
            pair_dat.line_element_for_task[0] = j;
            pair_dat.task_is_first_task_for_line_element[0] = 1;
          }
        }

        // With each pol/occ pair that will be examined in this iteration marked
        // in the which_pol_occ_for_wat_occ_pair, use an inclusive scan with
        // maximum to mark for each wat/occ pair which pol/occ pair it is coming
        // from
        DeviceDispatch<Dev>::
            template inclusive_scan_in_workgroup<nt, TILE_SIZE>(
                pair_dat.line_element_for_task,
                int(0),
                mgpu::maximum_t<int32_t>(),
                false);

        printf("line element for task:");
        for (int i = 0; i < TILE_SIZE; ++i) {
          printf(" %d", pair_dat.line_element_for_task[i]);
        }
        printf("\n");
        // Now with the line_element_for_task array initialized, we need to
        // figure out the rank of the task for the pol/occ pair. This rank
        // information will be used again later in the computation, but could
        // likely be computed twice, as it is not expensive to compute
        for (int j = tid; j < TILE_SIZE; j += nt) {
          int const j_pol_occ = pair_dat.line_element_for_task[j];
          int const j_pol_occ_offset =
              pair_dat.task_offset_for_line_element[j_pol_occ];
          pair_dat.rank_for_task[j] =
              (wat_occ_count * TILE_SIZE + j) - j_pol_occ_offset;
        }

        printf("rank for task:");
        for (int i = 0; i < TILE_SIZE; ++i) {
          printf(" %d", pair_dat.rank_for_task[i]);
        }
        printf("\n");
        // OK! now we know exactly which pol/occ pair and which water each
        // wat/occ pair is going to examine. So let's compute the exp_d2_delta
        // for each wat/occ pair and store that into shared memory. We will
        // perform a segmented scan to accumulate the exp_d2_deltas over all
        // water/occ pairs for a given pol/occ pair in the next step.
        for (int j = tid; j < TILE_SIZE; j += nt) {
          int const j_pol_occ = pair_dat.line_element_for_task[j];
          int const pair_ind = TILE_SIZE * count_row + j_pol_occ;
          int const pol_ind = pair_ind / occ_dat.n_occluders;
          if (pol_ind >= pol_dat.n_polars) {
            continue;
          }
          int const occ_ind = pair_ind % occ_dat.n_occluders;
          int const pol_tile_ind = pol_dat.pol_occ_tile_inds[pol_ind];
          int const occ_tile_ind = occ_dat.pol_occ_tile_inds[occ_ind];
          int const wat_ind = pair_dat.rank_for_task[j];
          if (wat_ind >= pair_dat.n_tasks_per_line_element[j_pol_occ]) {
            // the last pol/occ pair for a line will report tasks w/ ranks
            // greater than the number of tasks it defines, so we have to
            // skip these tasks
            continue;
          }
          printf(
              "j %d, j_pol_occ %d, pair_ind %d, pol_ind %d occ_ind %d "
              "pol_tile_ind %d occ_tile_ind %d wat_ind %d\n",
              j,
              j_pol_occ,
              pair_ind,
              pol_ind,
              occ_ind,
              pol_tile_ind,
              occ_tile_ind,
              wat_ind);

          // Now we can go and compute exp_d2_delta for this water/occluder pair
          // This is the actual work! We have spent quite a few lines of C++
          // trying to get to this point
          Real const lj_radius_j = occ_dat.lk_ball_params[occ_ind].lj_radius;
          // code stollen from lk_fraction::V() above
          Real d2_low = (1.4 + lj_radius_j) * (1.4 + lj_radius_j)
                        - lkball_globals<Real>::ramp_width_A2;
          Real3 wat_xyz = coord_from_shared(
              pol_dat.water_coords, pol_tile_ind * MAX_N_WATER + wat_ind);
          Real3 occ_xyz = coord_from_shared(occ_dat.pose_coords, occ_tile_ind);
          Real const d2_delta = (occ_xyz - wat_xyz).squaredNorm() - d2_low;
          Real const exp_d2_delta = std::exp(-d2_delta);
          pair_dat.exp_d2_delta_for_task[j] = exp_d2_delta;
          printf("exp_d2_delta %d: %f\n", j, exp_d2_delta);
        }

        // Now that we have the exp_d2_deltas computed for each wat/occ pair,
        // perform an inclusive segmented scan on them so that we can sum all of
        // the exp_d2_deltas for the waters of each pol/occ pair; in the next
        // step, we will add those exp_d2_delta sums into an lk-fraction array
        // in shared memory
        DeviceDispatch<Dev>::
            template inclusive_seg_scan_in_workgroup<nt, TILE_SIZE>(
                pair_dat.exp_d2_delta_for_task,
                Real(0.0),
                pair_dat.task_is_first_task_for_line_element,
                mgpu::plus_t<Real>());
        printf("exp d2 delta for task after seg scan:");
        for (int i = 0; i < TILE_SIZE; ++i) {
          printf(" %6.3f", pair_dat.exp_d2_delta_for_task[i]);
        }
        printf("\n");

        // Now accumulate the partition function into the lk-fraction array
        // After we have accumulated all the exp_d2_deltas for all water/occ
        // pairs, then we can compute the lk_ball component of the energy
        for (int j = tid; j < TILE_SIZE; j += nt) {
          if (pair_dat.task_is_first_task_for_line_element[j]) {
            int const pol_occ_line_pair = pair_dat.line_element_for_task[j];
            int const pair_block_ind =
                TILE_SIZE * count_row + pol_occ_line_pair;
            int const pol_ind =
                pair_block_ind / occ_dat.n_occluders;  // [0..n_polars)
            if (pol_ind >= pol_dat.n_polars) {
              continue;  // I beleive this check is unneccessary;
                         // that we are guaranteed that pol < n_polars by
                         // the check to is_first_task_for_line_element above
            }
            int const pol_n_wat = pol_dat.n_waters[pol_ind];

            // The first task for a line element usually has rank 0,
            // but in the event that this is the first line element and that
            // the rank-0 task for the line element fell in a previous chunk,
            // then we have to account for that when trying to find the
            // index of the highest-rank task for this line element
            int const task_offset = pair_dat.rank_for_task[j];

            int const last_water_for_pol_pair =
                min(TILE_SIZE - 1, j + pol_n_wat - task_offset - 1);
            printf(
                "accum for j %d pol_occ_line_pair %d, pair_block_ind %d, "
                "pol_ind %d, pol_n_wat %d, last_water_for_pol_pair %d, "
                "exp_d2_delta_for_task %6.3f\n",
                j,
                pol_occ_line_pair,
                pair_block_ind,
                pol_ind,
                pol_n_wat,
                last_water_for_pol_pair,
                pair_dat.exp_d2_delta_for_task[last_water_for_pol_pair]);
            pair_dat.lk_fraction_exp_d2_delta_sum[pol_occ_line_pair] +=
                pair_dat.exp_d2_delta_for_task[last_water_for_pol_pair];
          }
        }
      }  // for wat_occ_count
    });

    DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(
        eval_lk_fraction_d2_delta_for_row);

    // with the d2 delta for every pol/occ pair in the row summed, we may now
    // complete the lk-fraction calculation: one iteration pol_occ pair in the
    // line
    auto eval_lk_fraction_for_row = ([&] TMOL_DEVICE_FUNC(int tid) {
      for (int i = tid; i < TILE_SIZE; i += nt) {
        int const pol_ind = i / occ_dat.n_occluders;
        if (pol_ind >= pol_dat.n_polars) {
          continue;
        }
        Real const exp_d2_delta_sum = pair_dat.lk_fraction_exp_d2_delta_sum[i];
        Real const wted_d2_delta = -std::log(exp_d2_delta_sum);

        Real frac = 0;
        if (wted_d2_delta < 0) {
          frac = 1;
        } else if (wted_d2_delta < lkball_globals<Real>::ramp_width_A2) {
          Real const val_a =
              wted_d2_delta / lkball_globals<Real>::ramp_width_A2;
          Real const val_b = 1 - val_a * val_a;
          frac = val_b * val_b;
        }

        pair_dat.lk_fraction[i] = frac;
      }
    });

    DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(
        eval_lk_fraction_for_row);

    // Step 3: Now we need to compute the water/water partition functions for
    // lk-bridge Same strategy with lk-fraction's exp_d2_delta: a. find out how
    // many water/water tasks each polar/occluder pair generates b. use scan to
    // find offset for each polar/occluder pair c. iterate across chunks of 32
    // water/water tasks at a time d. for a given chunk, identify the
    // polar/occluder pair for each task
    //    and the rank of every task
    // e. knowing wihch water/water pair to look at, evaluate the exp_d2_delta
    // f. use segmented scan to sum the water/water exp_d2_deltas
    // g. with one thread per group of water/water pairs on the same
    // polar/occluder pair,
    //    accumulate the exp_d2_delta sum into the per-polar/occluder sum
    // After the exp_d2_deltas are accumulated, then step 4 will compute
    // the lk-bridge-fraction for this line.
    auto eval_lk_bridge_fraction_d2_delta_for_row = ([&] TMOL_DEVICE_FUNC(
                                                         int tid) {
      DeviceDispatch<Dev>::synchronize_workgroup();

      // first, we count the number of water pairs each polar of each
      // polar/occluder pairs in this line has; we will use these counts to
      // generate offsets

      for (int i = tid; i < TILE_SIZE; i += nt) {
        int const pair_ind = TILE_SIZE * count_row + i;
        int const pol_ind = pair_ind / occ_dat.n_occluders;
        int const occ_ind = pair_ind % occ_dat.n_occluders;
        int const pol_n_waters =
            pol_ind < pol_dat.n_polars ? pol_dat.n_waters[pol_ind] : 0;
        int const occ_n_waters =
            pol_ind < pol_dat.n_polars ? occ_dat.n_waters[occ_ind] : 0;
        pair_dat.n_tasks_per_line_element[i] = pol_n_waters * occ_n_waters;
        pair_dat.task_offset_for_line_element[i] = pol_n_waters * occ_n_waters;

        // Let's take care of some other initialization while we are here
        pair_dat.lk_fraction_exp_d2_delta_sum[i] = 0;
        pair_dat.lk_bridge_fraction_exp_d2_delta_sum[i] = 0;
      }

      // Now perform an exclusive scan of the water counts and broadcast the
      // number of water/water pairs total to all threads
      int const n_wat_wat_pairs_in_row = DeviceDispatch<Dev>::
          template exclusive_scan_in_workgroup<nt, TILE_SIZE>(
              pair_dat.task_offset_for_line_element,
              int(0),
              mgpu::plus_t<int>(),
              true);
      printf("lk_bridge_fraction:\n");
      printf("pair_dat.n_tasks_per_line_element:");
      for (int i = 0; i < TILE_SIZE; ++i) {
        printf(" %d", pair_dat.n_tasks_per_line_element[i]);
      }
      printf("\n");
      printf("task offset:");
      for (int i = 0; i < TILE_SIZE; ++i) {
        printf(" %d", pair_dat.task_offset_for_line_element[i]);
      }
      printf("\n");

      // Ok, knowing how many water/water interactions we have, we will process
      // each chunk of TILE_SIZE water/water pairs. For each chunk, we will need
      // to figure out exactly which polar/occluder pair the focused water/water
      // pair comes from; essentially a "load balance search"
      int const n_wat_wat_iterations =
          (n_wat_wat_pairs_in_row - 1) / TILE_SIZE + 1;
      for (int wat_wat_count = 0; wat_wat_count < n_wat_wat_iterations;
           wat_wat_count++) {
        // Pause to make sure everything in shared memory has been read from
        // before we write any more things
        DeviceDispatch<Dev>::synchronize_workgroup();

        // We need to figure out what set of waters we're going to operate on
        // in this iteration. One way to do this is to examine each
        // polar/occluder pair's offset and water count and if its waters fall
        // in the range between TILE_SIZE * (wat_wat_count) and TILE_SIZE *
        // (wat_wat_count + 1) then we have a pol/occ pair that we will examine
        // in this itration Start by resetting some shared-memory arrays
        for (int j = tid; j < TILE_SIZE; j += nt) {
          pair_dat.line_element_for_task[j] = -1;
          pair_dat.task_is_first_task_for_line_element[j] = 0;
          pair_dat.exp_d2_delta_for_task[j] = 0;
        }
        DeviceDispatch<Dev>::synchronize_workgroup();

        // For each pol/occ pair in this row, figure out whether they are active
        // in this iteration if so, mark a "this position belongs to me" entry
        // in the line_element_for_task shared-memory array; we will use
        // scan(max) to fill in this array completely in the next step.
        for (int j = tid; j < TILE_SIZE; j += nt) {
          int const j_wat_offset = pair_dat.task_offset_for_line_element[j];
          int const j_n_waters = pair_dat.n_tasks_per_line_element[j];
          if (j_n_waters == 0) {
            continue;
          }
          if (wat_wat_count * TILE_SIZE <= j_wat_offset
              && j_wat_offset < (wat_wat_count + 1) * TILE_SIZE) {
            // ok, then we are operating on some or all of the waters for this
            // pol/occ pair and the first water for this pair is in the range
            int const j_tile_offset = j_wat_offset - wat_wat_count * TILE_SIZE;
            pair_dat.line_element_for_task[j_tile_offset] = j;
            pair_dat.task_is_first_task_for_line_element[j_tile_offset] = 1;
          } else if (
              j_wat_offset < wat_wat_count * TILE_SIZE
              && j_wat_offset + j_n_waters >= wat_wat_count * TILE_SIZE) {
            // ok, then some of the waters for this pose pair are within this
            // tile, but it won't be the first water
            pair_dat.line_element_for_task[0] = j;
            pair_dat.task_is_first_task_for_line_element[0] = 1;
          }
        }

        // With each pol/occ pair that will be examined in this iteration marked
        // in the which_pol_occ_for_wat_wat_pair, use an inclusive scan with
        // maximum to mark for each wat/wat pair which pol/occ pair it is coming
        // from
        DeviceDispatch<Dev>::
            template inclusive_scan_in_workgroup<nt, TILE_SIZE>(
                pair_dat.line_element_for_task,
                int(-1),
                mgpu::maximum_t<int32_t>(),
                false);
        printf("line element for task:");
        for (int i = 0; i < TILE_SIZE; ++i) {
          printf(" %d", pair_dat.line_element_for_task[i]);
        }
        printf("\n");

        // Now with the line_element_for_task array initialized, we need to
        // figure out the rank of the task for the pol/occ pair. This rank
        // information will be used again later in the computation, but could
        // likely be computed twice, as it is not expensive to compute

        for (int j = tid; j < TILE_SIZE; j += nt) {
          int const j_pol_occ = pair_dat.line_element_for_task[j];
          int const j_pol_occ_offset =
              pair_dat.task_offset_for_line_element[j_pol_occ];
          pair_dat.rank_for_task[j] =
              (wat_wat_count * TILE_SIZE + j) - j_pol_occ_offset;
        }

        printf("rank for task:");
        for (int i = 0; i < TILE_SIZE; ++i) {
          printf(" %d", pair_dat.rank_for_task[i]);
        }
        printf("\n");
        // OK! now we know exactly which pol/occ pair and which water each
        // wat/wat pair is going to examine. So let's compute the exp_d2_delta
        // for each wat/wat pair and store that into shared memory; we will then
        // perform a segmented scan over all water/water pairs for a given
        // pol/occ pair in the next step.
        for (int j = tid; j < TILE_SIZE; j += nt) {
          int const j_pol_occ = pair_dat.line_element_for_task[j];
          int const pair_ind = TILE_SIZE * count_row + j_pol_occ;
          int const pol_ind = pair_ind / occ_dat.n_occluders;
          if (pol_ind >= pol_dat.n_polars) {
            continue;
          }
          int const occ_ind = pair_ind % occ_dat.n_occluders;
          int const wat_ind = pair_dat.rank_for_task[j];
          if (wat_ind >= pair_dat.n_tasks_per_line_element[j_pol_occ]) {
            // the last pol/occ pair for a line will report tasks w/ ranks
            // greater than the number of tasks it defines, so we have to
            // skip these tasks
            continue;
          }

          int const pol_tile_ind = pol_dat.pol_occ_tile_inds[pol_ind];
          int const occ_tile_ind = occ_dat.pol_occ_tile_inds[occ_ind];
          int const pol_wat_ind = wat_ind / occ_dat.n_waters[occ_ind];
          int const occ_wat_ind = wat_ind % occ_dat.n_waters[occ_ind];

          if (pol_wat_ind >= pol_dat.n_waters[pol_ind]) {
            continue;
          }

          printf(
              "j %d, j_pol_occ %d, pair_ind %d, pol_ind %d occ_ind %d wat_ind "
              "%d, pol_tile_ind %d, occ_tile_ind %d, pol_wat_ind %d "
              "occ_wat_ind %d\n",
              j,
              j_pol_occ,
              pair_ind,
              pol_ind,
              occ_ind,
              wat_ind,
              pol_tile_ind,
              occ_tile_ind,
              pol_wat_ind,
              occ_wat_ind);

          // Stealing code from lk_bridge_fraction
          Real const overlap_gap_A2 = lkball_globals<Real>::overlap_gap_A2;
          Real3 pol_wat_xyz = coord_from_shared(
              pol_dat.water_coords, pol_tile_ind * MAX_N_WATER + pol_wat_ind);
          Real3 occ_wat_xyz = coord_from_shared(
              occ_dat.water_coords, occ_tile_ind * MAX_N_WATER + occ_wat_ind);

          printf(
              "j %d waters: (%6.3f %6.3f %6.3f) vs (%6.3f %6.3f %6.3f)\n",
              j,
              pol_wat_xyz[0],
              pol_wat_xyz[1],
              pol_wat_xyz[2],
              occ_wat_xyz[0],
              occ_wat_xyz[1],
              occ_wat_xyz[2]);

          Real const d2_delta =
              (pol_wat_xyz - occ_wat_xyz).squaredNorm() - overlap_gap_A2;
          Real const exp_d2_delta = std::exp(-d2_delta);
          pair_dat.exp_d2_delta_for_task[j] = exp_d2_delta;
          printf("exp_d2_delta %d: %f\n", j, exp_d2_delta);
        }

        // Now that we have the exp_d2_deltas computed for each wat/occ pair,
        // perform a segmented scan on them so that we can sum all of the
        // exp_d2_deltas for the waters of each pol/occ pair; in the next step,
        // we will add those exp_d2_delta sums into an lk-fraction array in
        // shared memory
        DeviceDispatch<Dev>::
            template inclusive_seg_scan_in_workgroup<nt, TILE_SIZE>(
                pair_dat.exp_d2_delta_for_task,
                Real(0.0),
                pair_dat.task_is_first_task_for_line_element,
                mgpu::plus_t<Real>());
        printf("exp d2 delta for task after seg scan:");
        for (int i = 0; i < TILE_SIZE; ++i) {
          printf(" %6.3f", pair_dat.exp_d2_delta_for_task[i]);
        }
        printf("\n");

        // Now accumulate the partition function into the lk-fraction array
        // After we have accumulated all the exp_d2_deltas for all water/occ
        // pairs, then we can compute the lk_ball component of the energy
        for (int j = tid; j < TILE_SIZE; j += nt) {
          if (pair_dat.task_is_first_task_for_line_element[j]) {
            int const pol_pair = pair_dat.line_element_for_task[j];
            int const pair_ind = TILE_SIZE * count_row + pol_pair;
            int const pol_ind = pair_ind / occ_dat.n_occluders;
            if (pol_ind >= pol_dat.n_polars) {
              continue;  // unneccessary; guaranteed that this is true by
                         // the check first_pol_water_for_wat_occ_pair above
            }
            int const n_wat_pairs = pair_dat.n_tasks_per_line_element[pol_pair];

            // Rank is usually 0 for the first task for a line element,
            // but in the event that this is the first line element and that
            // the rank-0 task for the line element fell in a previous chunk,
            // then we have to account for that when trying to find the
            // index of the highest-rank task for this line element
            int const task_offset = pair_dat.rank_for_task[j];

            int const last_water_for_pol_pair =
                min(TILE_SIZE - 1, j + n_wat_pairs - task_offset - 1);
            printf(
                "accum for j %d pol_pair %d, pair_ind %d, pol_ind %d, "
                "n_wat_pairs %d, last_water_for_pol_pair %d, "
                "exp_d2_delta_for_task %6.3f\n",
                j,
                pol_pair,
                pair_ind,
                pol_ind,
                n_wat_pairs,
                last_water_for_pol_pair,
                pair_dat.exp_d2_delta_for_task[last_water_for_pol_pair]);
            pair_dat.lk_bridge_fraction_exp_d2_delta_sum[pol_pair] +=
                pair_dat.exp_d2_delta_for_task[last_water_for_pol_pair];
          }
        }
      }  // for wat_wat_count
    });

    DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(
        eval_lk_bridge_fraction_d2_delta_for_row);

    // with the d2 delta for every pol/occ pair in the row summed, we may now
    // complete the lk-fraction calculation:
    auto eval_lk_bridge_fraction_for_row = ([&] TMOL_DEVICE_FUNC(int tid) {
      for (int i = tid; i < TILE_SIZE; i += nt) {
        int const pair_ind = TILE_SIZE * count_row + i;
        int const pol_ind = pair_ind / occ_dat.n_occluders;
        int const occ_ind = pair_ind % occ_dat.n_occluders;
        if (pol_ind >= pol_dat.n_polars) {
          pair_dat.lk_bridge_fraction[i] = 0;
          continue;
        }
        int const pol_tile_ind = pol_dat.pol_occ_tile_inds[pol_ind];
        int const occ_tile_ind = occ_dat.pol_occ_tile_inds[occ_ind];
        Real3 pol_xyz = coord_from_shared(pol_dat.pose_coords, pol_tile_ind);
        Real3 occ_xyz = coord_from_shared(occ_dat.pose_coords, occ_tile_ind);

        Real const exp_d2_delta_sum =
            pair_dat.lk_bridge_fraction_exp_d2_delta_sum[i];
        Real const wted_d2_delta = -std::log(exp_d2_delta_sum);

        static constexpr Real overlap_width_A2 =
            lkball_globals<Real>::overlap_width_A2;
        static constexpr Real angle_overlap_A2 =
            lkball_globals<Real>::angle_overlap_A2;
        Real overlapfrac;
        if (wted_d2_delta > overlap_width_A2) {
          overlapfrac = 0;
        } else {
          // square-square -> 1 as x -> 0
          Real const val_a = wted_d2_delta / overlap_width_A2;
          Real const val_b = 1 - val_a * val_a;
          overlapfrac = val_b * val_b;
        }
        // base angle
        Real lkb_water_dist2 = pair_dat.global_params.lkb_water_dist
                               * pair_dat.global_params.lkb_water_dist;
        Real overlap_target_len2 = 8.0 / 3.0 * lkb_water_dist2;

        Real overlap_len2 = (pol_xyz - occ_xyz).squaredNorm();
        Real base_delta = fabs(overlap_len2 - overlap_target_len2);

        Real anglefrac;
        if (base_delta > angle_overlap_A2) {
          anglefrac = 0;
        } else {
          // square-square -> 1 as x -> 0
          Real const val_a = base_delta / angle_overlap_A2;
          Real const val_b = 1 - val_a * val_a;
          anglefrac = val_b * val_b;
        }

        pair_dat.lk_bridge_fraction[i] = overlapfrac * anglefrac;
      }
    });

    DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(
        eval_lk_bridge_fraction_for_row);

    auto accum_totals_in_registers = ([&] TMOL_DEVICE_FUNC(int tid) {
      for (int i = tid; i < TILE_SIZE; i += nt) {
        Real const lk_iso = pair_dat.lk_iso[i];
        printf("lk_iso[%d] = %9.5f\n", i, lk_iso);
        pair_dat.total_lk_ball_iso += lk_iso;
        pair_dat.total_lk_ball += lk_iso * pair_dat.lk_fraction[i];
        pair_dat.total_lk_bridge += lk_iso * pair_dat.lk_bridge_fraction[i];
        pair_dat.total_lk_bridge_uncpl += pair_dat.lk_bridge_fraction[i] / 2;
      }
    });

    DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(
        accum_totals_in_registers);
    // printf("lk_iso:");
    // for (int i = 0; i < TILE_SIZE; ++i) {
    //   printf(" %f", pair_dat.lk_iso[i]);
    // }
    // printf("\n");

  }  // for count_row
}

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    int TILE_SIZE,
    int MAX_N_WATER,
    typename Real>
void TMOL_DEVICE_FUNC eval_interres_pol_occ_pair_energies_new(
    LKBallScoringData2<Real> &inter_dat, int start_atom1, int start_atom2) {
  load_balanced_interres_polar_occluder_tile<
      DeviceDispatch,
      Dev,
      nt,
      TILE_SIZE,
      MAX_N_WATER>(inter_dat.r1, inter_dat.r2, inter_dat.pair_data, true);
  load_balanced_interres_polar_occluder_tile<
      DeviceDispatch,
      Dev,
      nt,
      TILE_SIZE,
      MAX_N_WATER>(inter_dat.r2, inter_dat.r1, inter_dat.pair_data, false);
}

}  // namespace potentials
}  // namespace lk_ball
}  // namespace score
}  // namespace tmol
