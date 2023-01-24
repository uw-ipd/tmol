#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <cppitertools/range.hpp>

#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/data_loading.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/hbond/identification.hh>

#include "params.hh"

#undef B0

namespace tmol {
namespace score {
namespace lk_ball {
namespace potentials {

#define def auto EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <typename Real>
struct build_don_water {
  typedef Vec<Real, 3> Real3;
  typedef Eigen::Matrix<Real, 3, 3> RealMat;

  struct dV_t {
    RealMat dD;
    RealMat dH;

    def astuple() { return tmol::score::common::make_tuple(dD, dH); }

    static def Zero()->dV_t { return {RealMat::Zero(), RealMat::Zero()}; }
  };

  static def V(Real3 D, Real3 H, Real dist)->Real3 {
    return D + dist * (H - D).normalized();
  }

  static def square(Real v)->Real { return v * v; }

  static def dV(Real3 D, Real3 H, Real dist)->dV_t {
    Real dhx = -D[0] + H[0];
    Real dhx2 = dhx * dhx;
    Real dhy = -D[1] + H[1];
    Real dhy2 = dhy * dhy;
    Real dhz = -D[2] + H[2];
    Real dhz2 = dhz * dhz;
    Real dh2 = dhx2 + dhy2 + dhz2;
    Real dist_norm = dist / std::sqrt(dh2);
    Real dist_norm_deriv = dist_norm / dh2;

    Eigen::Matrix<Real, 3, 3> dW_dD;
    dW_dD(0, 0) = dhx2 * dist_norm / dh2 + (1 - dist_norm);
    dW_dD(0, 1) = dhy * dhx * dist_norm_deriv;
    dW_dD(0, 2) = dhz * dhx * dist_norm_deriv;
    dW_dD(1, 0) = dhy * dhx * dist_norm_deriv;
    dW_dD(1, 1) = dhy2 * dist_norm_deriv + (1 - dist_norm);
    dW_dD(1, 2) = dhz * dhy * dist_norm_deriv;
    dW_dD(2, 0) = dhz * dhx * dist_norm_deriv;
    dW_dD(2, 1) = dhz * dhy * dist_norm_deriv;
    dW_dD(2, 2) = dhz2 * dist_norm_deriv + (1 - dist_norm);

    Eigen::Matrix<Real, 3, 3> dW_dH;
    dW_dH(0, 0) = dist_norm - dhx * dist_norm_deriv * dhx;
    dW_dH(0, 1) = -dhy * dist_norm_deriv * dhx;
    dW_dH(0, 2) = -dhx * dhz * dist_norm_deriv;
    dW_dH(1, 0) = -dhx * dist_norm_deriv * dhy;
    dW_dH(1, 1) = dist_norm - dhy * dist_norm_deriv * dhy;
    dW_dH(1, 2) = -dhz * dist_norm_deriv * dhy;
    dW_dH(2, 0) = -dhx * dist_norm_deriv * dhz;
    dW_dH(2, 1) = -dhy * dist_norm_deriv * dhz;
    dW_dH(2, 2) = dist_norm - dhz * dist_norm_deriv * dhz;

    return {dW_dD, dW_dH};
  }
};

template <typename Real>
struct build_acc_water {
  typedef Vec<Real, 3> Real3;
  typedef Eigen::Matrix<Real, 3, 3> RealMat;

  struct dV_t {
    RealMat dA;
    RealMat dB;
    RealMat dB0;

    def astuple() { return tmol::score::common::make_tuple(dA, dB, dB0); }

    static def Zero()->dV_t {
      return {RealMat::Zero(), RealMat::Zero(), RealMat::Zero()};
    }
  };

  static def V(Real3 A, Real3 B, Real3 B0, Real dist, Real angle, Real torsion)
      ->Real3 {
    const Real pi = EIGEN_PI;

    // Generate orientation frame
    Eigen::Matrix<Real, 3, 3> M;
    M.col(0) = (A - B).normalized();
    M.col(1) = (B0 - B);
    M.col(2) = M.col(0).cross(M.col(1));

    Real M2_norm = M.col(2).norm();
    if (M2_norm == 0) {
      // if a/b/c collinear, set M[:,2] to an arbitrary vector perp to
      // M[:,0]
      if (M(0, 0) != 1) {
        M.col(1) = Real3({1, 0, 0});
        M.col(2) = M.col(0).cross(M.col(1));
      } else {
        M.col(1) = Real3({0, 1, 0});
        M.col(2) = M.col(0).cross(M.col(1));
      }
      M2_norm = M.col(2).norm();
    }
    M.col(2) /= M2_norm;
    M.col(1) = M.col(2).cross(M.col(0));

    // Build water in frame
    return (
        M
            * Real3({dist * std::cos(pi - angle),
                     dist * std::sin(pi - angle) * std::cos(torsion),
                     dist * std::sin(pi - angle) * std::sin(torsion)})
        + A);
  }

  static def dV(Real3 A, Real3 B, Real3 B0, Real dist, Real angle, Real torsion)
      ->dV_t {
    const Real pi = EIGEN_PI;

    Real sin_a = std::sin(pi - angle);
    Real cos_a = std::cos(pi - angle);
    Real sin_t = std::sin(torsion);
    Real cos_t = std::cos(torsion);
    Real3 AB = A - B;
    Real3 B0B = B0 - B;
    Real a_m_b = AB.norm();
    Real a_m_b2 = a_m_b * a_m_b;
    Real b0_m_b = B0B.norm();
    Real3 ABXB0B = AB.cross(B0B);
    Real abxb0b2 = ABXB0B.squaredNorm();

    dV_t dW;
    dW.dA(0, 0) =
        (-ABXB0B[0] * AB[0] * dist * sin_a * sin_t
             / (a_m_b2 * a_m_b * std::sqrt(abxb0b2 / a_m_b2))
         + ABXB0B[0] * a_m_b * dist * sin_a * sin_t
               * (-(-ABXB0B[1] * (2 * B0[2] - 2 * B[2])
                    + ABXB0B[2] * (2 * B0[1] - 2 * B[1]))
                      / (2 * a_m_b2)
                  - abxb0b2 * (-2 * A[0] + 2 * B[0]) / (2 * a_m_b2 * a_m_b2))
               / (abxb0b2 * std::sqrt(abxb0b2 / a_m_b2))
         - AB[0] * AB[0] * cos_a * dist / (a_m_b2 * a_m_b) + 1
         - cos_t * dist * sin_a * (-ABXB0B[1] * AB[2] + ABXB0B[2] * AB[1])
               * (-(-ABXB0B[1] * (2 * B0[2] - 2 * B[2])
                    + ABXB0B[2] * (2 * B0[1] - 2 * B[1]))
                      / (2 * a_m_b2)
                  - abxb0b2 * (-2 * A[0] + 2 * B[0]) / (2 * a_m_b2 * a_m_b2))
               / (abxb0b2 * std::sqrt(abxb0b2 / a_m_b2))
         - cos_t * dist * sin_a * (AB[1] * B0B[1] + AB[2] * B0B[2])
               / (a_m_b2 * std::sqrt(abxb0b2 / a_m_b2))
         - cos_t * dist * sin_a * (-2 * A[0] + 2 * B[0])
               * (-ABXB0B[1] * AB[2] + ABXB0B[2] * AB[1])
               / (a_m_b2 * a_m_b2 * std::sqrt(abxb0b2 / a_m_b2))
         + cos_a * dist / a_m_b);
    dW.dA(0, 1) =
        (-ABXB0B[1] * AB[0] * dist * sin_a * sin_t
             / (a_m_b2 * a_m_b * std::sqrt(abxb0b2 / a_m_b2))
         + ABXB0B[1] * a_m_b * dist * sin_a * sin_t
               * (-(-ABXB0B[1] * (2 * B0[2] - 2 * B[2])
                    + ABXB0B[2] * (2 * B0[1] - 2 * B[1]))
                      / (2 * a_m_b2)
                  - abxb0b2 * (-2 * A[0] + 2 * B[0]) / (2 * a_m_b2 * a_m_b2))
               / (abxb0b2 * std::sqrt(abxb0b2 / a_m_b2))
         - AB[0] * AB[1] * cos_a * dist / (a_m_b2 * a_m_b)
         - B0B[2] * dist * sin_a * sin_t / (a_m_b * std::sqrt(abxb0b2 / a_m_b2))
         + cos_t * dist * sin_a * (-ABXB0B[0] * AB[2] + ABXB0B[2] * AB[0])
               * (-(-ABXB0B[1] * (2 * B0[2] - 2 * B[2])
                    + ABXB0B[2] * (2 * B0[1] - 2 * B[1]))
                      / (2 * a_m_b2)
                  - abxb0b2 * (-2 * A[0] + 2 * B[0]) / (2 * a_m_b2 * a_m_b2))
               / (abxb0b2 * std::sqrt(abxb0b2 / a_m_b2))
         + cos_t * dist * sin_a * (2 * AB[0] * B0B[1] - AB[1] * B0B[0])
               / (a_m_b2 * std::sqrt(abxb0b2 / a_m_b2))
         + cos_t * dist * sin_a * (-2 * A[0] + 2 * B[0])
               * (-ABXB0B[0] * AB[2] + ABXB0B[2] * AB[0])
               / (a_m_b2 * a_m_b2 * std::sqrt(abxb0b2 / a_m_b2)));
    dW.dA(0, 2) =
        (-ABXB0B[2] * AB[0] * dist * sin_a * sin_t
             / (a_m_b2 * a_m_b * std::sqrt(abxb0b2 / a_m_b2))
         + ABXB0B[2] * a_m_b * dist * sin_a * sin_t
               * (-(-ABXB0B[1] * (2 * B0[2] - 2 * B[2])
                    + ABXB0B[2] * (2 * B0[1] - 2 * B[1]))
                      / (2 * a_m_b2)
                  - abxb0b2 * (-2 * A[0] + 2 * B[0]) / (2 * a_m_b2 * a_m_b2))
               / (abxb0b2 * std::sqrt(abxb0b2 / a_m_b2))
         - AB[0] * AB[2] * cos_a * dist / (a_m_b2 * a_m_b)
         + B0B[1] * dist * sin_a * sin_t / (a_m_b * std::sqrt(abxb0b2 / a_m_b2))
         + cos_t * dist * sin_a * (ABXB0B[0] * AB[1] - ABXB0B[1] * AB[0])
               * (-(-ABXB0B[1] * (2 * B0[2] - 2 * B[2])
                    + ABXB0B[2] * (2 * B0[1] - 2 * B[1]))
                      / (2 * a_m_b2)
                  - abxb0b2 * (-2 * A[0] + 2 * B[0]) / (2 * a_m_b2 * a_m_b2))
               / (abxb0b2 * std::sqrt(abxb0b2 / a_m_b2))
         + cos_t * dist * sin_a * (2 * AB[0] * B0B[2] - AB[2] * B0B[0])
               / (a_m_b2 * std::sqrt(abxb0b2 / a_m_b2))
         + cos_t * dist * sin_a * (-2 * A[0] + 2 * B[0])
               * (ABXB0B[0] * AB[1] - ABXB0B[1] * AB[0])
               / (a_m_b2 * a_m_b2 * std::sqrt(abxb0b2 / a_m_b2)));
    dW.dA(1, 0) =
        (-ABXB0B[0] * AB[1] * dist * sin_a * sin_t
             / (a_m_b2 * a_m_b * std::sqrt(abxb0b2 / a_m_b2))
         + ABXB0B[0] * a_m_b * dist * sin_a * sin_t
               * (-(ABXB0B[0] * (2 * B0[2] - 2 * B[2])
                    + ABXB0B[2] * (-2 * B0[0] + 2 * B[0]))
                      / (2 * a_m_b2)
                  - abxb0b2 * (-2 * A[1] + 2 * B[1]) / (2 * a_m_b2 * a_m_b2))
               / (abxb0b2 * std::sqrt(abxb0b2 / a_m_b2))
         - AB[0] * AB[1] * cos_a * dist / (a_m_b * a_m_b2)
         + B0B[2] * dist * sin_a * sin_t / (a_m_b * std::sqrt(abxb0b2 / a_m_b2))
         - cos_t * dist * sin_a * (-ABXB0B[1] * AB[2] + ABXB0B[2] * AB[1])
               * (-(ABXB0B[0] * (2 * B0[2] - 2 * B[2])
                    + ABXB0B[2] * (-2 * B0[0] + 2 * B[0]))
                      / (2 * a_m_b2)
                  - abxb0b2 * (-2 * A[1] + 2 * B[1]) / (2 * a_m_b2 * a_m_b2))
               / (abxb0b2 * std::sqrt(abxb0b2 / a_m_b2))
         - cos_t * dist * sin_a * (AB[0] * B0B[1] - 2 * AB[1] * B0B[0])
               / (a_m_b2 * std::sqrt(abxb0b2 / a_m_b2))
         - cos_t * dist * sin_a * (-2 * A[1] + 2 * B[1])
               * (-ABXB0B[1] * AB[2] + ABXB0B[2] * AB[1])
               / (a_m_b2 * a_m_b2 * std::sqrt(abxb0b2 / a_m_b2)));
    dW.dA(1, 1) =
        (-ABXB0B[1] * AB[1] * dist * sin_a * sin_t
             / (a_m_b2 * a_m_b * std::sqrt(abxb0b2 / a_m_b2))
         + ABXB0B[1] * a_m_b * dist * sin_a * sin_t
               * (-(ABXB0B[0] * (2 * B0[2] - 2 * B[2])
                    + ABXB0B[2] * (-2 * B0[0] + 2 * B[0]))
                      / (2 * a_m_b2)
                  - abxb0b2 * (-2 * A[1] + 2 * B[1]) / (2 * a_m_b2 * a_m_b2))
               / (abxb0b2 * std::sqrt(abxb0b2 / a_m_b2))
         - AB[1] * AB[1] * cos_a * dist / (a_m_b2 * a_m_b) + 1
         + cos_t * dist * sin_a * (-ABXB0B[0] * AB[2] + ABXB0B[2] * AB[0])
               * (-(ABXB0B[0] * (2 * B0[2] - 2 * B[2])
                    + ABXB0B[2] * (-2 * B0[0] + 2 * B[0]))
                      / (2 * a_m_b2)
                  - abxb0b2 * (-2 * A[1] + 2 * B[1]) / (2 * a_m_b2 * a_m_b2))
               / (abxb0b2 * std::sqrt(abxb0b2 / a_m_b2))
         + cos_t * dist * sin_a * (-AB[0] * B0B[0] - AB[2] * B0B[2])
               / (a_m_b2 * std::sqrt(abxb0b2 / a_m_b2))
         + cos_t * dist * sin_a * (-2 * A[1] + 2 * B[1])
               * (-ABXB0B[0] * AB[2] + ABXB0B[2] * AB[0])
               / (a_m_b2 * a_m_b2 * std::sqrt(abxb0b2 / a_m_b2))
         + cos_a * dist / a_m_b);
    dW.dA(1, 2) =
        (-ABXB0B[2] * AB[1] * dist * sin_a * sin_t
             / (a_m_b2 * a_m_b * std::sqrt(abxb0b2 / a_m_b2))
         + ABXB0B[2] * a_m_b * dist * sin_a * sin_t
               * (-(ABXB0B[0] * (2 * B0[2] - 2 * B[2])
                    + ABXB0B[2] * (-2 * B0[0] + 2 * B[0]))
                      / (2 * a_m_b2)
                  - abxb0b2 * (-2 * A[1] + 2 * B[1]) / (2 * a_m_b2 * a_m_b2))
               / (abxb0b2 * std::sqrt(abxb0b2 / a_m_b2))
         - AB[1] * AB[2] * cos_a * dist / (a_m_b * a_m_b2)
         - B0B[0] * dist * sin_a * sin_t / (a_m_b * std::sqrt(abxb0b2 / a_m_b2))
         + cos_t * dist * sin_a * (ABXB0B[0] * AB[1] - ABXB0B[1] * AB[0])
               * (-(ABXB0B[0] * (2 * B0[2] - 2 * B[2])
                    + ABXB0B[2] * (-2 * B0[0] + 2 * B[0]))
                      / (2 * a_m_b2)
                  - abxb0b2 * (-2 * A[1] + 2 * B[1]) / (2 * a_m_b2 * a_m_b2))
               / (abxb0b2 * std::sqrt(abxb0b2 / a_m_b2))
         + cos_t * dist * sin_a * (2 * AB[1] * B0B[2] - AB[2] * B0B[1])
               / (a_m_b2 * std::sqrt(abxb0b2 / a_m_b2))
         + cos_t * dist * sin_a * (-2 * A[1] + 2 * B[1])
               * (ABXB0B[0] * AB[1] - ABXB0B[1] * AB[0])
               / (a_m_b2 * a_m_b2 * std::sqrt(abxb0b2 / a_m_b2)));
    dW.dA(2, 0) =
        (-ABXB0B[0] * AB[2] * dist * sin_a * sin_t
             / (a_m_b2 * a_m_b * std::sqrt(abxb0b2 / a_m_b2))
         + ABXB0B[0] * a_m_b * dist * sin_a * sin_t
               * (-(ABXB0B[0] * (-2 * B0[1] + 2 * B[1])
                    - ABXB0B[1] * (-2 * B0[0] + 2 * B[0]))
                      / (2 * a_m_b2)
                  - abxb0b2 * (-2 * A[2] + 2 * B[2]) / (2 * a_m_b2 * a_m_b2))
               / (abxb0b2 * std::sqrt(abxb0b2 / a_m_b2))
         - AB[0] * AB[2] * cos_a * dist / (a_m_b * a_m_b2)
         - B0B[1] * dist * sin_a * sin_t / (a_m_b * std::sqrt(abxb0b2 / a_m_b2))
         - cos_t * dist * sin_a * (-ABXB0B[1] * AB[2] + ABXB0B[2] * AB[1])
               * (-(ABXB0B[0] * (-2 * B0[1] + 2 * B[1])
                    - ABXB0B[1] * (-2 * B0[0] + 2 * B[0]))
                      / (2 * a_m_b2)
                  - abxb0b2 * (-2 * A[2] + 2 * B[2]) / (2 * a_m_b2 * a_m_b2))
               / (abxb0b2 * std::sqrt(abxb0b2 / a_m_b2))
         - cos_t * dist * sin_a * (AB[0] * B0B[2] - 2 * AB[2] * B0B[0])
               / (a_m_b2 * std::sqrt(abxb0b2 / a_m_b2))
         - cos_t * dist * sin_a * (-2 * A[2] + 2 * B[2])
               * (-ABXB0B[1] * AB[2] + ABXB0B[2] * AB[1])
               / (a_m_b2 * a_m_b2 * std::sqrt(abxb0b2 / a_m_b2)));
    dW.dA(2, 1) =
        (-ABXB0B[1] * AB[2] * dist * sin_a * sin_t
             / (a_m_b2 * a_m_b * std::sqrt(abxb0b2 / a_m_b2))
         + ABXB0B[1] * a_m_b * dist * sin_a * sin_t
               * (-(ABXB0B[0] * (-2 * B0[1] + 2 * B[1])
                    - ABXB0B[1] * (-2 * B0[0] + 2 * B[0]))
                      / (2 * a_m_b2)
                  - abxb0b2 * (-2 * A[2] + 2 * B[2]) / (2 * a_m_b2 * a_m_b2))
               / (abxb0b2 * std::sqrt(abxb0b2 / a_m_b2))
         - AB[1] * AB[2] * cos_a * dist / (a_m_b * a_m_b2)
         + B0B[0] * dist * sin_a * sin_t / (a_m_b * std::sqrt(abxb0b2 / a_m_b2))
         + cos_t * dist * sin_a * (-ABXB0B[0] * AB[2] + ABXB0B[2] * AB[0])
               * (-(ABXB0B[0] * (-2 * B0[1] + 2 * B[1])
                    - ABXB0B[1] * (-2 * B0[0] + 2 * B[0]))
                      / (2 * a_m_b2)
                  - abxb0b2 * (-2 * A[2] + 2 * B[2]) / (2 * a_m_b2 * a_m_b2))
               / (abxb0b2 * std::sqrt(abxb0b2 / a_m_b2))
         + cos_t * dist * sin_a * (-AB[1] * B0B[2] + 2 * AB[2] * B0B[1])
               / (a_m_b2 * std::sqrt(abxb0b2 / a_m_b2))
         + cos_t * dist * sin_a * (-2 * A[2] + 2 * B[2])
               * (-ABXB0B[0] * AB[2] + ABXB0B[2] * AB[0])
               / (a_m_b2 * a_m_b2 * std::sqrt(abxb0b2 / a_m_b2)));
    dW.dA(2, 2) =
        (-ABXB0B[2] * AB[2] * dist * sin_a * sin_t
             / (a_m_b2 * a_m_b * std::sqrt(abxb0b2 / a_m_b2))
         + ABXB0B[2] * a_m_b * dist * sin_a * sin_t
               * (-(ABXB0B[0] * (-2 * B0[1] + 2 * B[1])
                    - ABXB0B[1] * (-2 * B0[0] + 2 * B[0]))
                      / (2 * a_m_b2)
                  - abxb0b2 * (-2 * A[2] + 2 * B[2]) / (2 * a_m_b2 * a_m_b2))
               / (abxb0b2 * std::sqrt(abxb0b2 / a_m_b2))
         - AB[2] * AB[2] * cos_a * dist / (a_m_b * a_m_b2) + 1
         + cos_t * dist * sin_a * (ABXB0B[0] * AB[1] - ABXB0B[1] * AB[0])
               * (-(ABXB0B[0] * (-2 * B0[1] + 2 * B[1])
                    - ABXB0B[1] * (-2 * B0[0] + 2 * B[0]))
                      / (2 * a_m_b2)
                  - abxb0b2 * (-2 * A[2] + 2 * B[2]) / (2 * a_m_b2 * a_m_b2))
               / (abxb0b2 * std::sqrt(abxb0b2 / a_m_b2))
         + cos_t * dist * sin_a * (-AB[0] * B0B[0] - AB[1] * B0B[1])
               / (a_m_b2 * std::sqrt(abxb0b2 / a_m_b2))
         + cos_t * dist * sin_a * (-2 * A[2] + 2 * B[2])
               * (ABXB0B[0] * AB[1] - ABXB0B[1] * AB[0])
               / (a_m_b2 * a_m_b2 * std::sqrt(abxb0b2 / a_m_b2))
         + cos_a * dist / a_m_b);

    dW.dB(0, 0) =
        (ABXB0B[0] * AB[0] * dist * sin_a * sin_t
             / (a_m_b2 * a_m_b * std::sqrt(abxb0b2 / a_m_b2))
         + ABXB0B[0] * a_m_b * dist * sin_a * sin_t
               * (-(-ABXB0B[1] * (2 * A[2] - 2 * B0[2])
                    + ABXB0B[2] * (2 * A[1] - 2 * B0[1]))
                      / (2 * a_m_b2)
                  - abxb0b2 * (2 * A[0] - 2 * B[0]) / (2 * a_m_b2 * a_m_b2))
               / (abxb0b2 * std::sqrt(abxb0b2 / a_m_b2))
         + AB[0] * AB[0] * cos_a * dist / (a_m_b * a_m_b2)
         - cos_t * dist * sin_a * (-ABXB0B[1] * AB[2] + ABXB0B[2] * AB[1])
               * (-(-ABXB0B[1] * (2 * A[2] - 2 * B0[2])
                    + ABXB0B[2] * (2 * A[1] - 2 * B0[1]))
                      / (2 * a_m_b2)
                  - abxb0b2 * (2 * A[0] - 2 * B[0]) / (2 * a_m_b2 * a_m_b2))
               / (abxb0b2 * std::sqrt(abxb0b2 / a_m_b2))
         - cos_t * dist * sin_a
               * (AB[1] * (A[1] - B0[1]) + AB[2] * (A[2] - B0[2]))
               / (a_m_b2 * std::sqrt(abxb0b2 / a_m_b2))
         - cos_t * dist * sin_a * (2 * A[0] - 2 * B[0])
               * (-ABXB0B[1] * AB[2] + ABXB0B[2] * AB[1])
               / (a_m_b2 * a_m_b2 * std::sqrt(abxb0b2 / a_m_b2))
         - cos_a * dist / a_m_b);
    dW.dB(0, 1) =
        (ABXB0B[1] * AB[0] * dist * sin_a * sin_t
             / (a_m_b2 * a_m_b * std::sqrt(abxb0b2 / a_m_b2))
         + ABXB0B[1] * a_m_b * dist * sin_a * sin_t
               * (-(-ABXB0B[1] * (2 * A[2] - 2 * B0[2])
                    + ABXB0B[2] * (2 * A[1] - 2 * B0[1]))
                      / (2 * a_m_b2)
                  - abxb0b2 * (2 * A[0] - 2 * B[0]) / (2 * a_m_b2 * a_m_b2))
               / (abxb0b2 * std::sqrt(abxb0b2 / a_m_b2))
         + AB[0] * AB[1] * cos_a * dist / (a_m_b * a_m_b2)
         + cos_t * dist * sin_a * (-ABXB0B[0] * AB[2] + ABXB0B[2] * AB[0])
               * (-(-ABXB0B[1] * (2 * A[2] - 2 * B0[2])
                    + ABXB0B[2] * (2 * A[1] - 2 * B0[1]))
                      / (2 * a_m_b2)
                  - abxb0b2 * (2 * A[0] - 2 * B[0]) / (2 * a_m_b2 * a_m_b2))
               / (abxb0b2 * std::sqrt(abxb0b2 / a_m_b2))
         + cos_t * dist * sin_a * (-ABXB0B[2] + AB[0] * (A[1] - B0[1]))
               / (a_m_b2 * std::sqrt(abxb0b2 / a_m_b2))
         + cos_t * dist * sin_a * (2 * A[0] - 2 * B[0])
               * (-ABXB0B[0] * AB[2] + ABXB0B[2] * AB[0])
               / (a_m_b2 * a_m_b2 * std::sqrt(abxb0b2 / a_m_b2))
         + dist * sin_a * sin_t * (-A[2] + B0[2])
               / (a_m_b * std::sqrt(abxb0b2 / a_m_b2)));
    dW.dB(0, 2) =
        (ABXB0B[2] * AB[0] * dist * sin_a * sin_t
             / (a_m_b2 * a_m_b * std::sqrt(abxb0b2 / a_m_b2))
         + ABXB0B[2] * a_m_b * dist * sin_a * sin_t
               * (-(-ABXB0B[1] * (2 * A[2] - 2 * B0[2])
                    + ABXB0B[2] * (2 * A[1] - 2 * B0[1]))
                      / (2 * a_m_b2)
                  - abxb0b2 * (2 * A[0] - 2 * B[0]) / (2 * a_m_b2 * a_m_b2))
               / (abxb0b2 * std::sqrt(abxb0b2 / a_m_b2))
         + AB[0] * AB[2] * cos_a * dist / (a_m_b * a_m_b2)
         + cos_t * dist * sin_a * (ABXB0B[0] * AB[1] - ABXB0B[1] * AB[0])
               * (-(-ABXB0B[1] * (2 * A[2] - 2 * B0[2])
                    + ABXB0B[2] * (2 * A[1] - 2 * B0[1]))
                      / (2 * a_m_b2)
                  - abxb0b2 * (2 * A[0] - 2 * B[0]) / (2 * a_m_b2 * a_m_b2))
               / (abxb0b2 * std::sqrt(abxb0b2 / a_m_b2))
         + cos_t * dist * sin_a * (ABXB0B[1] + AB[0] * (A[2] - B0[2]))
               / (a_m_b2 * std::sqrt(abxb0b2 / a_m_b2))
         + cos_t * dist * sin_a * (2 * A[0] - 2 * B[0])
               * (ABXB0B[0] * AB[1] - ABXB0B[1] * AB[0])
               / (a_m_b2 * a_m_b2 * std::sqrt(abxb0b2 / a_m_b2))
         + dist * sin_a * sin_t * (A[1] - B0[1])
               / (a_m_b * std::sqrt(abxb0b2 / a_m_b2)));
    dW.dB(1, 0) =
        (ABXB0B[0] * AB[1] * dist * sin_a * sin_t
             / (a_m_b2 * a_m_b * std::sqrt(abxb0b2 / a_m_b2))
         + ABXB0B[0] * a_m_b * dist * sin_a * sin_t
               * (-(ABXB0B[0] * (2 * A[2] - 2 * B0[2])
                    + ABXB0B[2] * (-2 * A[0] + 2 * B0[0]))
                      / (2 * a_m_b2)
                  - abxb0b2 * (2 * A[1] - 2 * B[1]) / (2 * a_m_b2 * a_m_b2))
               / (abxb0b2 * std::sqrt(abxb0b2 / a_m_b2))
         + AB[0] * AB[1] * cos_a * dist / (a_m_b * a_m_b2)
         - cos_t * dist * sin_a * (-ABXB0B[1] * AB[2] + ABXB0B[2] * AB[1])
               * (-(ABXB0B[0] * (2 * A[2] - 2 * B0[2])
                    + ABXB0B[2] * (-2 * A[0] + 2 * B0[0]))
                      / (2 * a_m_b2)
                  - abxb0b2 * (2 * A[1] - 2 * B[1]) / (2 * a_m_b2 * a_m_b2))
               / (abxb0b2 * std::sqrt(abxb0b2 / a_m_b2))
         - cos_t * dist * sin_a * (-ABXB0B[2] + AB[1] * (-A[0] + B0[0]))
               / (a_m_b2 * std::sqrt(abxb0b2 / a_m_b2))
         - cos_t * dist * sin_a * (2 * A[1] - 2 * B[1])
               * (-ABXB0B[1] * AB[2] + ABXB0B[2] * AB[1])
               / (a_m_b2 * a_m_b2 * std::sqrt(abxb0b2 / a_m_b2))
         + dist * sin_a * sin_t * (A[2] - B0[2])
               / (a_m_b * std::sqrt(abxb0b2 / a_m_b2)));
    dW.dB(1, 1) =
        (ABXB0B[1] * AB[1] * dist * sin_a * sin_t
             / (a_m_b2 * a_m_b * std::sqrt(abxb0b2 / a_m_b2))
         + ABXB0B[1] * a_m_b * dist * sin_a * sin_t
               * (-(ABXB0B[0] * (2 * A[2] - 2 * B0[2])
                    + ABXB0B[2] * (-2 * A[0] + 2 * B0[0]))
                      / (2 * a_m_b2)
                  - abxb0b2 * (2 * A[1] - 2 * B[1]) / (2 * a_m_b2 * a_m_b2))
               / (abxb0b2 * std::sqrt(abxb0b2 / a_m_b2))
         + AB[1] * AB[1] * cos_a * dist / (a_m_b * a_m_b2)
         + cos_t * dist * sin_a * (-ABXB0B[0] * AB[2] + ABXB0B[2] * AB[0])
               * (-(ABXB0B[0] * (2 * A[2] - 2 * B0[2])
                    + ABXB0B[2] * (-2 * A[0] + 2 * B0[0]))
                      / (2 * a_m_b2)
                  - abxb0b2 * (2 * A[1] - 2 * B[1]) / (2 * a_m_b2 * a_m_b2))
               / (abxb0b2 * std::sqrt(abxb0b2 / a_m_b2))
         + cos_t * dist * sin_a
               * (AB[0] * (-A[0] + B0[0]) - AB[2] * (A[2] - B0[2]))
               / (a_m_b2 * std::sqrt(abxb0b2 / a_m_b2))
         + cos_t * dist * sin_a * (2 * A[1] - 2 * B[1])
               * (-ABXB0B[0] * AB[2] + ABXB0B[2] * AB[0])
               / (a_m_b2 * a_m_b2 * std::sqrt(abxb0b2 / a_m_b2))
         - cos_a * dist / a_m_b);
    dW.dB(1, 2) =
        (ABXB0B[2] * AB[1] * dist * sin_a * sin_t
             / (a_m_b2 * a_m_b * std::sqrt(abxb0b2 / a_m_b2))
         + ABXB0B[2] * a_m_b * dist * sin_a * sin_t
               * (-(ABXB0B[0] * (2 * A[2] - 2 * B0[2])
                    + ABXB0B[2] * (-2 * A[0] + 2 * B0[0]))
                      / (2 * a_m_b2)
                  - abxb0b2 * (2 * A[1] - 2 * B[1]) / (2 * a_m_b2 * a_m_b2))
               / (abxb0b2 * std::sqrt(abxb0b2 / a_m_b2))
         + AB[1] * AB[2] * cos_a * dist / (a_m_b * a_m_b2)
         + cos_t * dist * sin_a * (ABXB0B[0] * AB[1] - ABXB0B[1] * AB[0])
               * (-(ABXB0B[0] * (2 * A[2] - 2 * B0[2])
                    + ABXB0B[2] * (-2 * A[0] + 2 * B0[0]))
                      / (2 * a_m_b2)
                  - abxb0b2 * (2 * A[1] - 2 * B[1]) / (2 * a_m_b2 * a_m_b2))
               / (abxb0b2 * std::sqrt(abxb0b2 / a_m_b2))
         + cos_t * dist * sin_a * (-ABXB0B[0] + AB[1] * (A[2] - B0[2]))
               / (a_m_b2 * std::sqrt(abxb0b2 / a_m_b2))
         + cos_t * dist * sin_a * (2 * A[1] - 2 * B[1])
               * (ABXB0B[0] * AB[1] - ABXB0B[1] * AB[0])
               / (a_m_b2 * a_m_b2 * std::sqrt(abxb0b2 / a_m_b2))
         + dist * sin_a * sin_t * (-A[0] + B0[0])
               / (a_m_b * std::sqrt(abxb0b2 / a_m_b2)));
    dW.dB(2, 0) =
        (ABXB0B[0] * AB[2] * dist * sin_a * sin_t
             / (a_m_b2 * a_m_b * std::sqrt(abxb0b2 / a_m_b2))
         + ABXB0B[0] * a_m_b * dist * sin_a * sin_t
               * (-(ABXB0B[0] * (-2 * A[1] + 2 * B0[1])
                    - ABXB0B[1] * (-2 * A[0] + 2 * B0[0]))
                      / (2 * a_m_b2)
                  - abxb0b2 * (2 * A[2] - 2 * B[2]) / (2 * a_m_b2 * a_m_b2))
               / (abxb0b2 * std::sqrt(abxb0b2 / a_m_b2))
         + AB[0] * AB[2] * cos_a * dist / (a_m_b * a_m_b2)
         - cos_t * dist * sin_a * (-ABXB0B[1] * AB[2] + ABXB0B[2] * AB[1])
               * (-(ABXB0B[0] * (-2 * A[1] + 2 * B0[1])
                    - ABXB0B[1] * (-2 * A[0] + 2 * B0[0]))
                      / (2 * a_m_b2)
                  - abxb0b2 * (2 * A[2] - 2 * B[2]) / (2 * a_m_b2 * a_m_b2))
               / (abxb0b2 * std::sqrt(abxb0b2 / a_m_b2))
         - cos_t * dist * sin_a * (ABXB0B[1] + AB[2] * (-A[0] + B0[0]))
               / (a_m_b2 * std::sqrt(abxb0b2 / a_m_b2))
         - cos_t * dist * sin_a * (2 * A[2] - 2 * B[2])
               * (-ABXB0B[1] * AB[2] + ABXB0B[2] * AB[1])
               / (a_m_b2 * a_m_b2 * std::sqrt(abxb0b2 / a_m_b2))
         + dist * sin_a * sin_t * (-A[1] + B0[1])
               / (a_m_b * std::sqrt(abxb0b2 / a_m_b2)));
    dW.dB(2, 1) =
        (ABXB0B[1] * AB[2] * dist * sin_a * sin_t
             / (a_m_b2 * a_m_b * std::sqrt(abxb0b2 / a_m_b2))
         + ABXB0B[1] * a_m_b * dist * sin_a * sin_t
               * (-(ABXB0B[0] * (-2 * A[1] + 2 * B0[1])
                    - ABXB0B[1] * (-2 * A[0] + 2 * B0[0]))
                      / (2 * a_m_b2)
                  - abxb0b2 * (2 * A[2] - 2 * B[2]) / (2 * a_m_b2 * a_m_b2))
               / (abxb0b2 * std::sqrt(abxb0b2 / a_m_b2))
         + AB[1] * AB[2] * cos_a * dist / (a_m_b * a_m_b2)
         + cos_t * dist * sin_a * (-ABXB0B[0] * AB[2] + ABXB0B[2] * AB[0])
               * (-(ABXB0B[0] * (-2 * A[1] + 2 * B0[1])
                    - ABXB0B[1] * (-2 * A[0] + 2 * B0[0]))
                      / (2 * a_m_b2)
                  - abxb0b2 * (2 * A[2] - 2 * B[2]) / (2 * a_m_b2 * a_m_b2))
               / (abxb0b2 * std::sqrt(abxb0b2 / a_m_b2))
         + cos_t * dist * sin_a * (ABXB0B[0] - AB[2] * (-A[1] + B0[1]))
               / (a_m_b2 * std::sqrt(abxb0b2 / a_m_b2))
         + cos_t * dist * sin_a * (2 * A[2] - 2 * B[2])
               * (-ABXB0B[0] * AB[2] + ABXB0B[2] * AB[0])
               / (a_m_b2 * a_m_b2 * std::sqrt(abxb0b2 / a_m_b2))
         + dist * sin_a * sin_t * (A[0] - B0[0])
               / (a_m_b * std::sqrt(abxb0b2 / a_m_b2)));
    dW.dB(2, 2) =
        (ABXB0B[2] * AB[2] * dist * sin_a * sin_t
             / (a_m_b2 * a_m_b * std::sqrt(abxb0b2 / a_m_b2))
         + ABXB0B[2] * a_m_b * dist * sin_a * sin_t
               * (-(ABXB0B[0] * (-2 * A[1] + 2 * B0[1])
                    - ABXB0B[1] * (-2 * A[0] + 2 * B0[0]))
                      / (2 * a_m_b2)
                  - abxb0b2 * (2 * A[2] - 2 * B[2]) / (2 * a_m_b2 * a_m_b2))
               / (abxb0b2 * std::sqrt(abxb0b2 / a_m_b2))
         + AB[2] * AB[2] * cos_a * dist / (a_m_b * a_m_b2)
         + cos_t * dist * sin_a * (ABXB0B[0] * AB[1] - ABXB0B[1] * AB[0])
               * (-(ABXB0B[0] * (-2 * A[1] + 2 * B0[1])
                    - ABXB0B[1] * (-2 * A[0] + 2 * B0[0]))
                      / (2 * a_m_b2)
                  - abxb0b2 * (2 * A[2] - 2 * B[2]) / (2 * a_m_b2 * a_m_b2))
               / (abxb0b2 * std::sqrt(abxb0b2 / a_m_b2))
         + cos_t * dist * sin_a
               * (AB[0] * (-A[0] + B0[0]) + AB[1] * (-A[1] + B0[1]))
               / (a_m_b2 * std::sqrt(abxb0b2 / a_m_b2))
         + cos_t * dist * sin_a * (2 * A[2] - 2 * B[2])
               * (ABXB0B[0] * AB[1] - ABXB0B[1] * AB[0])
               / (a_m_b2 * a_m_b2 * std::sqrt(abxb0b2 / a_m_b2))
         - cos_a * dist / a_m_b);

    dW.dB0(0, 0) =
        (-ABXB0B[0] * dist * sin_a * sin_t
             * (-ABXB0B[1] * (-2 * A[2] + 2 * B[2])
                + ABXB0B[2] * (-2 * A[1] + 2 * B[1]))
             / (2 * a_m_b * abxb0b2 * std::sqrt(abxb0b2 / a_m_b2))
         - cos_t * dist * sin_a * (-AB[1] * AB[1] - AB[2] * AB[2])
               / (a_m_b2 * std::sqrt(abxb0b2 / a_m_b2))
         + cos_t * dist * sin_a * (-ABXB0B[1] * AB[2] + ABXB0B[2] * AB[1])
               * (-ABXB0B[1] * (-2 * A[2] + 2 * B[2])
                  + ABXB0B[2] * (-2 * A[1] + 2 * B[1]))
               / (2 * a_m_b2 * abxb0b2 * std::sqrt(abxb0b2 / a_m_b2)));
    dW.dB0(0, 1) =
        (-ABXB0B[1] * dist * sin_a * sin_t
             * (-ABXB0B[1] * (-2 * A[2] + 2 * B[2])
                + ABXB0B[2] * (-2 * A[1] + 2 * B[1]))
             / (2 * a_m_b * abxb0b2 * std::sqrt(abxb0b2 / a_m_b2))
         - AB[0] * AB[1] * cos_t * dist * sin_a
               / (a_m_b2 * std::sqrt(abxb0b2 / a_m_b2))
         + AB[2] * dist * sin_a * sin_t / (a_m_b * std::sqrt(abxb0b2 / a_m_b2))
         - cos_t * dist * sin_a * (-ABXB0B[0] * AB[2] + ABXB0B[2] * AB[0])
               * (-ABXB0B[1] * (-2 * A[2] + 2 * B[2])
                  + ABXB0B[2] * (-2 * A[1] + 2 * B[1]))
               / (2 * a_m_b2 * abxb0b2 * std::sqrt(abxb0b2 / a_m_b2)));
    dW.dB0(0, 2) =
        (-ABXB0B[2] * dist * sin_a * sin_t
             * (-ABXB0B[1] * (-2 * A[2] + 2 * B[2])
                + ABXB0B[2] * (-2 * A[1] + 2 * B[1]))
             / (2 * a_m_b * abxb0b2 * std::sqrt(abxb0b2 / a_m_b2))
         - AB[0] * AB[2] * cos_t * dist * sin_a
               / (a_m_b2 * std::sqrt(abxb0b2 / a_m_b2))
         - AB[1] * dist * sin_a * sin_t / (a_m_b * std::sqrt(abxb0b2 / a_m_b2))
         - cos_t * dist * sin_a * (ABXB0B[0] * AB[1] - ABXB0B[1] * AB[0])
               * (-ABXB0B[1] * (-2 * A[2] + 2 * B[2])
                  + ABXB0B[2] * (-2 * A[1] + 2 * B[1]))
               / (2 * a_m_b2 * abxb0b2 * std::sqrt(abxb0b2 / a_m_b2)));
    dW.dB0(1, 0) =
        (-ABXB0B[0] * dist * sin_a * sin_t
             * (ABXB0B[0] * (-2 * A[2] + 2 * B[2])
                + ABXB0B[2] * (2 * A[0] - 2 * B[0]))
             / (2 * a_m_b * abxb0b2 * std::sqrt(abxb0b2 / a_m_b2))
         - AB[0] * AB[1] * cos_t * dist * sin_a
               / (a_m_b2 * std::sqrt(abxb0b2 / a_m_b2))
         - AB[2] * dist * sin_a * sin_t / (a_m_b * std::sqrt(abxb0b2 / a_m_b2))
         + cos_t * dist * sin_a
               * (ABXB0B[0] * (-2 * A[2] + 2 * B[2])
                  + ABXB0B[2] * (2 * A[0] - 2 * B[0]))
               * (-ABXB0B[1] * AB[2] + ABXB0B[2] * AB[1])
               / (2 * a_m_b2 * abxb0b2 * std::sqrt(abxb0b2 / a_m_b2)));
    dW.dB0(1, 1) =
        (-ABXB0B[1] * dist * sin_a * sin_t
             * (ABXB0B[0] * (-2 * A[2] + 2 * B[2])
                + ABXB0B[2] * (2 * A[0] - 2 * B[0]))
             / (2 * a_m_b * abxb0b2 * std::sqrt(abxb0b2 / a_m_b2))
         + cos_t * dist * sin_a * (AB[0] * AB[0] + AB[2] * AB[2])
               / (a_m_b2 * std::sqrt(abxb0b2 / a_m_b2))
         - cos_t * dist * sin_a * (-ABXB0B[0] * AB[2] + ABXB0B[2] * AB[0])
               * (ABXB0B[0] * (-2 * A[2] + 2 * B[2])
                  + ABXB0B[2] * (2 * A[0] - 2 * B[0]))
               / (2 * a_m_b2 * abxb0b2 * std::sqrt(abxb0b2 / a_m_b2)));
    dW.dB0(1, 2) =
        (-ABXB0B[2] * dist * sin_a * sin_t
             * (ABXB0B[0] * (-2 * A[2] + 2 * B[2])
                + ABXB0B[2] * (2 * A[0] - 2 * B[0]))
             / (2 * a_m_b * abxb0b2 * std::sqrt(abxb0b2 / a_m_b2))
         + AB[0] * dist * sin_a * sin_t / (a_m_b * std::sqrt(abxb0b2 / a_m_b2))
         - AB[1] * AB[2] * cos_t * dist * sin_a
               / (a_m_b2 * std::sqrt(abxb0b2 / a_m_b2))
         - cos_t * dist * sin_a * (ABXB0B[0] * AB[1] - ABXB0B[1] * AB[0])
               * (ABXB0B[0] * (-2 * A[2] + 2 * B[2])
                  + ABXB0B[2] * (2 * A[0] - 2 * B[0]))
               / (2 * a_m_b2 * abxb0b2 * std::sqrt(abxb0b2 / a_m_b2)));
    dW.dB0(2, 0) =
        (-ABXB0B[0] * dist * sin_a * sin_t
             * (ABXB0B[0] * (2 * A[1] - 2 * B[1])
                - ABXB0B[1] * (2 * A[0] - 2 * B[0]))
             / (2 * a_m_b * abxb0b2 * std::sqrt(abxb0b2 / a_m_b2))
         - AB[0] * AB[2] * cos_t * dist * sin_a
               / (a_m_b2 * std::sqrt(abxb0b2 / a_m_b2))
         + AB[1] * dist * sin_a * sin_t / (a_m_b * std::sqrt(abxb0b2 / a_m_b2))
         + cos_t * dist * sin_a
               * (ABXB0B[0] * (2 * A[1] - 2 * B[1])
                  - ABXB0B[1] * (2 * A[0] - 2 * B[0]))
               * (-ABXB0B[1] * AB[2] + ABXB0B[2] * AB[1])
               / (2 * a_m_b2 * abxb0b2 * std::sqrt(abxb0b2 / a_m_b2)));
    dW.dB0(2, 1) =
        (-ABXB0B[1] * dist * sin_a * sin_t
             * (ABXB0B[0] * (2 * A[1] - 2 * B[1])
                - ABXB0B[1] * (2 * A[0] - 2 * B[0]))
             / (2 * a_m_b * abxb0b2 * std::sqrt(abxb0b2 / a_m_b2))
         - AB[0] * dist * sin_a * sin_t / (a_m_b * std::sqrt(abxb0b2 / a_m_b2))
         - AB[1] * AB[2] * cos_t * dist * sin_a
               / (a_m_b2 * std::sqrt(abxb0b2 / a_m_b2))
         - cos_t * dist * sin_a * (-ABXB0B[0] * AB[2] + ABXB0B[2] * AB[0])
               * (ABXB0B[0] * (2 * A[1] - 2 * B[1])
                  - ABXB0B[1] * (2 * A[0] - 2 * B[0]))
               / (2 * a_m_b2 * abxb0b2 * std::sqrt(abxb0b2 / a_m_b2)));
    dW.dB0(2, 2) =
        (-ABXB0B[2] * dist * sin_a * sin_t
             * (ABXB0B[0] * (2 * A[1] - 2 * B[1])
                - ABXB0B[1] * (2 * A[0] - 2 * B[0]))
             / (2 * a_m_b * abxb0b2 * std::sqrt(abxb0b2 / a_m_b2))
         + cos_t * dist * sin_a * (AB[0] * AB[0] + AB[1] * AB[1])
               / (a_m_b2 * std::sqrt(abxb0b2 / a_m_b2))
         - cos_t * dist * sin_a * (ABXB0B[0] * AB[1] - ABXB0B[1] * AB[0])
               * (ABXB0B[0] * (2 * A[1] - 2 * B[1])
                  - ABXB0B[1] * (2 * A[0] - 2 * B[0]))
               / (2 * a_m_b2 * abxb0b2 * std::sqrt(abxb0b2 / a_m_b2)));

    return dW;
  }
};

#undef def

template <typename Real>
class WaterGenSingleResData {
 public:
  int block_ind;
  int block_type;
  int block_coord_offset;
  int n_atoms;
  int n_conn;
  Real *coords;
  unsigned char n_donH;
  unsigned char n_acc;
  unsigned char *donH_tile_inds;
  unsigned char *don_hvy_inds;  // index of heavy atom that given donH bonds;
                                // NOTE: limit of 256 atoms per block.
  unsigned char *which_donH_for_hvy;  // for the given donH, what's its index in
                                      // the list of H's bound to it
  unsigned char *acc_tile_inds;
  // unsigned char *donH_type;
  // unsigned char *acc_type;
  unsigned char *acc_hybridization;
  unsigned char *acc_n_attached_H;  // for the given acc, how many H's it have?
};

template <tmol::Device Dev, typename Real, typename Int>
class WaterGenPoseContextData {
 public:
  int pose_ind;
  LKBallWaterGenGlobalParams<Real> global_params;

  // Tensors to help identify atoms outside of the target
  // block
  // If the hbond involves atoms from other residues, we need
  // to be able to retrieve their coordinates
  TView<Vec<Real, 3>, 2, Dev> coords;
  TView<Int, 2, Dev> pose_stack_block_coord_offset;
  TView<Int, 2, Dev> pose_stack_block_type;

  // For determining which atoms to retrieve from neighboring
  // residues we have to know how the blocks in the Pose
  // are connected
  TView<Vec<Int, 2>, 3, Dev> pose_stack_inter_residue_connections;

  // And we need to know the properties of the block types
  // that we are working with to iterate across chemical bonds
  TView<Int, 1, Dev> block_type_n_all_bonds;
  TView<Vec<Int, 3>, 2, Dev> block_type_all_bonds;
  TView<Vec<Int, 2>, 2, Dev> block_type_atom_all_bond_ranges;
  TView<Int, 2, Dev> block_type_atoms_forming_chemical_bonds;
  TView<Int, 2, Dev> block_type_atom_is_hydrogen;
};

template <tmol::Device Dev, typename Real, typename Int>
class WaterGenData {
 public:
  WaterGenSingleResData<Real> r_dat;
  WaterGenPoseContextData<Dev, Real, Int> pose_context;
};

template <typename Real, int TILE_SIZE>
struct WaterGenSharedData {
  Real coords[TILE_SIZE * 3];  // 384 bytes for coords
  unsigned char n_donH;        // 4 bytes for counts
  unsigned char n_acc;
  unsigned char don_inds[TILE_SIZE];  //
  unsigned char don_hvy_inds[TILE_SIZE];
  unsigned char which_donH_for_hvy[TILE_SIZE];
  unsigned char acc_inds[TILE_SIZE];
  unsigned char acc_hybridization[TILE_SIZE];
  unsigned char acc_n_attached_H[TILE_SIZE];
};

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    typename Real,
    typename Int>
void TMOL_DEVICE_FUNC water_gen_load_block_coords_and_params_into_shared(
    TView<Vec<Real, 3>, 2, Dev> coords,
    TView<Int, 2, Dev> block_type_tile_n_donH,
    TView<Int, 2, Dev> block_type_tile_n_acc,
    TView<Int, 3, Dev> block_type_tile_donH_inds,
    TView<Int, 3, Dev> block_type_tile_don_hvy_inds,
    TView<Int, 3, Dev> block_type_tile_which_donH_for_hhvy,
    TView<Int, 3, Dev> block_type_tile_acc_inds,
    TView<Int, 3, Dev> block_type_tile_hybridization,
    TView<Int, 3, Dev> block_type_tile_acc_n_attached_H,
    int pose_ind,
    int tile_ind,
    WaterGenSingleResData<Real> &r_dat,
    int n_atoms_to_load,
    int start_atom) {
  // pre-condition: n_atoms_to_load < TILE_SIZE
  // Note that TILE_SIZE is not explicitly passed in, but is "present"
  // in r_dat.coords allocation

  r_dat.n_donH = block_type_tile_n_donH[r_dat.block_type][tile_ind];
  r_dat.n_acc = block_type_tile_n_acc[r_dat.block_type][tile_ind];

  DeviceDispatch<Dev>::template copy_contiguous_data<nt, 3>(
      r_dat.coords,
      reinterpret_cast<Real *>(
          &coords[pose_ind][r_dat.block_coord_offset + start_atom]),
      n_atoms_to_load * 3);
  DeviceDispatch<Dev>::template copy_contiguous_data_and_cast<nt, 1>(
      r_dat.donH_tile_inds,
      &block_type_tile_donH_inds[r_dat.block_type][tile_ind][0],
      r_dat.n_donH);
  DeviceDispatch<Dev>::template copy_contiguous_data_and_cast<nt, 1>(
      r_dat.don_hvy_inds,
      &block_type_tile_don_hvy_inds[r_dat.block_type][tile_ind][0],
      r_dat.n_donH);
  DeviceDispatch<Dev>::template copy_contiguous_data_and_cast<nt, 1>(
      r_dat.which_donH_for_hvy,
      &block_type_tile_which_donH_for_hhvy[r_dat.block_type][tile_ind][0],
      r_dat.n_donH);

  DeviceDispatch<Dev>::template copy_contiguous_data_and_cast<nt, 1>(
      r_dat.acc_tile_inds,
      &block_type_tile_acc_inds[r_dat.block_type][tile_ind][0],
      r_dat.n_acc);
  // DeviceDispatch<Dev>::template copy_contiguous_data_and_cast<nt, 1>(
  //     r_dat.donH_type,
  //     &block_type_tile_donor_type[r_dat.block_type][tile_ind][0],
  //     r_dat.n_donH);
  // DeviceDispatch<Dev>::template copy_contiguous_data_and_cast<nt, 1>(
  //     r_dat.acc_type,
  //     &block_type_tile_acceptor_type[r_dat.block_type][tile_ind][0],
  //     r_dat.n_acc);
  DeviceDispatch<Dev>::template copy_contiguous_data_and_cast<nt, 1>(
      r_dat.acc_hybridization,
      &block_type_tile_hybridization[r_dat.block_type][tile_ind][0],
      r_dat.n_acc);
  DeviceDispatch<Dev>::template copy_contiguous_data_and_cast<nt, 1>(
      r_dat.acc_n_attached_H,
      &block_type_tile_acc_n_attached_H[r_dat.block_type][tile_ind][0],
      r_dat.n_acc);
}

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    typename Int,
    typename Real,
    int TILE_SIZE>
void TMOL_DEVICE_FUNC water_gen_load_tile_invariant_data(
    TView<Vec<Real, 3>, 2, Dev> coords,
    TView<Int, 2, Dev> pose_stack_block_coord_offset,
    TView<Int, 2, Dev> pose_stack_block_type,
    TView<Vec<Int, 2>, 3, Dev> pose_stack_inter_residue_connections,

    TView<Int, 1, Dev> block_type_n_all_bonds,
    TView<Vec<Int, 3>, 2, Dev> block_type_all_bonds,
    TView<Vec<Int, 2>, 2, Dev> block_type_atom_all_bond_ranges,
    TView<Int, 1, Dev> block_type_n_interblock_bonds,
    TView<Int, 2, Dev> block_type_atoms_forming_chemical_bonds,
    TView<Int, 2, Dev> block_type_atom_is_hydrogen,

    int pose_ind,
    int block_ind,
    int block_type,
    int n_atoms,

    WaterGenData<Dev, Real, Int> &water_gen_dat,
    WaterGenSharedData<Real, TILE_SIZE> &shared_m) {
  water_gen_dat.pose_context.pose_ind = pose_ind;
  water_gen_dat.r_dat.block_ind = block_ind;
  water_gen_dat.r_dat.block_type = block_type;
  water_gen_dat.r_dat.block_coord_offset =
      pose_stack_block_coord_offset[pose_ind][block_ind];
  water_gen_dat.r_dat.n_atoms = n_atoms;
  water_gen_dat.r_dat.n_conn = block_type_n_interblock_bonds[block_type];

  // set the pointers in inter_dat to point at the shared-memory arrays
  water_gen_dat.r_dat.coords = shared_m.coords;
  water_gen_dat.r_dat.donH_tile_inds = shared_m.don_inds;
  water_gen_dat.r_dat.which_donH_for_hvy = shared_m.which_donH_for_hvy;
  water_gen_dat.r_dat.acc_tile_inds = shared_m.acc_inds;
  water_gen_dat.r_dat.acc_hybridization = shared_m.acc_hybridization;
  water_gen_dat.r_dat.acc_n_attached_H = shared_m.acc_n_attached_H;

  // Final data members
  // Keep a "copy" of the tensors needed to traverse bonds during
  // the water-building step so they can be passed in to the lower
  // -level functions when needed; nvcc is smart enough not to
  // duplicate the registers used here
  water_gen_dat.pose_context.coords = coords;
  water_gen_dat.pose_context.pose_stack_block_coord_offset =
      pose_stack_block_coord_offset;
  water_gen_dat.pose_context.pose_stack_block_type = pose_stack_block_type;
  water_gen_dat.pose_context.pose_stack_inter_residue_connections =
      pose_stack_inter_residue_connections;
  water_gen_dat.pose_context.block_type_n_all_bonds = block_type_n_all_bonds;
  water_gen_dat.pose_context.block_type_all_bonds = block_type_all_bonds;
  water_gen_dat.pose_context.block_type_atom_all_bond_ranges =
      block_type_atom_all_bond_ranges;
  water_gen_dat.pose_context.block_type_atoms_forming_chemical_bonds =
      block_type_atoms_forming_chemical_bonds;
  water_gen_dat.pose_context.block_type_atom_is_hydrogen =
      block_type_atom_is_hydrogen;
}

// Some coordinates are available in shared memory, some we will
// have to go out to global memory for.
template <int TILE_SIZE, typename Real, typename Int, tmol::Device Dev>
TMOL_DEVICE_FUNC Eigen::Matrix<Real, 3, 1> load_coord(
    bonded_atom::BlockCentricAtom<Int> bcat,
    WaterGenSingleResData<Real> const &single_res_dat,
    WaterGenPoseContextData<Dev, Real, Int> const &context_dat,
    int tile_start) {
  Eigen::Matrix<Real, 3, 1> xyz{Real(0), Real(0), Real(0)};
  if (bcat.atom != -1) {
    bool in_smem = false;
    if (bcat.block == single_res_dat.block_ind) {
      int bcat_tile_ind = bcat.atom - tile_start;
      if (bcat_tile_ind >= 0 && bcat_tile_ind < TILE_SIZE) {
        in_smem = true;
        xyz = common::coord_from_shared(single_res_dat.coords, bcat_tile_ind);
      }
    }
    if (!in_smem) {
      // outside of tile or on other res, retrieve from global coords
      int coord_offset =
          (bcat.block == single_res_dat.block_ind
               ? single_res_dat.block_coord_offset
               : context_dat.pose_stack_block_coord_offset[context_dat.pose_ind]
                                                          [bcat.block]);
      xyz = context_dat.coords[context_dat.pose_ind][bcat.atom + coord_offset];
    }
  }
  return xyz;
}

template <int TILE_SIZE, tmol::Device Dev, typename Real, typename Int>
void TMOL_DEVICE_FUNC build_water_for_don(
    TView<Vec<Real, 3>, 3, Dev> water_coords,
    WaterGenData<Dev, Real, Int> wat_gen_dat,
    int tile_start,
    int don_h_ind  // [0..n_donH)
) {
  using Real3 = Vec<Real, 3>;

  auto res_dat = wat_gen_dat.r_dat;
  auto context_dat = wat_gen_dat.pose_context;
  int const don_h_atom_tile_ind = res_dat.donH_tile_inds[don_h_ind];
  Real3 Hxyz = common::coord_from_shared(res_dat.coords, don_h_atom_tile_ind);
  int const Dind = res_dat.don_hvy_inds[don_h_ind];
  bonded_atom::BlockCentricAtom<Int> const D{
      res_dat.block_ind, res_dat.block_type, Dind};

  Real3 Dxyz = load_coord<TILE_SIZE>(D, res_dat, context_dat, tile_start);

  auto Wxyz = build_don_water<Real>::V(
      Dxyz, Hxyz, context_dat.global_params.lkb_water_dist);

  // Now record the coordinates to global memory:
  int const which_water = res_dat.which_donH_for_hvy[don_h_ind];

  water_coords[context_dat.pose_ind][res_dat.block_coord_offset + Dind]
              [which_water] = Wxyz;
}

template <int TILE_SIZE, tmol::Device Dev, typename Real, typename Int>
void TMOL_DEVICE_FUNC build_water_for_acc(
    TView<Real, 1, Dev> sp2_water_tors,
    TView<Real, 1, Dev> sp3_water_tors,
    TView<Real, 1, Dev> ring_water_tors,
    TView<Vec<Real, 3>, 3, Dev> water_coords,
    WaterGenData<Dev, Real, Int> wat_gen_dat,
    int tile_start,
    int acc_ind,   // [0..n_acc)
    int water_ind  // [0..MAX_N_WATER)
) {
  using Real3 = Vec<Real, 3>;

  auto res_dat = wat_gen_dat.r_dat;
  auto context_dat = wat_gen_dat.pose_context;

  unsigned char hyb = res_dat.acc_hybridization[acc_ind];
  Real tor(0), ang(0);
  if (hyb == hbond::AcceptorHybridization::sp2) {
    if (water_ind >= sp2_water_tors.size(0)) {
      return;
    } else {
      tor = sp2_water_tors[water_ind];
      ang = context_dat.global_params.lkb_water_angle_sp2;
    }
  } else if (hyb == hbond::AcceptorHybridization::sp3) {
    if (water_ind >= sp3_water_tors.size(0)) {
      return;
    } else {
      tor = sp3_water_tors[water_ind];
      ang = context_dat.global_params.lkb_water_angle_sp3;
    }
  } else if (hyb == hbond::AcceptorHybridization::ring) {
    if (water_ind >= ring_water_tors.size(0)) {
      return;
    } else {
      tor = ring_water_tors[water_ind];
      ang = context_dat.global_params.lkb_water_angle_ring;
    }
  }

  unsigned char acc_atom_tile_ind = res_dat.acc_tile_inds[acc_ind];

  Real3 Axyz = common::coord_from_shared(res_dat.coords, acc_atom_tile_ind);
  bonded_atom::BlockCentricIndexedBonds<Int, Dev> bonds{
      context_dat.pose_stack_inter_residue_connections[context_dat.pose_ind],
      context_dat.pose_stack_block_type[context_dat.pose_ind],
      context_dat.block_type_n_all_bonds,
      context_dat.block_type_all_bonds,
      context_dat.block_type_atom_all_bond_ranges,
      context_dat.block_type_atoms_forming_chemical_bonds};
  bonded_atom::BlockCentricAtom<Int> A{
      res_dat.block_ind, res_dat.block_type, tile_start + acc_atom_tile_ind};
  auto acc_bases = hbond::BlockCentricAcceptorBases<Int>::for_acceptor(
      A, hyb, bonds, context_dat.block_type_atom_is_hydrogen);

  Real3 Bxyz =
      load_coord<TILE_SIZE>(acc_bases.B, res_dat, context_dat, tile_start);
  Real3 B0xyz =
      load_coord<TILE_SIZE>(acc_bases.B0, res_dat, context_dat, tile_start);

  if (hyb == hbond::AcceptorHybridization::ring) {
    // take the bisector of the line between B and B0 to build from
    Bxyz = (Bxyz + B0xyz) / 2;
  }

  auto Wxyz = build_acc_water<Real>::V(
      Axyz, Bxyz, B0xyz, context_dat.global_params.lkb_water_dist, ang, tor);

  // Now record the coordinates to global memory:
  // offset the water by the number of polar hydrogens on
  // this acceptor
  unsigned char water_offset = res_dat.acc_n_attached_H[acc_ind];

  water_coords[context_dat.pose_ind][res_dat.block_coord_offset + A.atom]
              [water_ind + water_offset] = Wxyz;
}

template <int TILE_SIZE, tmol::Device Dev, typename Real, typename Int>
void TMOL_DEVICE_FUNC d_build_water_for_don(
    TView<Vec<Real, 3>, 3, Dev> dE_dWxyz,
    TView<Vec<Real, 3>, 2, Dev> dE_d_pose_coords,
    WaterGenData<Dev, Real, Int> wat_gen_dat,
    int tile_start,
    int don_h_ind  // [0..n_donH)
) {
  using Real3 = Vec<Real, 3>;

  auto res_dat = wat_gen_dat.r_dat;
  auto context_dat = wat_gen_dat.pose_context;
  int const don_h_atom_tile_ind = res_dat.donH_tile_inds[don_h_ind];

  Real3 Hxyz = common::coord_from_shared(res_dat.coords, don_h_atom_tile_ind);
  int const Dind = res_dat.don_hvy_inds[don_h_ind];
  bonded_atom::BlockCentricAtom<Int> const D{
      res_dat.block_ind, res_dat.block_type, Dind};
  // int const D = res_dat.don_hvy_inds[don_h_ind];
  Real3 Dxyz = load_coord<TILE_SIZE>(D, res_dat, context_dat, tile_start);

  auto dW = build_don_water<Real>::dV(
      Dxyz, Hxyz, context_dat.global_params.lkb_water_dist);
  int const which_water = res_dat.which_donH_for_hvy[don_h_ind];

  // water_coords[context_dat.pose_ind][res_dat.block_coord_offset +
  // D][which_water] = Wxyz;
  int const pose_ind = wat_gen_dat.pose_context.pose_ind;
  int const D_atom_pose_ind = res_dat.block_coord_offset + Dind;
  int const H_atom_pose_ind =
      res_dat.block_coord_offset + tile_start + don_h_atom_tile_ind;
  Real3 dE_dW = dE_dWxyz[pose_ind][D_atom_pose_ind][which_water];

  common::accumulate<Dev, Vec<Real, 3>>::add(
      dE_d_pose_coords[pose_ind][D_atom_pose_ind], dW.dD * dE_dW);
  common::accumulate<Dev, Vec<Real, 3>>::add(
      dE_d_pose_coords[pose_ind][H_atom_pose_ind], dW.dH * dE_dW);
}

template <int TILE_SIZE, tmol::Device Dev, typename Real, typename Int>
void TMOL_DEVICE_FUNC d_build_water_for_acc(
    TView<Real, 1, Dev> sp2_water_tors,
    TView<Real, 1, Dev> sp3_water_tors,
    TView<Real, 1, Dev> ring_water_tors,
    TView<Vec<Real, 3>, 3, Dev> dE_dWxyz,
    TView<Vec<Real, 3>, 2, Dev> dE_d_pose_coords,
    WaterGenData<Dev, Real, Int> wat_gen_dat,
    int tile_start,
    int acc_ind,   // [0..n_acc)
    int water_ind  // [0..MAX_N_WATER)
) {
  using Real3 = Vec<Real, 3>;
  auto res_dat = wat_gen_dat.r_dat;
  auto context_dat = wat_gen_dat.pose_context;

  unsigned char hyb = res_dat.acc_hybridization[acc_ind];
  Real tor(0), ang(0);
  if (hyb == hbond::AcceptorHybridization::sp2) {
    if (water_ind >= sp2_water_tors.size(0)) {
      return;
    } else {
      tor = sp2_water_tors[water_ind];
      ang = context_dat.global_params.lkb_water_angle_sp2;
    }
  } else if (hyb == hbond::AcceptorHybridization::sp3) {
    if (water_ind >= sp3_water_tors.size(0)) {
      return;
    } else {
      tor = sp3_water_tors[water_ind];
      ang = context_dat.global_params.lkb_water_angle_sp3;
    }
  } else if (hyb == hbond::AcceptorHybridization::ring) {
    if (water_ind >= ring_water_tors.size(0)) {
      return;
    } else {
      tor = ring_water_tors[water_ind];
      ang = context_dat.global_params.lkb_water_angle_ring;
    }
  }

  unsigned char acc_atom_tile_ind = res_dat.acc_tile_inds[acc_ind];

  Real3 Axyz = common::coord_from_shared(res_dat.coords, acc_atom_tile_ind);
  bonded_atom::BlockCentricIndexedBonds<Int, Dev> bonds{
      context_dat.pose_stack_inter_residue_connections[context_dat.pose_ind],
      context_dat.pose_stack_block_type[context_dat.pose_ind],
      context_dat.block_type_n_all_bonds,
      context_dat.block_type_all_bonds,
      context_dat.block_type_atom_all_bond_ranges,
      context_dat.block_type_atoms_forming_chemical_bonds};
  bonded_atom::BlockCentricAtom<Int> A{
      res_dat.block_ind, res_dat.block_type, tile_start + acc_atom_tile_ind};
  auto acc_bases = hbond::BlockCentricAcceptorBases<Int>::for_acceptor(
      A, hyb, bonds, context_dat.block_type_atom_is_hydrogen);

  Real3 Bxyz =
      load_coord<TILE_SIZE>(acc_bases.B, res_dat, context_dat, tile_start);
  Real3 B0xyz =
      load_coord<TILE_SIZE>(acc_bases.B0, res_dat, context_dat, tile_start);

  if (hyb == hbond::AcceptorHybridization::ring) {
    // take the bisector of the line between B and B0 to build from
    Bxyz = (Bxyz + B0xyz) / 2;
  }

  auto dW = build_acc_water<Real>::dV(
      Axyz, Bxyz, B0xyz, context_dat.global_params.lkb_water_dist, ang, tor);

  // Now record the coordinates to global memory:
  // offset the water by the number of polar hydrogens on
  // this acceptor
  unsigned char water_offset = res_dat.acc_n_attached_H[acc_ind];

  int const pose_ind = context_dat.pose_ind;
  int const A_atom_pose_ind = res_dat.block_coord_offset + A.atom;
  int const B_atom_pose_ind =
      (acc_bases.B.block == A.block
           ? res_dat.block_coord_offset
           : context_dat
                 .pose_stack_block_coord_offset[pose_ind][acc_bases.B.block])
      + A.atom;
  int const B0_atom_pose_ind =
      (acc_bases.B0.block == A.block
           ? res_dat.block_coord_offset
           : context_dat
                 .pose_stack_block_coord_offset[pose_ind][acc_bases.B0.block])
      + A.atom;

  Real3 dE_dW = dE_dWxyz[pose_ind][A_atom_pose_ind][water_ind + water_offset];

  common::accumulate<Dev, Vec<Real, 3>>::add(
      dE_d_pose_coords[pose_ind][A_atom_pose_ind], dW.dA * dE_dW);
  if (hyb == hbond::AcceptorHybridization::ring) {
    // Since the Bxyz coordinate for ring acceptors is the
    // bisector of the B and B0 atoms, apply half of the
    // derivative to B and half to B0
    common::accumulate<Dev, Vec<Real, 3>>::add(
        dE_d_pose_coords[pose_ind][B_atom_pose_ind], 0.5 * dW.dB * dE_dW);
    common::accumulate<Dev, Vec<Real, 3>>::add(
        dE_d_pose_coords[pose_ind][B0_atom_pose_ind], 0.5 * dW.dB * dE_dW);
  } else {
    common::accumulate<Dev, Vec<Real, 3>>::add(
        dE_d_pose_coords[pose_ind][B_atom_pose_ind], dW.dB * dE_dW);
  }
  common::accumulate<Dev, Vec<Real, 3>>::add(
      dE_d_pose_coords[pose_ind][B0_atom_pose_ind], dW.dB0 * dE_dW);
}

}  // namespace potentials
}  // namespace lk_ball
}  // namespace score
}  // namespace tmol
