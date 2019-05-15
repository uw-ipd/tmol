#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <cppitertools/range.hpp>

#include <tmol/utility/tensor/TensorPack.h>
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

}  // namespace potentials
}  // namespace lk_ball
}  // namespace score
}  // namespace tmol
