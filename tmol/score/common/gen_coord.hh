#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <cppitertools/range.hpp>

#include <tmol/score/common/tuple.hh>
#include <tmol/utility/tensor/TensorPack.h>

namespace tmol {
namespace score {
namespace common {

#define def auto EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <typename Real>
struct build_coordinate {
  typedef Vec<Real, 3> Real3;
  typedef Eigen::Matrix<Real, 3, 3> RealMat;

  struct dV_t {
    RealMat dp;    // parent
    RealMat dgp;   // grand-parent
    RealMat dggp;  // great-grand parent

    def astuple() { return tmol::score::common::make_tuple(dp, dgp, dggp); }

    static def Zero() -> dV_t {
      return {RealMat::Zero(), RealMat::Zero(), RealMat::Zero()};
    }
  };

  // A = parent
  // B = grand parent
  // C = great-grant parent
  static def V(Real3 A, Real3 B, Real3 C, Real dist, Real angle, Real torsion)
      -> Real3 {
    const Real pi = EIGEN_PI;

    // Generate orientation frame
    Eigen::Matrix<Real, 3, 3> M;
    M.col(0) = (A - B).normalized();
    M.col(1) = (C - B);
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
            * Real3(
                {dist * std::cos(pi - angle),
                 dist * std::sin(pi - angle) * std::cos(torsion),
                 dist * std::sin(pi - angle) * std::sin(torsion)})
        + A);
  }

  // A = parent
  // B = grand parent
  // C = great-grant parent
  static def dV(Real3 A, Real3 B, Real3 C, Real dist, Real angle, Real torsion)
      -> dV_t {
    const Real pi = EIGEN_PI;

    Real sin_a = std::sin(pi - angle);
    Real cos_a = std::cos(pi - angle);
    Real sin_t = std::sin(torsion);
    Real cos_t = std::cos(torsion);
    Real3 AB = A - B;
    Real3 CB = C - B;
    Real a_m_b = AB.norm();
    Real a_m_b2 = a_m_b * a_m_b;
    Real c_m_b = CB.norm();
    Real3 ABXCB = AB.cross(CB);
    Real abxcb2 = ABXCB.squaredNorm();

    dV_t dcoord;
    dcoord.dp(0, 0) =
        (-ABXCB[0] * AB[0] * dist * sin_a * sin_t
             / (a_m_b2 * a_m_b * std::sqrt(abxcb2 / a_m_b2))
         + ABXCB[0] * a_m_b * dist * sin_a * sin_t
               * (-(-ABXCB[1] * (2 * C[2] - 2 * B[2])
                    + ABXCB[2] * (2 * C[1] - 2 * B[1]))
                      / (2 * a_m_b2)
                  - abxcb2 * (-2 * A[0] + 2 * B[0]) / (2 * a_m_b2 * a_m_b2))
               / (abxcb2 * std::sqrt(abxcb2 / a_m_b2))
         - AB[0] * AB[0] * cos_a * dist / (a_m_b2 * a_m_b) + 1
         - cos_t * dist * sin_a * (-ABXCB[1] * AB[2] + ABXCB[2] * AB[1])
               * (-(-ABXCB[1] * (2 * C[2] - 2 * B[2])
                    + ABXCB[2] * (2 * C[1] - 2 * B[1]))
                      / (2 * a_m_b2)
                  - abxcb2 * (-2 * A[0] + 2 * B[0]) / (2 * a_m_b2 * a_m_b2))
               / (abxcb2 * std::sqrt(abxcb2 / a_m_b2))
         - cos_t * dist * sin_a * (AB[1] * CB[1] + AB[2] * CB[2])
               / (a_m_b2 * std::sqrt(abxcb2 / a_m_b2))
         - cos_t * dist * sin_a * (-2 * A[0] + 2 * B[0])
               * (-ABXCB[1] * AB[2] + ABXCB[2] * AB[1])
               / (a_m_b2 * a_m_b2 * std::sqrt(abxcb2 / a_m_b2))
         + cos_a * dist / a_m_b);
    dcoord.dp(0, 1) =
        (-ABXCB[1] * AB[0] * dist * sin_a * sin_t
             / (a_m_b2 * a_m_b * std::sqrt(abxcb2 / a_m_b2))
         + ABXCB[1] * a_m_b * dist * sin_a * sin_t
               * (-(-ABXCB[1] * (2 * C[2] - 2 * B[2])
                    + ABXCB[2] * (2 * C[1] - 2 * B[1]))
                      / (2 * a_m_b2)
                  - abxcb2 * (-2 * A[0] + 2 * B[0]) / (2 * a_m_b2 * a_m_b2))
               / (abxcb2 * std::sqrt(abxcb2 / a_m_b2))
         - AB[0] * AB[1] * cos_a * dist / (a_m_b2 * a_m_b)
         - CB[2] * dist * sin_a * sin_t / (a_m_b * std::sqrt(abxcb2 / a_m_b2))
         + cos_t * dist * sin_a * (-ABXCB[0] * AB[2] + ABXCB[2] * AB[0])
               * (-(-ABXCB[1] * (2 * C[2] - 2 * B[2])
                    + ABXCB[2] * (2 * C[1] - 2 * B[1]))
                      / (2 * a_m_b2)
                  - abxcb2 * (-2 * A[0] + 2 * B[0]) / (2 * a_m_b2 * a_m_b2))
               / (abxcb2 * std::sqrt(abxcb2 / a_m_b2))
         + cos_t * dist * sin_a * (2 * AB[0] * CB[1] - AB[1] * CB[0])
               / (a_m_b2 * std::sqrt(abxcb2 / a_m_b2))
         + cos_t * dist * sin_a * (-2 * A[0] + 2 * B[0])
               * (-ABXCB[0] * AB[2] + ABXCB[2] * AB[0])
               / (a_m_b2 * a_m_b2 * std::sqrt(abxcb2 / a_m_b2)));
    dcoord.dp(0, 2) =
        (-ABXCB[2] * AB[0] * dist * sin_a * sin_t
             / (a_m_b2 * a_m_b * std::sqrt(abxcb2 / a_m_b2))
         + ABXCB[2] * a_m_b * dist * sin_a * sin_t
               * (-(-ABXCB[1] * (2 * C[2] - 2 * B[2])
                    + ABXCB[2] * (2 * C[1] - 2 * B[1]))
                      / (2 * a_m_b2)
                  - abxcb2 * (-2 * A[0] + 2 * B[0]) / (2 * a_m_b2 * a_m_b2))
               / (abxcb2 * std::sqrt(abxcb2 / a_m_b2))
         - AB[0] * AB[2] * cos_a * dist / (a_m_b2 * a_m_b)
         + CB[1] * dist * sin_a * sin_t / (a_m_b * std::sqrt(abxcb2 / a_m_b2))
         + cos_t * dist * sin_a * (ABXCB[0] * AB[1] - ABXCB[1] * AB[0])
               * (-(-ABXCB[1] * (2 * C[2] - 2 * B[2])
                    + ABXCB[2] * (2 * C[1] - 2 * B[1]))
                      / (2 * a_m_b2)
                  - abxcb2 * (-2 * A[0] + 2 * B[0]) / (2 * a_m_b2 * a_m_b2))
               / (abxcb2 * std::sqrt(abxcb2 / a_m_b2))
         + cos_t * dist * sin_a * (2 * AB[0] * CB[2] - AB[2] * CB[0])
               / (a_m_b2 * std::sqrt(abxcb2 / a_m_b2))
         + cos_t * dist * sin_a * (-2 * A[0] + 2 * B[0])
               * (ABXCB[0] * AB[1] - ABXCB[1] * AB[0])
               / (a_m_b2 * a_m_b2 * std::sqrt(abxcb2 / a_m_b2)));
    dcoord.dp(1, 0) =
        (-ABXCB[0] * AB[1] * dist * sin_a * sin_t
             / (a_m_b2 * a_m_b * std::sqrt(abxcb2 / a_m_b2))
         + ABXCB[0] * a_m_b * dist * sin_a * sin_t
               * (-(ABXCB[0] * (2 * C[2] - 2 * B[2])
                    + ABXCB[2] * (-2 * C[0] + 2 * B[0]))
                      / (2 * a_m_b2)
                  - abxcb2 * (-2 * A[1] + 2 * B[1]) / (2 * a_m_b2 * a_m_b2))
               / (abxcb2 * std::sqrt(abxcb2 / a_m_b2))
         - AB[0] * AB[1] * cos_a * dist / (a_m_b * a_m_b2)
         + CB[2] * dist * sin_a * sin_t / (a_m_b * std::sqrt(abxcb2 / a_m_b2))
         - cos_t * dist * sin_a * (-ABXCB[1] * AB[2] + ABXCB[2] * AB[1])
               * (-(ABXCB[0] * (2 * C[2] - 2 * B[2])
                    + ABXCB[2] * (-2 * C[0] + 2 * B[0]))
                      / (2 * a_m_b2)
                  - abxcb2 * (-2 * A[1] + 2 * B[1]) / (2 * a_m_b2 * a_m_b2))
               / (abxcb2 * std::sqrt(abxcb2 / a_m_b2))
         - cos_t * dist * sin_a * (AB[0] * CB[1] - 2 * AB[1] * CB[0])
               / (a_m_b2 * std::sqrt(abxcb2 / a_m_b2))
         - cos_t * dist * sin_a * (-2 * A[1] + 2 * B[1])
               * (-ABXCB[1] * AB[2] + ABXCB[2] * AB[1])
               / (a_m_b2 * a_m_b2 * std::sqrt(abxcb2 / a_m_b2)));
    dcoord.dp(1, 1) =
        (-ABXCB[1] * AB[1] * dist * sin_a * sin_t
             / (a_m_b2 * a_m_b * std::sqrt(abxcb2 / a_m_b2))
         + ABXCB[1] * a_m_b * dist * sin_a * sin_t
               * (-(ABXCB[0] * (2 * C[2] - 2 * B[2])
                    + ABXCB[2] * (-2 * C[0] + 2 * B[0]))
                      / (2 * a_m_b2)
                  - abxcb2 * (-2 * A[1] + 2 * B[1]) / (2 * a_m_b2 * a_m_b2))
               / (abxcb2 * std::sqrt(abxcb2 / a_m_b2))
         - AB[1] * AB[1] * cos_a * dist / (a_m_b2 * a_m_b) + 1
         + cos_t * dist * sin_a * (-ABXCB[0] * AB[2] + ABXCB[2] * AB[0])
               * (-(ABXCB[0] * (2 * C[2] - 2 * B[2])
                    + ABXCB[2] * (-2 * C[0] + 2 * B[0]))
                      / (2 * a_m_b2)
                  - abxcb2 * (-2 * A[1] + 2 * B[1]) / (2 * a_m_b2 * a_m_b2))
               / (abxcb2 * std::sqrt(abxcb2 / a_m_b2))
         + cos_t * dist * sin_a * (-AB[0] * CB[0] - AB[2] * CB[2])
               / (a_m_b2 * std::sqrt(abxcb2 / a_m_b2))
         + cos_t * dist * sin_a * (-2 * A[1] + 2 * B[1])
               * (-ABXCB[0] * AB[2] + ABXCB[2] * AB[0])
               / (a_m_b2 * a_m_b2 * std::sqrt(abxcb2 / a_m_b2))
         + cos_a * dist / a_m_b);
    dcoord.dp(1, 2) =
        (-ABXCB[2] * AB[1] * dist * sin_a * sin_t
             / (a_m_b2 * a_m_b * std::sqrt(abxcb2 / a_m_b2))
         + ABXCB[2] * a_m_b * dist * sin_a * sin_t
               * (-(ABXCB[0] * (2 * C[2] - 2 * B[2])
                    + ABXCB[2] * (-2 * C[0] + 2 * B[0]))
                      / (2 * a_m_b2)
                  - abxcb2 * (-2 * A[1] + 2 * B[1]) / (2 * a_m_b2 * a_m_b2))
               / (abxcb2 * std::sqrt(abxcb2 / a_m_b2))
         - AB[1] * AB[2] * cos_a * dist / (a_m_b * a_m_b2)
         - CB[0] * dist * sin_a * sin_t / (a_m_b * std::sqrt(abxcb2 / a_m_b2))
         + cos_t * dist * sin_a * (ABXCB[0] * AB[1] - ABXCB[1] * AB[0])
               * (-(ABXCB[0] * (2 * C[2] - 2 * B[2])
                    + ABXCB[2] * (-2 * C[0] + 2 * B[0]))
                      / (2 * a_m_b2)
                  - abxcb2 * (-2 * A[1] + 2 * B[1]) / (2 * a_m_b2 * a_m_b2))
               / (abxcb2 * std::sqrt(abxcb2 / a_m_b2))
         + cos_t * dist * sin_a * (2 * AB[1] * CB[2] - AB[2] * CB[1])
               / (a_m_b2 * std::sqrt(abxcb2 / a_m_b2))
         + cos_t * dist * sin_a * (-2 * A[1] + 2 * B[1])
               * (ABXCB[0] * AB[1] - ABXCB[1] * AB[0])
               / (a_m_b2 * a_m_b2 * std::sqrt(abxcb2 / a_m_b2)));
    dcoord.dp(2, 0) =
        (-ABXCB[0] * AB[2] * dist * sin_a * sin_t
             / (a_m_b2 * a_m_b * std::sqrt(abxcb2 / a_m_b2))
         + ABXCB[0] * a_m_b * dist * sin_a * sin_t
               * (-(ABXCB[0] * (-2 * C[1] + 2 * B[1])
                    - ABXCB[1] * (-2 * C[0] + 2 * B[0]))
                      / (2 * a_m_b2)
                  - abxcb2 * (-2 * A[2] + 2 * B[2]) / (2 * a_m_b2 * a_m_b2))
               / (abxcb2 * std::sqrt(abxcb2 / a_m_b2))
         - AB[0] * AB[2] * cos_a * dist / (a_m_b * a_m_b2)
         - CB[1] * dist * sin_a * sin_t / (a_m_b * std::sqrt(abxcb2 / a_m_b2))
         - cos_t * dist * sin_a * (-ABXCB[1] * AB[2] + ABXCB[2] * AB[1])
               * (-(ABXCB[0] * (-2 * C[1] + 2 * B[1])
                    - ABXCB[1] * (-2 * C[0] + 2 * B[0]))
                      / (2 * a_m_b2)
                  - abxcb2 * (-2 * A[2] + 2 * B[2]) / (2 * a_m_b2 * a_m_b2))
               / (abxcb2 * std::sqrt(abxcb2 / a_m_b2))
         - cos_t * dist * sin_a * (AB[0] * CB[2] - 2 * AB[2] * CB[0])
               / (a_m_b2 * std::sqrt(abxcb2 / a_m_b2))
         - cos_t * dist * sin_a * (-2 * A[2] + 2 * B[2])
               * (-ABXCB[1] * AB[2] + ABXCB[2] * AB[1])
               / (a_m_b2 * a_m_b2 * std::sqrt(abxcb2 / a_m_b2)));
    dcoord.dp(2, 1) =
        (-ABXCB[1] * AB[2] * dist * sin_a * sin_t
             / (a_m_b2 * a_m_b * std::sqrt(abxcb2 / a_m_b2))
         + ABXCB[1] * a_m_b * dist * sin_a * sin_t
               * (-(ABXCB[0] * (-2 * C[1] + 2 * B[1])
                    - ABXCB[1] * (-2 * C[0] + 2 * B[0]))
                      / (2 * a_m_b2)
                  - abxcb2 * (-2 * A[2] + 2 * B[2]) / (2 * a_m_b2 * a_m_b2))
               / (abxcb2 * std::sqrt(abxcb2 / a_m_b2))
         - AB[1] * AB[2] * cos_a * dist / (a_m_b * a_m_b2)
         + CB[0] * dist * sin_a * sin_t / (a_m_b * std::sqrt(abxcb2 / a_m_b2))
         + cos_t * dist * sin_a * (-ABXCB[0] * AB[2] + ABXCB[2] * AB[0])
               * (-(ABXCB[0] * (-2 * C[1] + 2 * B[1])
                    - ABXCB[1] * (-2 * C[0] + 2 * B[0]))
                      / (2 * a_m_b2)
                  - abxcb2 * (-2 * A[2] + 2 * B[2]) / (2 * a_m_b2 * a_m_b2))
               / (abxcb2 * std::sqrt(abxcb2 / a_m_b2))
         + cos_t * dist * sin_a * (-AB[1] * CB[2] + 2 * AB[2] * CB[1])
               / (a_m_b2 * std::sqrt(abxcb2 / a_m_b2))
         + cos_t * dist * sin_a * (-2 * A[2] + 2 * B[2])
               * (-ABXCB[0] * AB[2] + ABXCB[2] * AB[0])
               / (a_m_b2 * a_m_b2 * std::sqrt(abxcb2 / a_m_b2)));
    dcoord.dp(2, 2) =
        (-ABXCB[2] * AB[2] * dist * sin_a * sin_t
             / (a_m_b2 * a_m_b * std::sqrt(abxcb2 / a_m_b2))
         + ABXCB[2] * a_m_b * dist * sin_a * sin_t
               * (-(ABXCB[0] * (-2 * C[1] + 2 * B[1])
                    - ABXCB[1] * (-2 * C[0] + 2 * B[0]))
                      / (2 * a_m_b2)
                  - abxcb2 * (-2 * A[2] + 2 * B[2]) / (2 * a_m_b2 * a_m_b2))
               / (abxcb2 * std::sqrt(abxcb2 / a_m_b2))
         - AB[2] * AB[2] * cos_a * dist / (a_m_b * a_m_b2) + 1
         + cos_t * dist * sin_a * (ABXCB[0] * AB[1] - ABXCB[1] * AB[0])
               * (-(ABXCB[0] * (-2 * C[1] + 2 * B[1])
                    - ABXCB[1] * (-2 * C[0] + 2 * B[0]))
                      / (2 * a_m_b2)
                  - abxcb2 * (-2 * A[2] + 2 * B[2]) / (2 * a_m_b2 * a_m_b2))
               / (abxcb2 * std::sqrt(abxcb2 / a_m_b2))
         + cos_t * dist * sin_a * (-AB[0] * CB[0] - AB[1] * CB[1])
               / (a_m_b2 * std::sqrt(abxcb2 / a_m_b2))
         + cos_t * dist * sin_a * (-2 * A[2] + 2 * B[2])
               * (ABXCB[0] * AB[1] - ABXCB[1] * AB[0])
               / (a_m_b2 * a_m_b2 * std::sqrt(abxcb2 / a_m_b2))
         + cos_a * dist / a_m_b);

    dcoord.dgp(0, 0) =
        (ABXCB[0] * AB[0] * dist * sin_a * sin_t
             / (a_m_b2 * a_m_b * std::sqrt(abxcb2 / a_m_b2))
         + ABXCB[0] * a_m_b * dist * sin_a * sin_t
               * (-(-ABXCB[1] * (2 * A[2] - 2 * C[2])
                    + ABXCB[2] * (2 * A[1] - 2 * C[1]))
                      / (2 * a_m_b2)
                  - abxcb2 * (2 * A[0] - 2 * B[0]) / (2 * a_m_b2 * a_m_b2))
               / (abxcb2 * std::sqrt(abxcb2 / a_m_b2))
         + AB[0] * AB[0] * cos_a * dist / (a_m_b * a_m_b2)
         - cos_t * dist * sin_a * (-ABXCB[1] * AB[2] + ABXCB[2] * AB[1])
               * (-(-ABXCB[1] * (2 * A[2] - 2 * C[2])
                    + ABXCB[2] * (2 * A[1] - 2 * C[1]))
                      / (2 * a_m_b2)
                  - abxcb2 * (2 * A[0] - 2 * B[0]) / (2 * a_m_b2 * a_m_b2))
               / (abxcb2 * std::sqrt(abxcb2 / a_m_b2))
         - cos_t * dist * sin_a
               * (AB[1] * (A[1] - C[1]) + AB[2] * (A[2] - C[2]))
               / (a_m_b2 * std::sqrt(abxcb2 / a_m_b2))
         - cos_t * dist * sin_a * (2 * A[0] - 2 * B[0])
               * (-ABXCB[1] * AB[2] + ABXCB[2] * AB[1])
               / (a_m_b2 * a_m_b2 * std::sqrt(abxcb2 / a_m_b2))
         - cos_a * dist / a_m_b);
    dcoord.dgp(0, 1) =
        (ABXCB[1] * AB[0] * dist * sin_a * sin_t
             / (a_m_b2 * a_m_b * std::sqrt(abxcb2 / a_m_b2))
         + ABXCB[1] * a_m_b * dist * sin_a * sin_t
               * (-(-ABXCB[1] * (2 * A[2] - 2 * C[2])
                    + ABXCB[2] * (2 * A[1] - 2 * C[1]))
                      / (2 * a_m_b2)
                  - abxcb2 * (2 * A[0] - 2 * B[0]) / (2 * a_m_b2 * a_m_b2))
               / (abxcb2 * std::sqrt(abxcb2 / a_m_b2))
         + AB[0] * AB[1] * cos_a * dist / (a_m_b * a_m_b2)
         + cos_t * dist * sin_a * (-ABXCB[0] * AB[2] + ABXCB[2] * AB[0])
               * (-(-ABXCB[1] * (2 * A[2] - 2 * C[2])
                    + ABXCB[2] * (2 * A[1] - 2 * C[1]))
                      / (2 * a_m_b2)
                  - abxcb2 * (2 * A[0] - 2 * B[0]) / (2 * a_m_b2 * a_m_b2))
               / (abxcb2 * std::sqrt(abxcb2 / a_m_b2))
         + cos_t * dist * sin_a * (-ABXCB[2] + AB[0] * (A[1] - C[1]))
               / (a_m_b2 * std::sqrt(abxcb2 / a_m_b2))
         + cos_t * dist * sin_a * (2 * A[0] - 2 * B[0])
               * (-ABXCB[0] * AB[2] + ABXCB[2] * AB[0])
               / (a_m_b2 * a_m_b2 * std::sqrt(abxcb2 / a_m_b2))
         + dist * sin_a * sin_t * (-A[2] + C[2])
               / (a_m_b * std::sqrt(abxcb2 / a_m_b2)));
    dcoord.dgp(0, 2) =
        (ABXCB[2] * AB[0] * dist * sin_a * sin_t
             / (a_m_b2 * a_m_b * std::sqrt(abxcb2 / a_m_b2))
         + ABXCB[2] * a_m_b * dist * sin_a * sin_t
               * (-(-ABXCB[1] * (2 * A[2] - 2 * C[2])
                    + ABXCB[2] * (2 * A[1] - 2 * C[1]))
                      / (2 * a_m_b2)
                  - abxcb2 * (2 * A[0] - 2 * B[0]) / (2 * a_m_b2 * a_m_b2))
               / (abxcb2 * std::sqrt(abxcb2 / a_m_b2))
         + AB[0] * AB[2] * cos_a * dist / (a_m_b * a_m_b2)
         + cos_t * dist * sin_a * (ABXCB[0] * AB[1] - ABXCB[1] * AB[0])
               * (-(-ABXCB[1] * (2 * A[2] - 2 * C[2])
                    + ABXCB[2] * (2 * A[1] - 2 * C[1]))
                      / (2 * a_m_b2)
                  - abxcb2 * (2 * A[0] - 2 * B[0]) / (2 * a_m_b2 * a_m_b2))
               / (abxcb2 * std::sqrt(abxcb2 / a_m_b2))
         + cos_t * dist * sin_a * (ABXCB[1] + AB[0] * (A[2] - C[2]))
               / (a_m_b2 * std::sqrt(abxcb2 / a_m_b2))
         + cos_t * dist * sin_a * (2 * A[0] - 2 * B[0])
               * (ABXCB[0] * AB[1] - ABXCB[1] * AB[0])
               / (a_m_b2 * a_m_b2 * std::sqrt(abxcb2 / a_m_b2))
         + dist * sin_a * sin_t * (A[1] - C[1])
               / (a_m_b * std::sqrt(abxcb2 / a_m_b2)));
    dcoord.dgp(1, 0) =
        (ABXCB[0] * AB[1] * dist * sin_a * sin_t
             / (a_m_b2 * a_m_b * std::sqrt(abxcb2 / a_m_b2))
         + ABXCB[0] * a_m_b * dist * sin_a * sin_t
               * (-(ABXCB[0] * (2 * A[2] - 2 * C[2])
                    + ABXCB[2] * (-2 * A[0] + 2 * C[0]))
                      / (2 * a_m_b2)
                  - abxcb2 * (2 * A[1] - 2 * B[1]) / (2 * a_m_b2 * a_m_b2))
               / (abxcb2 * std::sqrt(abxcb2 / a_m_b2))
         + AB[0] * AB[1] * cos_a * dist / (a_m_b * a_m_b2)
         - cos_t * dist * sin_a * (-ABXCB[1] * AB[2] + ABXCB[2] * AB[1])
               * (-(ABXCB[0] * (2 * A[2] - 2 * C[2])
                    + ABXCB[2] * (-2 * A[0] + 2 * C[0]))
                      / (2 * a_m_b2)
                  - abxcb2 * (2 * A[1] - 2 * B[1]) / (2 * a_m_b2 * a_m_b2))
               / (abxcb2 * std::sqrt(abxcb2 / a_m_b2))
         - cos_t * dist * sin_a * (-ABXCB[2] + AB[1] * (-A[0] + C[0]))
               / (a_m_b2 * std::sqrt(abxcb2 / a_m_b2))
         - cos_t * dist * sin_a * (2 * A[1] - 2 * B[1])
               * (-ABXCB[1] * AB[2] + ABXCB[2] * AB[1])
               / (a_m_b2 * a_m_b2 * std::sqrt(abxcb2 / a_m_b2))
         + dist * sin_a * sin_t * (A[2] - C[2])
               / (a_m_b * std::sqrt(abxcb2 / a_m_b2)));
    dcoord.dgp(1, 1) =
        (ABXCB[1] * AB[1] * dist * sin_a * sin_t
             / (a_m_b2 * a_m_b * std::sqrt(abxcb2 / a_m_b2))
         + ABXCB[1] * a_m_b * dist * sin_a * sin_t
               * (-(ABXCB[0] * (2 * A[2] - 2 * C[2])
                    + ABXCB[2] * (-2 * A[0] + 2 * C[0]))
                      / (2 * a_m_b2)
                  - abxcb2 * (2 * A[1] - 2 * B[1]) / (2 * a_m_b2 * a_m_b2))
               / (abxcb2 * std::sqrt(abxcb2 / a_m_b2))
         + AB[1] * AB[1] * cos_a * dist / (a_m_b * a_m_b2)
         + cos_t * dist * sin_a * (-ABXCB[0] * AB[2] + ABXCB[2] * AB[0])
               * (-(ABXCB[0] * (2 * A[2] - 2 * C[2])
                    + ABXCB[2] * (-2 * A[0] + 2 * C[0]))
                      / (2 * a_m_b2)
                  - abxcb2 * (2 * A[1] - 2 * B[1]) / (2 * a_m_b2 * a_m_b2))
               / (abxcb2 * std::sqrt(abxcb2 / a_m_b2))
         + cos_t * dist * sin_a
               * (AB[0] * (-A[0] + C[0]) - AB[2] * (A[2] - C[2]))
               / (a_m_b2 * std::sqrt(abxcb2 / a_m_b2))
         + cos_t * dist * sin_a * (2 * A[1] - 2 * B[1])
               * (-ABXCB[0] * AB[2] + ABXCB[2] * AB[0])
               / (a_m_b2 * a_m_b2 * std::sqrt(abxcb2 / a_m_b2))
         - cos_a * dist / a_m_b);
    dcoord.dgp(1, 2) =
        (ABXCB[2] * AB[1] * dist * sin_a * sin_t
             / (a_m_b2 * a_m_b * std::sqrt(abxcb2 / a_m_b2))
         + ABXCB[2] * a_m_b * dist * sin_a * sin_t
               * (-(ABXCB[0] * (2 * A[2] - 2 * C[2])
                    + ABXCB[2] * (-2 * A[0] + 2 * C[0]))
                      / (2 * a_m_b2)
                  - abxcb2 * (2 * A[1] - 2 * B[1]) / (2 * a_m_b2 * a_m_b2))
               / (abxcb2 * std::sqrt(abxcb2 / a_m_b2))
         + AB[1] * AB[2] * cos_a * dist / (a_m_b * a_m_b2)
         + cos_t * dist * sin_a * (ABXCB[0] * AB[1] - ABXCB[1] * AB[0])
               * (-(ABXCB[0] * (2 * A[2] - 2 * C[2])
                    + ABXCB[2] * (-2 * A[0] + 2 * C[0]))
                      / (2 * a_m_b2)
                  - abxcb2 * (2 * A[1] - 2 * B[1]) / (2 * a_m_b2 * a_m_b2))
               / (abxcb2 * std::sqrt(abxcb2 / a_m_b2))
         + cos_t * dist * sin_a * (-ABXCB[0] + AB[1] * (A[2] - C[2]))
               / (a_m_b2 * std::sqrt(abxcb2 / a_m_b2))
         + cos_t * dist * sin_a * (2 * A[1] - 2 * B[1])
               * (ABXCB[0] * AB[1] - ABXCB[1] * AB[0])
               / (a_m_b2 * a_m_b2 * std::sqrt(abxcb2 / a_m_b2))
         + dist * sin_a * sin_t * (-A[0] + C[0])
               / (a_m_b * std::sqrt(abxcb2 / a_m_b2)));
    dcoord.dgp(2, 0) =
        (ABXCB[0] * AB[2] * dist * sin_a * sin_t
             / (a_m_b2 * a_m_b * std::sqrt(abxcb2 / a_m_b2))
         + ABXCB[0] * a_m_b * dist * sin_a * sin_t
               * (-(ABXCB[0] * (-2 * A[1] + 2 * C[1])
                    - ABXCB[1] * (-2 * A[0] + 2 * C[0]))
                      / (2 * a_m_b2)
                  - abxcb2 * (2 * A[2] - 2 * B[2]) / (2 * a_m_b2 * a_m_b2))
               / (abxcb2 * std::sqrt(abxcb2 / a_m_b2))
         + AB[0] * AB[2] * cos_a * dist / (a_m_b * a_m_b2)
         - cos_t * dist * sin_a * (-ABXCB[1] * AB[2] + ABXCB[2] * AB[1])
               * (-(ABXCB[0] * (-2 * A[1] + 2 * C[1])
                    - ABXCB[1] * (-2 * A[0] + 2 * C[0]))
                      / (2 * a_m_b2)
                  - abxcb2 * (2 * A[2] - 2 * B[2]) / (2 * a_m_b2 * a_m_b2))
               / (abxcb2 * std::sqrt(abxcb2 / a_m_b2))
         - cos_t * dist * sin_a * (ABXCB[1] + AB[2] * (-A[0] + C[0]))
               / (a_m_b2 * std::sqrt(abxcb2 / a_m_b2))
         - cos_t * dist * sin_a * (2 * A[2] - 2 * B[2])
               * (-ABXCB[1] * AB[2] + ABXCB[2] * AB[1])
               / (a_m_b2 * a_m_b2 * std::sqrt(abxcb2 / a_m_b2))
         + dist * sin_a * sin_t * (-A[1] + C[1])
               / (a_m_b * std::sqrt(abxcb2 / a_m_b2)));
    dcoord.dgp(2, 1) =
        (ABXCB[1] * AB[2] * dist * sin_a * sin_t
             / (a_m_b2 * a_m_b * std::sqrt(abxcb2 / a_m_b2))
         + ABXCB[1] * a_m_b * dist * sin_a * sin_t
               * (-(ABXCB[0] * (-2 * A[1] + 2 * C[1])
                    - ABXCB[1] * (-2 * A[0] + 2 * C[0]))
                      / (2 * a_m_b2)
                  - abxcb2 * (2 * A[2] - 2 * B[2]) / (2 * a_m_b2 * a_m_b2))
               / (abxcb2 * std::sqrt(abxcb2 / a_m_b2))
         + AB[1] * AB[2] * cos_a * dist / (a_m_b * a_m_b2)
         + cos_t * dist * sin_a * (-ABXCB[0] * AB[2] + ABXCB[2] * AB[0])
               * (-(ABXCB[0] * (-2 * A[1] + 2 * C[1])
                    - ABXCB[1] * (-2 * A[0] + 2 * C[0]))
                      / (2 * a_m_b2)
                  - abxcb2 * (2 * A[2] - 2 * B[2]) / (2 * a_m_b2 * a_m_b2))
               / (abxcb2 * std::sqrt(abxcb2 / a_m_b2))
         + cos_t * dist * sin_a * (ABXCB[0] - AB[2] * (-A[1] + C[1]))
               / (a_m_b2 * std::sqrt(abxcb2 / a_m_b2))
         + cos_t * dist * sin_a * (2 * A[2] - 2 * B[2])
               * (-ABXCB[0] * AB[2] + ABXCB[2] * AB[0])
               / (a_m_b2 * a_m_b2 * std::sqrt(abxcb2 / a_m_b2))
         + dist * sin_a * sin_t * (A[0] - C[0])
               / (a_m_b * std::sqrt(abxcb2 / a_m_b2)));
    dcoord.dgp(2, 2) =
        (ABXCB[2] * AB[2] * dist * sin_a * sin_t
             / (a_m_b2 * a_m_b * std::sqrt(abxcb2 / a_m_b2))
         + ABXCB[2] * a_m_b * dist * sin_a * sin_t
               * (-(ABXCB[0] * (-2 * A[1] + 2 * C[1])
                    - ABXCB[1] * (-2 * A[0] + 2 * C[0]))
                      / (2 * a_m_b2)
                  - abxcb2 * (2 * A[2] - 2 * B[2]) / (2 * a_m_b2 * a_m_b2))
               / (abxcb2 * std::sqrt(abxcb2 / a_m_b2))
         + AB[2] * AB[2] * cos_a * dist / (a_m_b * a_m_b2)
         + cos_t * dist * sin_a * (ABXCB[0] * AB[1] - ABXCB[1] * AB[0])
               * (-(ABXCB[0] * (-2 * A[1] + 2 * C[1])
                    - ABXCB[1] * (-2 * A[0] + 2 * C[0]))
                      / (2 * a_m_b2)
                  - abxcb2 * (2 * A[2] - 2 * B[2]) / (2 * a_m_b2 * a_m_b2))
               / (abxcb2 * std::sqrt(abxcb2 / a_m_b2))
         + cos_t * dist * sin_a
               * (AB[0] * (-A[0] + C[0]) + AB[1] * (-A[1] + C[1]))
               / (a_m_b2 * std::sqrt(abxcb2 / a_m_b2))
         + cos_t * dist * sin_a * (2 * A[2] - 2 * B[2])
               * (ABXCB[0] * AB[1] - ABXCB[1] * AB[0])
               / (a_m_b2 * a_m_b2 * std::sqrt(abxcb2 / a_m_b2))
         - cos_a * dist / a_m_b);

    dcoord.dggp(0, 0) =
        (-ABXCB[0] * dist * sin_a * sin_t
             * (-ABXCB[1] * (-2 * A[2] + 2 * B[2])
                + ABXCB[2] * (-2 * A[1] + 2 * B[1]))
             / (2 * a_m_b * abxcb2 * std::sqrt(abxcb2 / a_m_b2))
         - cos_t * dist * sin_a * (-AB[1] * AB[1] - AB[2] * AB[2])
               / (a_m_b2 * std::sqrt(abxcb2 / a_m_b2))
         + cos_t * dist * sin_a * (-ABXCB[1] * AB[2] + ABXCB[2] * AB[1])
               * (-ABXCB[1] * (-2 * A[2] + 2 * B[2])
                  + ABXCB[2] * (-2 * A[1] + 2 * B[1]))
               / (2 * a_m_b2 * abxcb2 * std::sqrt(abxcb2 / a_m_b2)));
    dcoord.dggp(0, 1) =
        (-ABXCB[1] * dist * sin_a * sin_t
             * (-ABXCB[1] * (-2 * A[2] + 2 * B[2])
                + ABXCB[2] * (-2 * A[1] + 2 * B[1]))
             / (2 * a_m_b * abxcb2 * std::sqrt(abxcb2 / a_m_b2))
         - AB[0] * AB[1] * cos_t * dist * sin_a
               / (a_m_b2 * std::sqrt(abxcb2 / a_m_b2))
         + AB[2] * dist * sin_a * sin_t / (a_m_b * std::sqrt(abxcb2 / a_m_b2))
         - cos_t * dist * sin_a * (-ABXCB[0] * AB[2] + ABXCB[2] * AB[0])
               * (-ABXCB[1] * (-2 * A[2] + 2 * B[2])
                  + ABXCB[2] * (-2 * A[1] + 2 * B[1]))
               / (2 * a_m_b2 * abxcb2 * std::sqrt(abxcb2 / a_m_b2)));
    dcoord.dggp(0, 2) =
        (-ABXCB[2] * dist * sin_a * sin_t
             * (-ABXCB[1] * (-2 * A[2] + 2 * B[2])
                + ABXCB[2] * (-2 * A[1] + 2 * B[1]))
             / (2 * a_m_b * abxcb2 * std::sqrt(abxcb2 / a_m_b2))
         - AB[0] * AB[2] * cos_t * dist * sin_a
               / (a_m_b2 * std::sqrt(abxcb2 / a_m_b2))
         - AB[1] * dist * sin_a * sin_t / (a_m_b * std::sqrt(abxcb2 / a_m_b2))
         - cos_t * dist * sin_a * (ABXCB[0] * AB[1] - ABXCB[1] * AB[0])
               * (-ABXCB[1] * (-2 * A[2] + 2 * B[2])
                  + ABXCB[2] * (-2 * A[1] + 2 * B[1]))
               / (2 * a_m_b2 * abxcb2 * std::sqrt(abxcb2 / a_m_b2)));
    dcoord.dggp(1, 0) =
        (-ABXCB[0] * dist * sin_a * sin_t
             * (ABXCB[0] * (-2 * A[2] + 2 * B[2])
                + ABXCB[2] * (2 * A[0] - 2 * B[0]))
             / (2 * a_m_b * abxcb2 * std::sqrt(abxcb2 / a_m_b2))
         - AB[0] * AB[1] * cos_t * dist * sin_a
               / (a_m_b2 * std::sqrt(abxcb2 / a_m_b2))
         - AB[2] * dist * sin_a * sin_t / (a_m_b * std::sqrt(abxcb2 / a_m_b2))
         + cos_t * dist * sin_a
               * (ABXCB[0] * (-2 * A[2] + 2 * B[2])
                  + ABXCB[2] * (2 * A[0] - 2 * B[0]))
               * (-ABXCB[1] * AB[2] + ABXCB[2] * AB[1])
               / (2 * a_m_b2 * abxcb2 * std::sqrt(abxcb2 / a_m_b2)));
    dcoord.dggp(1, 1) =
        (-ABXCB[1] * dist * sin_a * sin_t
             * (ABXCB[0] * (-2 * A[2] + 2 * B[2])
                + ABXCB[2] * (2 * A[0] - 2 * B[0]))
             / (2 * a_m_b * abxcb2 * std::sqrt(abxcb2 / a_m_b2))
         + cos_t * dist * sin_a * (AB[0] * AB[0] + AB[2] * AB[2])
               / (a_m_b2 * std::sqrt(abxcb2 / a_m_b2))
         - cos_t * dist * sin_a * (-ABXCB[0] * AB[2] + ABXCB[2] * AB[0])
               * (ABXCB[0] * (-2 * A[2] + 2 * B[2])
                  + ABXCB[2] * (2 * A[0] - 2 * B[0]))
               / (2 * a_m_b2 * abxcb2 * std::sqrt(abxcb2 / a_m_b2)));
    dcoord.dggp(1, 2) =
        (-ABXCB[2] * dist * sin_a * sin_t
             * (ABXCB[0] * (-2 * A[2] + 2 * B[2])
                + ABXCB[2] * (2 * A[0] - 2 * B[0]))
             / (2 * a_m_b * abxcb2 * std::sqrt(abxcb2 / a_m_b2))
         + AB[0] * dist * sin_a * sin_t / (a_m_b * std::sqrt(abxcb2 / a_m_b2))
         - AB[1] * AB[2] * cos_t * dist * sin_a
               / (a_m_b2 * std::sqrt(abxcb2 / a_m_b2))
         - cos_t * dist * sin_a * (ABXCB[0] * AB[1] - ABXCB[1] * AB[0])
               * (ABXCB[0] * (-2 * A[2] + 2 * B[2])
                  + ABXCB[2] * (2 * A[0] - 2 * B[0]))
               / (2 * a_m_b2 * abxcb2 * std::sqrt(abxcb2 / a_m_b2)));
    dcoord.dggp(2, 0) =
        (-ABXCB[0] * dist * sin_a * sin_t
             * (ABXCB[0] * (2 * A[1] - 2 * B[1])
                - ABXCB[1] * (2 * A[0] - 2 * B[0]))
             / (2 * a_m_b * abxcb2 * std::sqrt(abxcb2 / a_m_b2))
         - AB[0] * AB[2] * cos_t * dist * sin_a
               / (a_m_b2 * std::sqrt(abxcb2 / a_m_b2))
         + AB[1] * dist * sin_a * sin_t / (a_m_b * std::sqrt(abxcb2 / a_m_b2))
         + cos_t * dist * sin_a
               * (ABXCB[0] * (2 * A[1] - 2 * B[1])
                  - ABXCB[1] * (2 * A[0] - 2 * B[0]))
               * (-ABXCB[1] * AB[2] + ABXCB[2] * AB[1])
               / (2 * a_m_b2 * abxcb2 * std::sqrt(abxcb2 / a_m_b2)));
    dcoord.dggp(2, 1) =
        (-ABXCB[1] * dist * sin_a * sin_t
             * (ABXCB[0] * (2 * A[1] - 2 * B[1])
                - ABXCB[1] * (2 * A[0] - 2 * B[0]))
             / (2 * a_m_b * abxcb2 * std::sqrt(abxcb2 / a_m_b2))
         - AB[0] * dist * sin_a * sin_t / (a_m_b * std::sqrt(abxcb2 / a_m_b2))
         - AB[1] * AB[2] * cos_t * dist * sin_a
               / (a_m_b2 * std::sqrt(abxcb2 / a_m_b2))
         - cos_t * dist * sin_a * (-ABXCB[0] * AB[2] + ABXCB[2] * AB[0])
               * (ABXCB[0] * (2 * A[1] - 2 * B[1])
                  - ABXCB[1] * (2 * A[0] - 2 * B[0]))
               / (2 * a_m_b2 * abxcb2 * std::sqrt(abxcb2 / a_m_b2)));
    dcoord.dggp(2, 2) =
        (-ABXCB[2] * dist * sin_a * sin_t
             * (ABXCB[0] * (2 * A[1] - 2 * B[1])
                - ABXCB[1] * (2 * A[0] - 2 * B[0]))
             / (2 * a_m_b * abxcb2 * std::sqrt(abxcb2 / a_m_b2))
         + cos_t * dist * sin_a * (AB[0] * AB[0] + AB[1] * AB[1])
               / (a_m_b2 * std::sqrt(abxcb2 / a_m_b2))
         - cos_t * dist * sin_a * (ABXCB[0] * AB[1] - ABXCB[1] * AB[0])
               * (ABXCB[0] * (2 * A[1] - 2 * B[1])
                  - ABXCB[1] * (2 * A[0] - 2 * B[0]))
               / (2 * a_m_b2 * abxcb2 * std::sqrt(abxcb2 / a_m_b2)));

    return dcoord;
  }
};

#undef def

}  // namespace common
}  // namespace score
}  // namespace tmol
