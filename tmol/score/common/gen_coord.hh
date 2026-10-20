#pragma once

#include <tmol/utility/portable/math.hh>
#include <tmol/utility/portable/matrix.hh>
#include <tmol/score/common/tuple.hh>

namespace tmol {
namespace score {
namespace common {

#define def auto TMOL_HD

using tmol::portable::col_ref;
using tmol::portable::Mat;
using tmol::portable::Vec;

template <typename Real>
struct build_coordinate {
  typedef Vec<Real, 3> Real3;
  typedef Mat<Real, 3, 3> RealMat;

  struct dV_t {
    RealMat dp;    // parent
    RealMat dgp;   // grand-parent
    RealMat dggp;  // great-grand parent

    def astuple() { return make_tuple(dp, dgp, dggp); }

    static def Zero() -> dV_t {
      return {RealMat::Zero(), RealMat::Zero(), RealMat::Zero()};
    }
  };

  // A = parent
  // B = grand parent
  // C = great-grant parent
  static def V(Real3 A, Real3 B, Real3 C, Real dist, Real angle, Real torsion)
      -> Real3 {
    const Real pi = tmol::portable::math::pi<Real>();

    // Generate orientation frame
    Mat<Real, 3, 3> M{};
    col_ref(M, 0) = (A - B).normalized();
    col_ref(M, 1) = (C - B);
    col_ref(M, 2) = M.col(0).cross(M.col(1));

    Real M2_norm = M.col(2).norm();
    if (M2_norm == 0) {
      // if a/b/c collinear, set M[:,2] to an arbitrary vector perp to
      // M[:,0].
      if (M(0, 0) != 1 && M(0, 0) != -1) {
        col_ref(M, 1) = Real3({1, 0, 0});
        col_ref(M, 2) = M.col(0).cross(M.col(1));
      } else {
        col_ref(M, 1) = Real3({0, 1, 0});
        col_ref(M, 2) = M.col(0).cross(M.col(1));
      }
      M2_norm = M.col(2).norm();
    }
    col_ref(M, 2) /= M2_norm;
    col_ref(M, 1) = M.col(2).cross(M.col(0));

    // Build water in frame
    return (
        M
            * Real3(
                {dist * tmol::portable::math::cos(pi - angle),
                 dist * tmol::portable::math::sin(pi - angle)
                     * tmol::portable::math::cos(torsion),
                 dist * tmol::portable::math::sin(pi - angle)
                     * tmol::portable::math::sin(torsion)})
        + A);
  }

  // A = parent
  // B = grand parent
  // C = great-grant parent
  static def dV(Real3 A, Real3 B, Real3 C, Real dist, Real angle, Real torsion)
      -> dV_t {
    const Real pi = tmol::portable::math::pi<Real>();

    Real sin_a = tmol::portable::math::sin(pi - angle);
    Real cos_a = tmol::portable::math::cos(pi - angle);
    Real sin_t = tmol::portable::math::sin(torsion);
    Real cos_t = tmol::portable::math::cos(torsion);
    Real3 AB = A - B;
    Real3 CB = C - B;
    Real a_m_b = AB.norm();
    Real a_m_b2 = a_m_b * a_m_b;
    Real c_m_b = CB.norm();
    Real3 ABXCB = AB.cross(CB);
    Real abxcb2 = ABXCB.squaredNorm();

    if (abxcb2 == 0) {
      // A, B, C collinear: set deriv to 0
      dV_t dcoord = dV_t::Zero();
      dcoord.dp = RealMat::Identity();
      return dcoord;
    }

    // Hoisted loop invariants.
    //
    // These three appear 135, 54 and 54 times across the twenty-seven
    // expressions below, which are machine-generated from a symbolic
    // derivation and make no attempt to share anything. The host compiler
    // common-subexpression-eliminates them; Metal's does not, across a block
    // this size, and the cost is extreme -- computing this derivative is 1.8x
    // the forward on the CPU and was 17x on Metal.
    Real const norm_abxcb = tmol::portable::math::sqrt(abxcb2 / a_m_b2);
    Real const abxcb2_norm_abxcb = abxcb2 * norm_abxcb;
    Real const a_m_b4 = a_m_b2 * a_m_b2;
    Real const inv_a_m_b = Real(1) / a_m_b;
    Real const inv_two_a_m_b_abxcb2_norm =
        Real(1) / (2 * a_m_b * abxcb2_norm_abxcb);

    // Reciprocals of the loop invariants, formed once.
    //
    // The generated expressions below divide by products of these roughly
    // 250 times. Metal compiles with MTLMathModeSafe by default, so each of
    // those is an IEEE-correct division -- a multi-instruction sequence, not
    // one op -- which is why this derivative costs 17x its forward there
    // against 1.8x on the host.
    //
    // Reciprocal-then-multiply is not bit-identical to division; it is
    // accurate to about one ulp, which is far below the 2^-13 grid the Metal
    // gradient is snapped to anyway (accumulate.hh).
    Real const inv_two_a_m_b2 = Real(1) / (2 * a_m_b2);
    Real const inv_two_a_m_b4 = Real(1) / (2 * a_m_b4);
    Real const inv_abxcb2_norm = Real(1) / abxcb2_norm_abxcb;
    Real const inv_a_m_b3_norm = Real(1) / (a_m_b2 * a_m_b * norm_abxcb);
    Real const inv_a_m_b4_norm = Real(1) / (a_m_b4 * norm_abxcb);
    Real const inv_a_m_b2_norm = Real(1) / (a_m_b2 * norm_abxcb);
    Real const inv_a_m_b_norm = Real(1) / (a_m_b * norm_abxcb);
    Real const inv_a_m_b3 = Real(1) / (a_m_b * a_m_b2);
    Real const inv_two_a_m_b2_abxcb2_norm =
        Real(1) / (2 * a_m_b2 * abxcb2_norm_abxcb);

    dV_t dcoord;
    dcoord.dp(0, 0) =
        (-ABXCB[0] * AB[0] * dist * sin_a * sin_t * inv_a_m_b3_norm
         + ABXCB[0] * a_m_b * dist * sin_a * sin_t
               * (-(-ABXCB[1] * (2 * C[2] - 2 * B[2])
                    + ABXCB[2] * (2 * C[1] - 2 * B[1]))
                      * inv_two_a_m_b2
                  - abxcb2 * (-2 * A[0] + 2 * B[0]) * inv_two_a_m_b4)
               * inv_abxcb2_norm
         - AB[0] * AB[0] * cos_a * dist * inv_a_m_b3 + 1
         - cos_t * dist * sin_a * (-ABXCB[1] * AB[2] + ABXCB[2] * AB[1])
               * (-(-ABXCB[1] * (2 * C[2] - 2 * B[2])
                    + ABXCB[2] * (2 * C[1] - 2 * B[1]))
                      * inv_two_a_m_b2
                  - abxcb2 * (-2 * A[0] + 2 * B[0]) * inv_two_a_m_b4)
               * inv_abxcb2_norm
         - cos_t * dist * sin_a * (AB[1] * CB[1] + AB[2] * CB[2])
               * inv_a_m_b2_norm
         - cos_t * dist * sin_a * (-2 * A[0] + 2 * B[0])
               * (-ABXCB[1] * AB[2] + ABXCB[2] * AB[1]) * inv_a_m_b4_norm
         + cos_a * dist * inv_a_m_b);
    dcoord.dp(0, 1) =
        (-ABXCB[1] * AB[0] * dist * sin_a * sin_t * inv_a_m_b3_norm
         + ABXCB[1] * a_m_b * dist * sin_a * sin_t
               * (-(-ABXCB[1] * (2 * C[2] - 2 * B[2])
                    + ABXCB[2] * (2 * C[1] - 2 * B[1]))
                      * inv_two_a_m_b2
                  - abxcb2 * (-2 * A[0] + 2 * B[0]) * inv_two_a_m_b4)
               * inv_abxcb2_norm
         - AB[0] * AB[1] * cos_a * dist * inv_a_m_b3
         - CB[2] * dist * sin_a * sin_t * inv_a_m_b_norm
         + cos_t * dist * sin_a * (-ABXCB[0] * AB[2] + ABXCB[2] * AB[0])
               * (-(-ABXCB[1] * (2 * C[2] - 2 * B[2])
                    + ABXCB[2] * (2 * C[1] - 2 * B[1]))
                      * inv_two_a_m_b2
                  - abxcb2 * (-2 * A[0] + 2 * B[0]) * inv_two_a_m_b4)
               * inv_abxcb2_norm
         + cos_t * dist * sin_a * (2 * AB[0] * CB[1] - AB[1] * CB[0])
               * inv_a_m_b2_norm
         + cos_t * dist * sin_a * (-2 * A[0] + 2 * B[0])
               * (-ABXCB[0] * AB[2] + ABXCB[2] * AB[0]) * inv_a_m_b4_norm);
    dcoord.dp(0, 2) =
        (-ABXCB[2] * AB[0] * dist * sin_a * sin_t * inv_a_m_b3_norm
         + ABXCB[2] * a_m_b * dist * sin_a * sin_t
               * (-(-ABXCB[1] * (2 * C[2] - 2 * B[2])
                    + ABXCB[2] * (2 * C[1] - 2 * B[1]))
                      * inv_two_a_m_b2
                  - abxcb2 * (-2 * A[0] + 2 * B[0]) * inv_two_a_m_b4)
               * inv_abxcb2_norm
         - AB[0] * AB[2] * cos_a * dist * inv_a_m_b3
         + CB[1] * dist * sin_a * sin_t * inv_a_m_b_norm
         + cos_t * dist * sin_a * (ABXCB[0] * AB[1] - ABXCB[1] * AB[0])
               * (-(-ABXCB[1] * (2 * C[2] - 2 * B[2])
                    + ABXCB[2] * (2 * C[1] - 2 * B[1]))
                      * inv_two_a_m_b2
                  - abxcb2 * (-2 * A[0] + 2 * B[0]) * inv_two_a_m_b4)
               * inv_abxcb2_norm
         + cos_t * dist * sin_a * (2 * AB[0] * CB[2] - AB[2] * CB[0])
               * inv_a_m_b2_norm
         + cos_t * dist * sin_a * (-2 * A[0] + 2 * B[0])
               * (ABXCB[0] * AB[1] - ABXCB[1] * AB[0]) * inv_a_m_b4_norm);
    dcoord.dp(1, 0) =
        (-ABXCB[0] * AB[1] * dist * sin_a * sin_t * inv_a_m_b3_norm
         + ABXCB[0] * a_m_b * dist * sin_a * sin_t
               * (-(ABXCB[0] * (2 * C[2] - 2 * B[2])
                    + ABXCB[2] * (-2 * C[0] + 2 * B[0]))
                      * inv_two_a_m_b2
                  - abxcb2 * (-2 * A[1] + 2 * B[1]) * inv_two_a_m_b4)
               * inv_abxcb2_norm
         - AB[0] * AB[1] * cos_a * dist * inv_a_m_b3
         + CB[2] * dist * sin_a * sin_t * inv_a_m_b_norm
         - cos_t * dist * sin_a * (-ABXCB[1] * AB[2] + ABXCB[2] * AB[1])
               * (-(ABXCB[0] * (2 * C[2] - 2 * B[2])
                    + ABXCB[2] * (-2 * C[0] + 2 * B[0]))
                      * inv_two_a_m_b2
                  - abxcb2 * (-2 * A[1] + 2 * B[1]) * inv_two_a_m_b4)
               * inv_abxcb2_norm
         - cos_t * dist * sin_a * (AB[0] * CB[1] - 2 * AB[1] * CB[0])
               * inv_a_m_b2_norm
         - cos_t * dist * sin_a * (-2 * A[1] + 2 * B[1])
               * (-ABXCB[1] * AB[2] + ABXCB[2] * AB[1]) * inv_a_m_b4_norm);
    dcoord.dp(1, 1) =
        (-ABXCB[1] * AB[1] * dist * sin_a * sin_t * inv_a_m_b3_norm
         + ABXCB[1] * a_m_b * dist * sin_a * sin_t
               * (-(ABXCB[0] * (2 * C[2] - 2 * B[2])
                    + ABXCB[2] * (-2 * C[0] + 2 * B[0]))
                      * inv_two_a_m_b2
                  - abxcb2 * (-2 * A[1] + 2 * B[1]) * inv_two_a_m_b4)
               * inv_abxcb2_norm
         - AB[1] * AB[1] * cos_a * dist * inv_a_m_b3 + 1
         + cos_t * dist * sin_a * (-ABXCB[0] * AB[2] + ABXCB[2] * AB[0])
               * (-(ABXCB[0] * (2 * C[2] - 2 * B[2])
                    + ABXCB[2] * (-2 * C[0] + 2 * B[0]))
                      * inv_two_a_m_b2
                  - abxcb2 * (-2 * A[1] + 2 * B[1]) * inv_two_a_m_b4)
               * inv_abxcb2_norm
         + cos_t * dist * sin_a * (-AB[0] * CB[0] - AB[2] * CB[2])
               * inv_a_m_b2_norm
         + cos_t * dist * sin_a * (-2 * A[1] + 2 * B[1])
               * (-ABXCB[0] * AB[2] + ABXCB[2] * AB[0]) * inv_a_m_b4_norm
         + cos_a * dist * inv_a_m_b);
    dcoord.dp(1, 2) =
        (-ABXCB[2] * AB[1] * dist * sin_a * sin_t * inv_a_m_b3_norm
         + ABXCB[2] * a_m_b * dist * sin_a * sin_t
               * (-(ABXCB[0] * (2 * C[2] - 2 * B[2])
                    + ABXCB[2] * (-2 * C[0] + 2 * B[0]))
                      * inv_two_a_m_b2
                  - abxcb2 * (-2 * A[1] + 2 * B[1]) * inv_two_a_m_b4)
               * inv_abxcb2_norm
         - AB[1] * AB[2] * cos_a * dist * inv_a_m_b3
         - CB[0] * dist * sin_a * sin_t * inv_a_m_b_norm
         + cos_t * dist * sin_a * (ABXCB[0] * AB[1] - ABXCB[1] * AB[0])
               * (-(ABXCB[0] * (2 * C[2] - 2 * B[2])
                    + ABXCB[2] * (-2 * C[0] + 2 * B[0]))
                      * inv_two_a_m_b2
                  - abxcb2 * (-2 * A[1] + 2 * B[1]) * inv_two_a_m_b4)
               * inv_abxcb2_norm
         + cos_t * dist * sin_a * (2 * AB[1] * CB[2] - AB[2] * CB[1])
               * inv_a_m_b2_norm
         + cos_t * dist * sin_a * (-2 * A[1] + 2 * B[1])
               * (ABXCB[0] * AB[1] - ABXCB[1] * AB[0]) * inv_a_m_b4_norm);
    dcoord.dp(2, 0) =
        (-ABXCB[0] * AB[2] * dist * sin_a * sin_t * inv_a_m_b3_norm
         + ABXCB[0] * a_m_b * dist * sin_a * sin_t
               * (-(ABXCB[0] * (-2 * C[1] + 2 * B[1])
                    - ABXCB[1] * (-2 * C[0] + 2 * B[0]))
                      * inv_two_a_m_b2
                  - abxcb2 * (-2 * A[2] + 2 * B[2]) * inv_two_a_m_b4)
               * inv_abxcb2_norm
         - AB[0] * AB[2] * cos_a * dist * inv_a_m_b3
         - CB[1] * dist * sin_a * sin_t * inv_a_m_b_norm
         - cos_t * dist * sin_a * (-ABXCB[1] * AB[2] + ABXCB[2] * AB[1])
               * (-(ABXCB[0] * (-2 * C[1] + 2 * B[1])
                    - ABXCB[1] * (-2 * C[0] + 2 * B[0]))
                      * inv_two_a_m_b2
                  - abxcb2 * (-2 * A[2] + 2 * B[2]) * inv_two_a_m_b4)
               * inv_abxcb2_norm
         - cos_t * dist * sin_a * (AB[0] * CB[2] - 2 * AB[2] * CB[0])
               * inv_a_m_b2_norm
         - cos_t * dist * sin_a * (-2 * A[2] + 2 * B[2])
               * (-ABXCB[1] * AB[2] + ABXCB[2] * AB[1]) * inv_a_m_b4_norm);
    dcoord.dp(2, 1) =
        (-ABXCB[1] * AB[2] * dist * sin_a * sin_t * inv_a_m_b3_norm
         + ABXCB[1] * a_m_b * dist * sin_a * sin_t
               * (-(ABXCB[0] * (-2 * C[1] + 2 * B[1])
                    - ABXCB[1] * (-2 * C[0] + 2 * B[0]))
                      * inv_two_a_m_b2
                  - abxcb2 * (-2 * A[2] + 2 * B[2]) * inv_two_a_m_b4)
               * inv_abxcb2_norm
         - AB[1] * AB[2] * cos_a * dist * inv_a_m_b3
         + CB[0] * dist * sin_a * sin_t * inv_a_m_b_norm
         + cos_t * dist * sin_a * (-ABXCB[0] * AB[2] + ABXCB[2] * AB[0])
               * (-(ABXCB[0] * (-2 * C[1] + 2 * B[1])
                    - ABXCB[1] * (-2 * C[0] + 2 * B[0]))
                      * inv_two_a_m_b2
                  - abxcb2 * (-2 * A[2] + 2 * B[2]) * inv_two_a_m_b4)
               * inv_abxcb2_norm
         + cos_t * dist * sin_a * (-AB[1] * CB[2] + 2 * AB[2] * CB[1])
               * inv_a_m_b2_norm
         + cos_t * dist * sin_a * (-2 * A[2] + 2 * B[2])
               * (-ABXCB[0] * AB[2] + ABXCB[2] * AB[0]) * inv_a_m_b4_norm);
    dcoord.dp(2, 2) =
        (-ABXCB[2] * AB[2] * dist * sin_a * sin_t * inv_a_m_b3_norm
         + ABXCB[2] * a_m_b * dist * sin_a * sin_t
               * (-(ABXCB[0] * (-2 * C[1] + 2 * B[1])
                    - ABXCB[1] * (-2 * C[0] + 2 * B[0]))
                      * inv_two_a_m_b2
                  - abxcb2 * (-2 * A[2] + 2 * B[2]) * inv_two_a_m_b4)
               * inv_abxcb2_norm
         - AB[2] * AB[2] * cos_a * dist * inv_a_m_b3 + 1
         + cos_t * dist * sin_a * (ABXCB[0] * AB[1] - ABXCB[1] * AB[0])
               * (-(ABXCB[0] * (-2 * C[1] + 2 * B[1])
                    - ABXCB[1] * (-2 * C[0] + 2 * B[0]))
                      * inv_two_a_m_b2
                  - abxcb2 * (-2 * A[2] + 2 * B[2]) * inv_two_a_m_b4)
               * inv_abxcb2_norm
         + cos_t * dist * sin_a * (-AB[0] * CB[0] - AB[1] * CB[1])
               * inv_a_m_b2_norm
         + cos_t * dist * sin_a * (-2 * A[2] + 2 * B[2])
               * (ABXCB[0] * AB[1] - ABXCB[1] * AB[0]) * inv_a_m_b4_norm
         + cos_a * dist * inv_a_m_b);

    dcoord.dgp(0, 0) =
        (ABXCB[0] * AB[0] * dist * sin_a * sin_t * inv_a_m_b3_norm
         + ABXCB[0] * a_m_b * dist * sin_a * sin_t
               * (-(-ABXCB[1] * (2 * A[2] - 2 * C[2])
                    + ABXCB[2] * (2 * A[1] - 2 * C[1]))
                      * inv_two_a_m_b2
                  - abxcb2 * (2 * A[0] - 2 * B[0]) * inv_two_a_m_b4)
               * inv_abxcb2_norm
         + AB[0] * AB[0] * cos_a * dist * inv_a_m_b3
         - cos_t * dist * sin_a * (-ABXCB[1] * AB[2] + ABXCB[2] * AB[1])
               * (-(-ABXCB[1] * (2 * A[2] - 2 * C[2])
                    + ABXCB[2] * (2 * A[1] - 2 * C[1]))
                      * inv_two_a_m_b2
                  - abxcb2 * (2 * A[0] - 2 * B[0]) * inv_two_a_m_b4)
               * inv_abxcb2_norm
         - cos_t * dist * sin_a
               * (AB[1] * (A[1] - C[1]) + AB[2] * (A[2] - C[2]))
               * inv_a_m_b2_norm
         - cos_t * dist * sin_a * (2 * A[0] - 2 * B[0])
               * (-ABXCB[1] * AB[2] + ABXCB[2] * AB[1]) * inv_a_m_b4_norm
         - cos_a * dist * inv_a_m_b);
    dcoord.dgp(0, 1) =
        (ABXCB[1] * AB[0] * dist * sin_a * sin_t * inv_a_m_b3_norm
         + ABXCB[1] * a_m_b * dist * sin_a * sin_t
               * (-(-ABXCB[1] * (2 * A[2] - 2 * C[2])
                    + ABXCB[2] * (2 * A[1] - 2 * C[1]))
                      * inv_two_a_m_b2
                  - abxcb2 * (2 * A[0] - 2 * B[0]) * inv_two_a_m_b4)
               * inv_abxcb2_norm
         + AB[0] * AB[1] * cos_a * dist * inv_a_m_b3
         + cos_t * dist * sin_a * (-ABXCB[0] * AB[2] + ABXCB[2] * AB[0])
               * (-(-ABXCB[1] * (2 * A[2] - 2 * C[2])
                    + ABXCB[2] * (2 * A[1] - 2 * C[1]))
                      * inv_two_a_m_b2
                  - abxcb2 * (2 * A[0] - 2 * B[0]) * inv_two_a_m_b4)
               * inv_abxcb2_norm
         + cos_t * dist * sin_a * (-ABXCB[2] + AB[0] * (A[1] - C[1]))
               * inv_a_m_b2_norm
         + cos_t * dist * sin_a * (2 * A[0] - 2 * B[0])
               * (-ABXCB[0] * AB[2] + ABXCB[2] * AB[0]) * inv_a_m_b4_norm
         + dist * sin_a * sin_t * (-A[2] + C[2]) * inv_a_m_b_norm);
    dcoord.dgp(0, 2) =
        (ABXCB[2] * AB[0] * dist * sin_a * sin_t * inv_a_m_b3_norm
         + ABXCB[2] * a_m_b * dist * sin_a * sin_t
               * (-(-ABXCB[1] * (2 * A[2] - 2 * C[2])
                    + ABXCB[2] * (2 * A[1] - 2 * C[1]))
                      * inv_two_a_m_b2
                  - abxcb2 * (2 * A[0] - 2 * B[0]) * inv_two_a_m_b4)
               * inv_abxcb2_norm
         + AB[0] * AB[2] * cos_a * dist * inv_a_m_b3
         + cos_t * dist * sin_a * (ABXCB[0] * AB[1] - ABXCB[1] * AB[0])
               * (-(-ABXCB[1] * (2 * A[2] - 2 * C[2])
                    + ABXCB[2] * (2 * A[1] - 2 * C[1]))
                      * inv_two_a_m_b2
                  - abxcb2 * (2 * A[0] - 2 * B[0]) * inv_two_a_m_b4)
               * inv_abxcb2_norm
         + cos_t * dist * sin_a * (ABXCB[1] + AB[0] * (A[2] - C[2]))
               * inv_a_m_b2_norm
         + cos_t * dist * sin_a * (2 * A[0] - 2 * B[0])
               * (ABXCB[0] * AB[1] - ABXCB[1] * AB[0]) * inv_a_m_b4_norm
         + dist * sin_a * sin_t * (A[1] - C[1]) * inv_a_m_b_norm);
    dcoord.dgp(1, 0) =
        (ABXCB[0] * AB[1] * dist * sin_a * sin_t * inv_a_m_b3_norm
         + ABXCB[0] * a_m_b * dist * sin_a * sin_t
               * (-(ABXCB[0] * (2 * A[2] - 2 * C[2])
                    + ABXCB[2] * (-2 * A[0] + 2 * C[0]))
                      * inv_two_a_m_b2
                  - abxcb2 * (2 * A[1] - 2 * B[1]) * inv_two_a_m_b4)
               * inv_abxcb2_norm
         + AB[0] * AB[1] * cos_a * dist * inv_a_m_b3
         - cos_t * dist * sin_a * (-ABXCB[1] * AB[2] + ABXCB[2] * AB[1])
               * (-(ABXCB[0] * (2 * A[2] - 2 * C[2])
                    + ABXCB[2] * (-2 * A[0] + 2 * C[0]))
                      * inv_two_a_m_b2
                  - abxcb2 * (2 * A[1] - 2 * B[1]) * inv_two_a_m_b4)
               * inv_abxcb2_norm
         - cos_t * dist * sin_a * (-ABXCB[2] + AB[1] * (-A[0] + C[0]))
               * inv_a_m_b2_norm
         - cos_t * dist * sin_a * (2 * A[1] - 2 * B[1])
               * (-ABXCB[1] * AB[2] + ABXCB[2] * AB[1]) * inv_a_m_b4_norm
         + dist * sin_a * sin_t * (A[2] - C[2]) * inv_a_m_b_norm);
    dcoord.dgp(1, 1) =
        (ABXCB[1] * AB[1] * dist * sin_a * sin_t * inv_a_m_b3_norm
         + ABXCB[1] * a_m_b * dist * sin_a * sin_t
               * (-(ABXCB[0] * (2 * A[2] - 2 * C[2])
                    + ABXCB[2] * (-2 * A[0] + 2 * C[0]))
                      * inv_two_a_m_b2
                  - abxcb2 * (2 * A[1] - 2 * B[1]) * inv_two_a_m_b4)
               * inv_abxcb2_norm
         + AB[1] * AB[1] * cos_a * dist * inv_a_m_b3
         + cos_t * dist * sin_a * (-ABXCB[0] * AB[2] + ABXCB[2] * AB[0])
               * (-(ABXCB[0] * (2 * A[2] - 2 * C[2])
                    + ABXCB[2] * (-2 * A[0] + 2 * C[0]))
                      * inv_two_a_m_b2
                  - abxcb2 * (2 * A[1] - 2 * B[1]) * inv_two_a_m_b4)
               * inv_abxcb2_norm
         + cos_t * dist * sin_a
               * (AB[0] * (-A[0] + C[0]) - AB[2] * (A[2] - C[2]))
               * inv_a_m_b2_norm
         + cos_t * dist * sin_a * (2 * A[1] - 2 * B[1])
               * (-ABXCB[0] * AB[2] + ABXCB[2] * AB[0]) * inv_a_m_b4_norm
         - cos_a * dist * inv_a_m_b);
    dcoord.dgp(1, 2) =
        (ABXCB[2] * AB[1] * dist * sin_a * sin_t * inv_a_m_b3_norm
         + ABXCB[2] * a_m_b * dist * sin_a * sin_t
               * (-(ABXCB[0] * (2 * A[2] - 2 * C[2])
                    + ABXCB[2] * (-2 * A[0] + 2 * C[0]))
                      * inv_two_a_m_b2
                  - abxcb2 * (2 * A[1] - 2 * B[1]) * inv_two_a_m_b4)
               * inv_abxcb2_norm
         + AB[1] * AB[2] * cos_a * dist * inv_a_m_b3
         + cos_t * dist * sin_a * (ABXCB[0] * AB[1] - ABXCB[1] * AB[0])
               * (-(ABXCB[0] * (2 * A[2] - 2 * C[2])
                    + ABXCB[2] * (-2 * A[0] + 2 * C[0]))
                      * inv_two_a_m_b2
                  - abxcb2 * (2 * A[1] - 2 * B[1]) * inv_two_a_m_b4)
               * inv_abxcb2_norm
         + cos_t * dist * sin_a * (-ABXCB[0] + AB[1] * (A[2] - C[2]))
               * inv_a_m_b2_norm
         + cos_t * dist * sin_a * (2 * A[1] - 2 * B[1])
               * (ABXCB[0] * AB[1] - ABXCB[1] * AB[0]) * inv_a_m_b4_norm
         + dist * sin_a * sin_t * (-A[0] + C[0]) * inv_a_m_b_norm);
    dcoord.dgp(2, 0) =
        (ABXCB[0] * AB[2] * dist * sin_a * sin_t * inv_a_m_b3_norm
         + ABXCB[0] * a_m_b * dist * sin_a * sin_t
               * (-(ABXCB[0] * (-2 * A[1] + 2 * C[1])
                    - ABXCB[1] * (-2 * A[0] + 2 * C[0]))
                      * inv_two_a_m_b2
                  - abxcb2 * (2 * A[2] - 2 * B[2]) * inv_two_a_m_b4)
               * inv_abxcb2_norm
         + AB[0] * AB[2] * cos_a * dist * inv_a_m_b3
         - cos_t * dist * sin_a * (-ABXCB[1] * AB[2] + ABXCB[2] * AB[1])
               * (-(ABXCB[0] * (-2 * A[1] + 2 * C[1])
                    - ABXCB[1] * (-2 * A[0] + 2 * C[0]))
                      * inv_two_a_m_b2
                  - abxcb2 * (2 * A[2] - 2 * B[2]) * inv_two_a_m_b4)
               * inv_abxcb2_norm
         - cos_t * dist * sin_a * (ABXCB[1] + AB[2] * (-A[0] + C[0]))
               * inv_a_m_b2_norm
         - cos_t * dist * sin_a * (2 * A[2] - 2 * B[2])
               * (-ABXCB[1] * AB[2] + ABXCB[2] * AB[1]) * inv_a_m_b4_norm
         + dist * sin_a * sin_t * (-A[1] + C[1]) * inv_a_m_b_norm);
    dcoord.dgp(2, 1) =
        (ABXCB[1] * AB[2] * dist * sin_a * sin_t * inv_a_m_b3_norm
         + ABXCB[1] * a_m_b * dist * sin_a * sin_t
               * (-(ABXCB[0] * (-2 * A[1] + 2 * C[1])
                    - ABXCB[1] * (-2 * A[0] + 2 * C[0]))
                      * inv_two_a_m_b2
                  - abxcb2 * (2 * A[2] - 2 * B[2]) * inv_two_a_m_b4)
               * inv_abxcb2_norm
         + AB[1] * AB[2] * cos_a * dist * inv_a_m_b3
         + cos_t * dist * sin_a * (-ABXCB[0] * AB[2] + ABXCB[2] * AB[0])
               * (-(ABXCB[0] * (-2 * A[1] + 2 * C[1])
                    - ABXCB[1] * (-2 * A[0] + 2 * C[0]))
                      * inv_two_a_m_b2
                  - abxcb2 * (2 * A[2] - 2 * B[2]) * inv_two_a_m_b4)
               * inv_abxcb2_norm
         + cos_t * dist * sin_a * (ABXCB[0] - AB[2] * (-A[1] + C[1]))
               * inv_a_m_b2_norm
         + cos_t * dist * sin_a * (2 * A[2] - 2 * B[2])
               * (-ABXCB[0] * AB[2] + ABXCB[2] * AB[0]) * inv_a_m_b4_norm
         + dist * sin_a * sin_t * (A[0] - C[0]) * inv_a_m_b_norm);
    dcoord.dgp(2, 2) =
        (ABXCB[2] * AB[2] * dist * sin_a * sin_t * inv_a_m_b3_norm
         + ABXCB[2] * a_m_b * dist * sin_a * sin_t
               * (-(ABXCB[0] * (-2 * A[1] + 2 * C[1])
                    - ABXCB[1] * (-2 * A[0] + 2 * C[0]))
                      * inv_two_a_m_b2
                  - abxcb2 * (2 * A[2] - 2 * B[2]) * inv_two_a_m_b4)
               * inv_abxcb2_norm
         + AB[2] * AB[2] * cos_a * dist * inv_a_m_b3
         + cos_t * dist * sin_a * (ABXCB[0] * AB[1] - ABXCB[1] * AB[0])
               * (-(ABXCB[0] * (-2 * A[1] + 2 * C[1])
                    - ABXCB[1] * (-2 * A[0] + 2 * C[0]))
                      * inv_two_a_m_b2
                  - abxcb2 * (2 * A[2] - 2 * B[2]) * inv_two_a_m_b4)
               * inv_abxcb2_norm
         + cos_t * dist * sin_a
               * (AB[0] * (-A[0] + C[0]) + AB[1] * (-A[1] + C[1]))
               * inv_a_m_b2_norm
         + cos_t * dist * sin_a * (2 * A[2] - 2 * B[2])
               * (ABXCB[0] * AB[1] - ABXCB[1] * AB[0]) * inv_a_m_b4_norm
         - cos_a * dist * inv_a_m_b);

    dcoord.dggp(0, 0) =
        (-ABXCB[0] * dist * sin_a * sin_t
             * (-ABXCB[1] * (-2 * A[2] + 2 * B[2])
                + ABXCB[2] * (-2 * A[1] + 2 * B[1]))
             * inv_two_a_m_b_abxcb2_norm
         - cos_t * dist * sin_a * (-AB[1] * AB[1] - AB[2] * AB[2])
               * inv_a_m_b2_norm
         + cos_t * dist * sin_a * (-ABXCB[1] * AB[2] + ABXCB[2] * AB[1])
               * (-ABXCB[1] * (-2 * A[2] + 2 * B[2])
                  + ABXCB[2] * (-2 * A[1] + 2 * B[1]))
               * inv_two_a_m_b2_abxcb2_norm);
    dcoord.dggp(0, 1) =
        (-ABXCB[1] * dist * sin_a * sin_t
             * (-ABXCB[1] * (-2 * A[2] + 2 * B[2])
                + ABXCB[2] * (-2 * A[1] + 2 * B[1]))
             * inv_two_a_m_b_abxcb2_norm
         - AB[0] * AB[1] * cos_t * dist * sin_a * inv_a_m_b2_norm
         + AB[2] * dist * sin_a * sin_t * inv_a_m_b_norm
         - cos_t * dist * sin_a * (-ABXCB[0] * AB[2] + ABXCB[2] * AB[0])
               * (-ABXCB[1] * (-2 * A[2] + 2 * B[2])
                  + ABXCB[2] * (-2 * A[1] + 2 * B[1]))
               * inv_two_a_m_b2_abxcb2_norm);
    dcoord.dggp(0, 2) =
        (-ABXCB[2] * dist * sin_a * sin_t
             * (-ABXCB[1] * (-2 * A[2] + 2 * B[2])
                + ABXCB[2] * (-2 * A[1] + 2 * B[1]))
             * inv_two_a_m_b_abxcb2_norm
         - AB[0] * AB[2] * cos_t * dist * sin_a * inv_a_m_b2_norm
         - AB[1] * dist * sin_a * sin_t * inv_a_m_b_norm
         - cos_t * dist * sin_a * (ABXCB[0] * AB[1] - ABXCB[1] * AB[0])
               * (-ABXCB[1] * (-2 * A[2] + 2 * B[2])
                  + ABXCB[2] * (-2 * A[1] + 2 * B[1]))
               * inv_two_a_m_b2_abxcb2_norm);
    dcoord.dggp(1, 0) =
        (-ABXCB[0] * dist * sin_a * sin_t
             * (ABXCB[0] * (-2 * A[2] + 2 * B[2])
                + ABXCB[2] * (2 * A[0] - 2 * B[0]))
             * inv_two_a_m_b_abxcb2_norm
         - AB[0] * AB[1] * cos_t * dist * sin_a * inv_a_m_b2_norm
         - AB[2] * dist * sin_a * sin_t * inv_a_m_b_norm
         + cos_t * dist * sin_a
               * (ABXCB[0] * (-2 * A[2] + 2 * B[2])
                  + ABXCB[2] * (2 * A[0] - 2 * B[0]))
               * (-ABXCB[1] * AB[2] + ABXCB[2] * AB[1])
               * inv_two_a_m_b2_abxcb2_norm);
    dcoord.dggp(1, 1) =
        (-ABXCB[1] * dist * sin_a * sin_t
             * (ABXCB[0] * (-2 * A[2] + 2 * B[2])
                + ABXCB[2] * (2 * A[0] - 2 * B[0]))
             * inv_two_a_m_b_abxcb2_norm
         + cos_t * dist * sin_a * (AB[0] * AB[0] + AB[2] * AB[2])
               * inv_a_m_b2_norm
         - cos_t * dist * sin_a * (-ABXCB[0] * AB[2] + ABXCB[2] * AB[0])
               * (ABXCB[0] * (-2 * A[2] + 2 * B[2])
                  + ABXCB[2] * (2 * A[0] - 2 * B[0]))
               * inv_two_a_m_b2_abxcb2_norm);
    dcoord.dggp(1, 2) =
        (-ABXCB[2] * dist * sin_a * sin_t
             * (ABXCB[0] * (-2 * A[2] + 2 * B[2])
                + ABXCB[2] * (2 * A[0] - 2 * B[0]))
             * inv_two_a_m_b_abxcb2_norm
         + AB[0] * dist * sin_a * sin_t * inv_a_m_b_norm
         - AB[1] * AB[2] * cos_t * dist * sin_a * inv_a_m_b2_norm
         - cos_t * dist * sin_a * (ABXCB[0] * AB[1] - ABXCB[1] * AB[0])
               * (ABXCB[0] * (-2 * A[2] + 2 * B[2])
                  + ABXCB[2] * (2 * A[0] - 2 * B[0]))
               * inv_two_a_m_b2_abxcb2_norm);
    dcoord.dggp(2, 0) =
        (-ABXCB[0] * dist * sin_a * sin_t
             * (ABXCB[0] * (2 * A[1] - 2 * B[1])
                - ABXCB[1] * (2 * A[0] - 2 * B[0]))
             * inv_two_a_m_b_abxcb2_norm
         - AB[0] * AB[2] * cos_t * dist * sin_a * inv_a_m_b2_norm
         + AB[1] * dist * sin_a * sin_t * inv_a_m_b_norm
         + cos_t * dist * sin_a
               * (ABXCB[0] * (2 * A[1] - 2 * B[1])
                  - ABXCB[1] * (2 * A[0] - 2 * B[0]))
               * (-ABXCB[1] * AB[2] + ABXCB[2] * AB[1])
               * inv_two_a_m_b2_abxcb2_norm);
    dcoord.dggp(2, 1) =
        (-ABXCB[1] * dist * sin_a * sin_t
             * (ABXCB[0] * (2 * A[1] - 2 * B[1])
                - ABXCB[1] * (2 * A[0] - 2 * B[0]))
             * inv_two_a_m_b_abxcb2_norm
         - AB[0] * dist * sin_a * sin_t * inv_a_m_b_norm
         - AB[1] * AB[2] * cos_t * dist * sin_a * inv_a_m_b2_norm
         - cos_t * dist * sin_a * (-ABXCB[0] * AB[2] + ABXCB[2] * AB[0])
               * (ABXCB[0] * (2 * A[1] - 2 * B[1])
                  - ABXCB[1] * (2 * A[0] - 2 * B[0]))
               * inv_two_a_m_b2_abxcb2_norm);
    dcoord.dggp(2, 2) =
        (-ABXCB[2] * dist * sin_a * sin_t
             * (ABXCB[0] * (2 * A[1] - 2 * B[1])
                - ABXCB[1] * (2 * A[0] - 2 * B[0]))
             * inv_two_a_m_b_abxcb2_norm
         + cos_t * dist * sin_a * (AB[0] * AB[0] + AB[1] * AB[1])
               * inv_a_m_b2_norm
         - cos_t * dist * sin_a * (ABXCB[0] * AB[1] - ABXCB[1] * AB[0])
               * (ABXCB[0] * (2 * A[1] - 2 * B[1])
                  - ABXCB[1] * (2 * A[0] - 2 * B[0]))
               * inv_two_a_m_b2_abxcb2_norm);

    return dcoord;
  }
};

#undef def

}  // namespace common
}  // namespace score
}  // namespace tmol
