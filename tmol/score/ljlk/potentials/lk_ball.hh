#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <typename Real, int N, int M>
using Mat = Eigen::Matrix<Real, N, M>;

#define def auto EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
#define Real3 Vec<Real, 3>
#define Real33 Mat<Real, 3, 3>

template <typename Real>
struct build_don_water {
  static def V(Real3 D, Real3 H, Real dist)->Real3 {
    return D + dist * (H - D).normalized();
  }

  static def dV(Real3 D, Real3 H, Real dist)
      ->std::tuple<Mat<Real, 3, 3>, Mat<Real, 3, 3>> {
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

template <typename Real, int N>
struct build_acc_waters {
  typedef Eigen::Matrix<Real, N, 3> WaterCoords;

  static def V(
      Real3 A, Real3 B, Real3 B0, Real dist, Real angle, Vec<Real, N> tors)
      ->WaterCoords {
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
    std::cout << M << std::endl;

    // build waters
    WaterCoords waters;
    for (int i = 0; i < N; ++i) {
      waters.row(i) =
          (M
               * Real3({dist * std::cos(pi - angle),
                        dist * std::sin(pi - angle) * std::cos(tors(i)),
                        dist * std::sin(pi - angle) * std::sin(tors(i))})
           + A);
    }

    return waters;
  }
};

#undef def
#undef Real3

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
