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

#define def auto EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
#define Real3 Vec<Real, 3>

template <typename Real>
struct build_don_water {
  static def V(Real3 D, Real3 H, Real dist)->Real3 {
    return D + dist * (H - D).normalized();
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
