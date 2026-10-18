#pragma once

// Conversions between tmol::portable::Mat and Eigen, used by the files that
// still rely on Eigen expression templates (4x4 transform algebra in
// kinematics, comma-initializers in cartbonded). Metal never includes this.

#include <Eigen/Core>

#include "matrix.hh"

namespace tmol {
namespace portable {

template <typename Real, int R, int C>
TMOL_HD Mat<Real, R, C> from_eigen(Eigen::Matrix<Real, R, C> const& e) {
  Mat<Real, R, C> r{};
  for (int i = 0; i < R; ++i) {
    for (int j = 0; j < C; ++j) {
      r(i, j) = e(i, j);
    }
  }
  return r;
}

template <typename Real, int R, int C>
TMOL_HD Eigen::Matrix<Real, R, C> to_eigen(Mat<Real, R, C> const& a) {
  Eigen::Matrix<Real, R, C> r;
  for (int i = 0; i < R; ++i) {
    for (int j = 0; j < C; ++j) {
      r(i, j) = a(i, j);
    }
  }
  return r;
}

// TView reinterprets torch storage as an array of these element types, so the
// element stride -- sizeof -- must match Eigen exactly, as must the in-memory
// element order (asserted by the byte-layout test in
// tmol/tests/utility/portable).
//
// Alignment deliberately is NOT asserted. Eigen over-aligns fixed-size types
// whose size is a multiple of EIGEN_MAX_STATIC_ALIGN_BYTES so it can emit
// aligned SIMD loads: Eigen::Matrix<float, 4, 1> is 16-byte aligned where the
// portable type is 4-byte aligned. That is safe in one direction only --- the
// portable type makes no alignment assumption, so it can read storage Eigen
// laid out, but a struct that embeds one may pack its members more tightly
// than the Eigen equivalent did. Structs viewed as tensors therefore carry
// their own size assertions.
#define TMOL_ASSERT_EIGEN_LAYOUT(Real, R, C) \
  static_assert(sizeof(Mat<Real, R, C>) == sizeof(Eigen::Matrix<Real, R, C>))

TMOL_ASSERT_EIGEN_LAYOUT(float, 2, 1);
TMOL_ASSERT_EIGEN_LAYOUT(float, 3, 1);
TMOL_ASSERT_EIGEN_LAYOUT(float, 4, 1);
TMOL_ASSERT_EIGEN_LAYOUT(double, 2, 1);
TMOL_ASSERT_EIGEN_LAYOUT(double, 3, 1);
TMOL_ASSERT_EIGEN_LAYOUT(double, 4, 1);
TMOL_ASSERT_EIGEN_LAYOUT(float, 4, 3);
TMOL_ASSERT_EIGEN_LAYOUT(double, 4, 3);
TMOL_ASSERT_EIGEN_LAYOUT(float, 3, 3);
TMOL_ASSERT_EIGEN_LAYOUT(double, 3, 3);

#undef TMOL_ASSERT_EIGEN_LAYOUT

}  // namespace portable
}  // namespace tmol
