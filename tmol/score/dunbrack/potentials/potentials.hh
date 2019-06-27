#pragma once

#include <math.h>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/score/common/geom.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/tuple_operators.hh>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/numeric/bspline_compiled/bspline.hh>

namespace tmol {
namespace score {
namespace dunbrack {
namespace potentials {

using namespace tmol::score::common;

#define def auto EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

#define CoordQuad Eigen::Matrix<Real, 4, 3>

constexpr float operator"" _2rad(long double deg) {
  return float(M_PI * deg / 180.);
}

// return x mod y in range [0,y)
template <typename Real>
def pos_fmod(Real x, Real y) {
  Real xmody = fmod(x, y);
  if (xmody < 0) xmody += y;
  return xmody;
}

// return x mod y in range [-y/2,y/2]
template <typename Real>
def min_fmod(Real x, Real y) {
  Real xmody = fmod(x, y);
  if (xmody < -y / 2) xmody += y;
  if (xmody > y / 2) xmody -= y;
  return xmody;
}

template <typename Real>
def classify_rotamer(Vec<Real, 4> dihedrals, int n_rotameric_chi)->int {
  // Input dihedral value must be in the range [-pi,+pi)
  // three bins: g+ (  0--120) = 0
  //             t  (120--240) = 1
  //             g- (240--360) = 2
  int rotamer_ind = 0;

  for (int ii = 0; ii < n_rotameric_chi; ++ii) {
    Real iidihe = dihedrals[ii];
    rotamer_ind *= 3;
    if (iidihe < -120.0_2rad) {
      // dihedral between -180 and -120: trans
      rotamer_ind += 1;
    } else if (iidihe < 0) {
      // dihedral between -120 and 0: g-
      rotamer_ind += 2;
    } else if (iidihe < 120.0_2rad) {
      // dihedral between 0 and +120: g+
      rotamer_ind += 0;
    } else {
      // dihedral between +120 and +180: trans
      rotamer_ind += 1;
    }
  }
  return rotamer_ind;
}

#undef def
}  // namespace potentials
}  // namespace dunbrack
}  // namespace score
}  // namespace tmol
