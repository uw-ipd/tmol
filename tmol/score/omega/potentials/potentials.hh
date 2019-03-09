#pragma once

#include <cmath>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pybind11/pybind11.h>

#include <tmol/score/common/geom.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/tuple_operators.hh>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/numeric/bspline_compiled/bspline.hh>

namespace tmol {
namespace score {
namespace rama {
namespace potentials {

using namespace tmol::score::common;

#define def auto EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE

template <typename Real, int N>

using Vec = Eigen::Matrix<Real, N, 1>;

#define CoordQuad Eigen::Matrix<Real, 4, 3>

template <tmol::Device D, typename Real, typename Int>
def omega_V_dV(CoordQuad omega, Real K)->tuple<Real, CoordQuad> {
  Real V;
  Real dVdomega;
  CoordQuad dV_domegaatm;

  auto omegaang = dihedral_angle<Real>::V_dV(
      omega.row(0), omega.row(1), omega.row(2), omega.row(3));

  // note: the angle returned by dihedral_angle is in [-pi,pi]
  Real omega_offset = omegaang.V;
  if (omega_offset > 0.5 * EIGEN_PI) {
    omega_offset -= EIGEN_PI;
  } else if (omega_offset < -0.5 * EIGEN_PI) {
    omega_offset += EIGEN_PI;
  }

  V = K * omega_offset * omega_offset;
  dVdomega = 2 * K * omega_offset;

  dV_domegaatm.row(0) = omegaang.dV_dI;
  dV_domegaatm.row(1) = omegaang.dV_dJ;
  dV_domegaatm.row(2) = omegaang.dV_dK;
  dV_domegaatm.row(3) = omegaang.dV_dL;

  return {V, dVdomega * dV_domegaatm};
}

#undef CoordQuad

#undef def
}  // namespace potentials
}  // namespace rama
}  // namespace score
}  // namespace tmol
