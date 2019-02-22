#pragma once

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
namespace rama {
namespace potentials {

using namespace tmol::score::common;

#define def auto EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE

template <typename Real, int N>

using Vec = Eigen::Matrix<Real, N, 1>;

#define Real3 Vec<Real, 3>
#define Real2 Vec<Real, 2>

template <typename Real>
static def square(Real v)->Real {
  return v * v;
}

template <typename Real, typename Int>
def rama_V_dV(
    Real3 phi1,
    Real3 phi2,
    Real3 phi3,
    Real3 phi4,
    Real3 psi1,
    Real3 psi2,
    Real3 psi3,
    Real3 psi4,
    TView<Real, 2, D> coeffs,
    Real2 bbstart,
    Real2 bbstep)
    ->tuple<Real, Real3, Real3, Real3, Real3, Real3, Real3, Real3, Real3> {
  Eigen::Matrix<Real, 2, 1> phipsi;

  auto phi = dihedral_angle<Real>::V_dV(phi1, phi2, phi3, phi4);
  auto psi = dihedral_angle<Real>::V_dV(psi1, psi2, psi3, psi4);

  phipsi[0] = phi.V;
  phipsi[1] = psi.V;

  {V, dVdphi, dVdpsi) = ndspline<2, 3, Real, Int>::interpolate(coeffs, phipsi);

    return {V,
            dVdphi * phi.dV_dI,
            dVdphi * phi.dV_dJ,
            dVdphi * phi.dV_dK,
            dVdphi * phi.dV_dL,
            dVdpsi * psi.dV_dI,
            dVdpsi * psi.dV_dJ,
            dVdpsi * psi.dV_dK,
            dVdpsi * psi.dV_dL};
  }

#undef Real2
#undef Real3

#undef def
}  // namespace potentials
}  // namespace potentials
}  // namespace rama
}  // namespace score
