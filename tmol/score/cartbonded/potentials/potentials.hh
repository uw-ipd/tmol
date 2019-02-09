#pragma once

#include <cmath>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/score/common/geom.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/tuple_operators.hh>

namespace tmol {
namespace score {
namespace cartbonded {
namespace potentials {

using namespace tmol::score::common;

#define def auto EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

#define Real3 Vec<Real, 3>

template <typename Real>
static def square(Real v)->Real {
  return v * v;
}

template <typename Real>
def cblength_V_dV(Real3 atm1, Real3 atm2, Real K, Real x0)
    ->tuple<Real, Real3, Real3> {
  auto dist = distance<Real>::V_dV(atm1, atm2);
  Real E = 0.5 * K * square(dist.V - x0);
  Real dE = K * (dist.V - x0);
  return {E, dE * dist.dV_dA, dE * dist.dV_dB};
}

#undef Real3
#undef def
}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol
