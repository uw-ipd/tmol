#pragma once

#include <cmath>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/score/common/geom.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/tuple_operators.hh>

#include <pybind11/pybind11.h>

namespace tmol {
namespace score {
namespace genbonded {
namespace potentials {

using namespace tmol::score::common;

#define def auto EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;
#define Real3 Vec<Real, 3>

// ---------------------------------------------------------------------------
// gbtorsion_V_dV
//
// Compute the genbonded torsion energy and its gradient with respect to the
// four atomic Cartesian coordinates.
//
// Energy form:
//   E = k1*(1 + cos(  theta - offset))
//     + k2*(1 + cos(2*theta - offset))
//     + k3*(1 + cos(3*theta - offset))
//     + k4*(1 + cos(4*theta - offset))
//
// Returns: tuple<E, dE/dx[4]>
//   where dE/dx[i] is a 3-vector of Cartesian gradient for atom i.
// ---------------------------------------------------------------------------
template <typename Real>
def gbtorsion_V_dV(
    Real3 atm1,
    Real3 atm2,
    Real3 atm3,
    Real3 atm4,
    Real k1,
    Real k2,
    Real k3,
    Real k4,
    Real offset) -> tuple<Real, Vec<Real3, 4>> {
  auto torsion = dihedral_angle<Real>::V_dV(atm1, atm2, atm3, atm4);
  Real theta = torsion.V;

  // offset is an additive constant (Rosetta convention), not a phase.
  // Per-period phases (f1..f4) are all 0 in the current database.
  Real E = k1 * (1 + std::cos(theta)) + k2 * (1 + std::cos(2 * theta))
           + k3 * (1 + std::cos(3 * theta)) + k4 * (1 + std::cos(4 * theta));

  if (k1 < 0) E += -2.0 * k1;
  if (k2 < 0) E += -2.0 * k2;
  if (k3 < 0) E += -2.0 * k3;
  if (k4 < 0) E += -2.0 * k4;

  E += offset;

  // printf("gbtorsion_V_dV E=%f (%f/%f/%f/%f)\n",E,k1,k2,k3,k4);

  Real dEdtheta = -k1 * std::sin(theta) - 2 * k2 * std::sin(2 * theta)
                  - 3 * k3 * std::sin(3 * theta) - 4 * k4 * std::sin(4 * theta);

  Vec<Real3, 4> dEdx;
  dEdx[0] = dEdtheta * torsion.dV_dI;
  dEdx[1] = dEdtheta * torsion.dV_dJ;
  dEdx[2] = dEdtheta * torsion.dV_dK;
  dEdx[3] = dEdtheta * torsion.dV_dL;

  return {E, dEdx};
}

template <typename Real>
def gbimproper_V_dV(
    Real3 atm1, Real3 atm2, Real3 atm3, Real3 atm4, Real k, Real delta)
    -> tuple<Real, Vec<Real3, 4>> {
  auto torsion = dihedral_angle<Real>::V_dV(atm1, atm2, atm3, atm4);
  Real theta = torsion.V;

  // --- energy ------------------------------------------------------------
  Real d = (theta - delta);
  Real E = k * d * d;

  // printf("gbimproper_V_dV E=%f (%f)\n",E,k);

  // --- gradient  ---------------------------------------------------------
  Real dEdtheta = 2 * k * d;

  Vec<Real3, 4> dEdx;
  dEdx[0] = dEdtheta * torsion.dV_dI;
  dEdx[1] = dEdtheta * torsion.dV_dJ;
  dEdx[2] = dEdtheta * torsion.dV_dK;
  dEdx[3] = dEdtheta * torsion.dV_dL;

  return {E, dEdx};
}

#undef def
#undef Real3

}  // namespace potentials
}  // namespace genbonded
}  // namespace score
}  // namespace tmol
