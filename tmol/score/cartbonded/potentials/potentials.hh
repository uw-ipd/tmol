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

template <typename Real>
def cblength_V_dV2(Real3 atm1, Real3 atm2, Real K, Real x0)
    ->tuple<Real, Eigen::Matrix<Real, 3, 2>> {
  auto dist = distance<Real>::V_dV(atm1, atm2);
  Real E = 0.5 * K * square(dist.V - x0);
  Real dE = K * (dist.V - x0);
  Eigen::Matrix<Real, 3, 2> dEout;
  dEout.col(0) = dE * dist.dV_dA;
  dEout.col(1) = dE * dist.dV_dB;
  return {E, dEout};
}

template <typename Real>
def cbangle_V_dV2(Real3 atm1, Real3 atm2, Real3 atm3, Real K, Real x0)
    ->tuple<Real, Eigen::Matrix<Real, 3, 3>> {
  auto angle = pt_interior_angle<Real>::V_dV(atm1, atm2, atm3);
  Real E = 0.5 * K * square(angle.V - x0);
  Real dE = K * (angle.V - x0);
  Eigen::Matrix<Real, 3, 3> dEout;
  dEout.col(0) = dE * angle.dV_dA;
  dEout.col(1) = dE * angle.dV_dB;
  dEout.col(2) = dE * angle.dV_dC;
  return {E, dEout};
}

// torsions use a sum of three sin funcs
//   sum_n K*(cos(n*x-x0)+1) for n=1,2,3
template <typename Real>
def cbtorsion_V_dV2(
    Real3 atm1,
    Real3 atm2,
    Real3 atm3,
    Real3 atm4,
    Real K1,
    Real K2,
    Real K3,
    Real phi1,
    Real phi2,
    Real phi3)
    ->tuple<Real, Eigen::Matrix<Real, 3, 4>> {
  auto torsion = dihedral_angle<Real>::V_dV(atm1, atm2, atm3, atm4);

  Real E = K1 * (std::cos(1.0 * torsion.V - phi1) + 1.0)
           + K2 * (std::cos(2.0 * torsion.V - phi2) + 1.0)
           + K3 * (std::cos(3.0 * torsion.V - phi3) + 1.0);
  Real dE = -1.0 * K1 * std::sin(1.0 * torsion.V - phi1)
            - 2.0 * K2 * std::sin(2.0 * torsion.V - phi2)
            - 3.0 * K3 * std::sin(3.0 * torsion.V - phi3);

  Eigen::Matrix<Real, 3, 4> dEout;
  dEout.col(0) = dE * torsion.dV_dI;
  dEout.col(1) = dE * torsion.dV_dJ;
  dEout.col(2) = dE * torsion.dV_dK;
  dEout.col(3) = dE * torsion.dV_dL;

  return {E, dEout};
}

template <typename Real>
def cbangle_V_dV(Real3 atm1, Real3 atm2, Real3 atm3, Real K, Real x0)
    ->tuple<Real, Real3, Real3, Real3> {
  auto angle = pt_interior_angle<Real>::V_dV(atm1, atm2, atm3);
  Real E = 0.5 * K * square(angle.V - x0);
  Real dE = K * (angle.V - x0);
  return {E, dE * angle.dV_dA, dE * angle.dV_dB, dE * angle.dV_dC};
}

// normal torsions use single sin func
//     Keff*(1-cos(period*(x-x0)))
template <typename Real, typename Int>
def cbtorsion_V_dV(
    Real3 atm1, Real3 atm2, Real3 atm3, Real3 atm4, Real K, Real x0, Int period)
    ->tuple<Real, Real3, Real3, Real3, Real3> {
  auto torsion = dihedral_angle<Real>::V_dV(atm1, atm2, atm3, atm4);

  Real Keff = K / (period * period);  // map spring constant to cos
  Real E = Keff * (1.0 - std::cos(period * (torsion.V - x0)));
  Real dE = Keff * period * std::sin(period * (torsion.V - x0));
  return {
      E,
      dE * torsion.dV_dI,
      dE * torsion.dV_dJ,
      dE * torsion.dV_dK,
      dE * torsion.dV_dL};
}

// hydroxyl torsions use a sum of three sin funcs
//   sum_n K*(cos(n*x-x0)+1) for n=1,2,3
template <typename Real>
def cbhxltorsion_V_dV(
    Real3 atm1,
    Real3 atm2,
    Real3 atm3,
    Real3 atm4,
    Real K1,
    Real K2,
    Real K3,
    Real phi1,
    Real phi2,
    Real phi3)
    ->tuple<Real, Real3, Real3, Real3, Real3> {
  auto torsion = dihedral_angle<Real>::V_dV(atm1, atm2, atm3, atm4);

  Real E = K1 * (std::cos(1.0 * torsion.V - phi1) + 1.0)
           + K2 * (std::cos(2.0 * torsion.V - phi2) + 1.0)
           + K3 * (std::cos(3.0 * torsion.V - phi3) + 1.0);
  Real dE = -1.0 * K1 * std::sin(1.0 * torsion.V - phi1)
            - 2.0 * K2 * std::sin(2.0 * torsion.V - phi2)
            - 3.0 * K3 * std::sin(3.0 * torsion.V - phi3);

  return {
      E,
      dE * torsion.dV_dI,
      dE * torsion.dV_dJ,
      dE * torsion.dV_dK,
      dE * torsion.dV_dL};
}

#undef Real3
#undef def
}  // namespace potentials
}  // namespace cartbonded
}  // namespace score
}  // namespace tmol
