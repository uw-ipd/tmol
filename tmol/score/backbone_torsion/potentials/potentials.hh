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
namespace backbone_torsion {
namespace potentials {

using namespace tmol::score::common;

#define def auto EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE

template <typename Real, int N>

using Vec = Eigen::Matrix<Real, N, 1>;

#define CoordQuad Eigen::Matrix<Real, 4, 3>
#define Real2 Vec<Real, 2>

template <tmol::Device D, typename Real, typename Int>
def rama_V_dV(
    CoordQuad phi,
    CoordQuad psi,
    TensorAccessor<Real, 2, D> coeffs,
    Real2 bbstart,
    Real2 bbstep)
    ->tuple<Real, CoordQuad, CoordQuad> {
  Real V;
  Real2 dVdphipsi;
  CoordQuad dV_dphiatm;
  CoordQuad dV_dpsiatm;

  auto phiang = dihedral_angle<Real>::V_dV(
      phi.row(0), phi.row(1), phi.row(2), phi.row(3));
  auto psiang = dihedral_angle<Real>::V_dV(
      psi.row(0), psi.row(1), psi.row(2), psi.row(3));

  Real2 phipsi_idx;
  phipsi_idx[0] = (phiang.V - bbstart[0]) / bbstep[0];
  phipsi_idx[1] = (psiang.V - bbstart[1]) / bbstep[1];

  tie(V, dVdphipsi) =
      tmol::numeric::bspline::ndspline<2, 3, D, Real, Int>::interpolate(
          coeffs, phipsi_idx);

  dV_dphiatm.row(0) = phiang.dV_dI / bbstep[0];
  dV_dphiatm.row(1) = phiang.dV_dJ / bbstep[0];
  dV_dphiatm.row(2) = phiang.dV_dK / bbstep[0];
  dV_dphiatm.row(3) = phiang.dV_dL / bbstep[0];

  dV_dpsiatm.row(0) = psiang.dV_dI / bbstep[1];
  dV_dpsiatm.row(1) = psiang.dV_dJ / bbstep[1];
  dV_dpsiatm.row(2) = psiang.dV_dK / bbstep[1];
  dV_dpsiatm.row(3) = psiang.dV_dL / bbstep[1];

  return {V, dVdphipsi[0] * dV_dphiatm, dVdphipsi[1] * dV_dpsiatm};
}

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

#undef Real2
#undef CoordQuad

#undef def
}  // namespace potentials
}  // namespace backbone_torsion
}  // namespace score
}  // namespace tmol
