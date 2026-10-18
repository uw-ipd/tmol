#pragma once

#include <tmol/utility/portable/math.hh>
#include <tmol/utility/portable/matrix.hh>

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

#define def auto TMOL_HD

using tmol::portable::Vec;

#define CoordQuad tmol::portable::Mat<Real, 4, 3>
#define Real2 Vec<Real, 2>

template <tmol::Device D, typename Real, typename Int>
def rama_V_dV(
    CoordQuad phi,
    CoordQuad psi,
    TensorAccessor<Real, 2, D> coeffs,
    Real2 bbstart,
    Real2 bbstep) -> tuple<Real, CoordQuad, CoordQuad> {
  Real V = 0.0;
  Real2 dVdphipsi = {0.0, 0.0};
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

  tmol::portable::row_ref(dV_dphiatm, 0) =
      dVdphipsi[0] * phiang.dV_dI / bbstep[0];
  tmol::portable::row_ref(dV_dphiatm, 1) =
      dVdphipsi[0] * phiang.dV_dJ / bbstep[0];
  tmol::portable::row_ref(dV_dphiatm, 2) =
      dVdphipsi[0] * phiang.dV_dK / bbstep[0];
  tmol::portable::row_ref(dV_dphiatm, 3) =
      dVdphipsi[0] * phiang.dV_dL / bbstep[0];

  tmol::portable::row_ref(dV_dpsiatm, 0) =
      dVdphipsi[1] * psiang.dV_dI / bbstep[1];
  tmol::portable::row_ref(dV_dpsiatm, 1) =
      dVdphipsi[1] * psiang.dV_dJ / bbstep[1];
  tmol::portable::row_ref(dV_dpsiatm, 2) =
      dVdphipsi[1] * psiang.dV_dK / bbstep[1];
  tmol::portable::row_ref(dV_dpsiatm, 3) =
      dVdphipsi[1] * psiang.dV_dL / bbstep[1];
  return {V, dV_dphiatm, dV_dpsiatm};
}

template <tmol::Device D, typename Real, typename Int>
def rama_V(
    CoordQuad phi,
    CoordQuad psi,
    TensorAccessor<Real, 2, D> coeffs,
    Real2 bbstart,
    Real2 bbstep) -> Real {
  Real2 phipsi_idx;
  phipsi_idx[0] =
      (dihedral_angle<Real>::V(phi.row(0), phi.row(1), phi.row(2), phi.row(3))
       - bbstart[0])
      / bbstep[0];
  phipsi_idx[1] =
      (dihedral_angle<Real>::V(psi.row(0), psi.row(1), psi.row(2), psi.row(3))
       - bbstart[1])
      / bbstep[1];
  return tmol::numeric::bspline::ndspline<2, 3, D, Real, Int>::interpolate_V(
      coeffs, phipsi_idx);
}

template <tmol::Device D, typename Real, typename Int>
def omega_V_dV(CoordQuad omega, Real K) -> tuple<Real, CoordQuad> {
  Real V;
  Real dVdomega;
  CoordQuad dV_domegaatm;

  auto omegaang = dihedral_angle<Real>::V_dV(
      omega.row(0), omega.row(1), omega.row(2), omega.row(3));

  // note: the angle returned by dihedral_angle is in [-pi,pi]
  Real omega_offset = omegaang.V;
  if (omega_offset > 0.5 * tmol::portable::math::pi<Real>()) {
    omega_offset -= tmol::portable::math::pi<Real>();
  } else if (omega_offset < -0.5 * tmol::portable::math::pi<Real>()) {
    omega_offset += tmol::portable::math::pi<Real>();
  }

  V = K * omega_offset * omega_offset;
  dVdomega = 2 * K * omega_offset;

  tmol::portable::row_ref(dV_domegaatm, 0) = omegaang.dV_dI;
  tmol::portable::row_ref(dV_domegaatm, 1) = omegaang.dV_dJ;
  tmol::portable::row_ref(dV_domegaatm, 2) = omegaang.dV_dK;
  tmol::portable::row_ref(dV_domegaatm, 3) = omegaang.dV_dL;

  return {V, dVdomega * dV_domegaatm};
}

template <typename Real>
def omega_V(CoordQuad omega, Real K) -> Real {
  Real omega_offset = dihedral_angle<Real>::V(
      omega.row(0), omega.row(1), omega.row(2), omega.row(3));
  if (omega_offset > 0.5 * tmol::portable::math::pi<Real>()) {
    omega_offset -= tmol::portable::math::pi<Real>();
  } else if (omega_offset < -0.5 * tmol::portable::math::pi<Real>()) {
    omega_offset += tmol::portable::math::pi<Real>();
  }
  return K * omega_offset * omega_offset;
}

// bb-dep version of omega
template <tmol::Device D, typename Real, typename Int>
def omega_bbdep_V_dV(
    CoordQuad phi,
    CoordQuad psi,
    CoordQuad omega,
    TensorAccessor<Real, 2, D> coeffs_mu,
    TensorAccessor<Real, 2, D> coeffs_sig,
    Real2 bbstart,
    Real2 bbstep,
    Real K) -> tuple<Real, CoordQuad, CoordQuad, CoordQuad> {
  const Real pi = tmol::portable::math::pi<Real>();
  Real V;
  Real dVdomega, dVdphi = 0.0, dVdpsi = 0.0;
  CoordQuad dV_domegaatm, dV_dphiatm, dV_dpsiatm;

  auto omegaang = dihedral_angle<Real>::V_dV(
      omega.row(0), omega.row(1), omega.row(2), omega.row(3));

  // note: the angle returned by dihedral_angle is in [-pi,pi]
  if (omegaang.V > -0.5 * tmol::portable::math::pi<Real>()
      && omegaang.V < 0.5 * tmol::portable::math::pi<Real>()) {
    // cis omega, use simple version
    tie(V, dV_domegaatm) = omega_V_dV<D, Real, Int>(omega, K);
    dV_dphiatm.setZero();
    dV_dpsiatm.setZero();
  } else {
    auto phiang = dihedral_angle<Real>::V_dV(
        phi.row(0), phi.row(1), phi.row(2), phi.row(3));
    auto psiang = dihedral_angle<Real>::V_dV(
        psi.row(0), psi.row(1), psi.row(2), psi.row(3));

    Real mu, sig;
    Real2 dmudphipsi, dsigdphipsi;

    Real2 phipsi_idx;
    phipsi_idx[0] = (phiang.V - bbstart[0]) / bbstep[0];
    phipsi_idx[1] = (psiang.V - bbstart[1]) / bbstep[1];

    tie(mu, dmudphipsi) =
        tmol::numeric::bspline::ndspline<2, 3, D, Real, Int>::interpolate(
            coeffs_mu, phipsi_idx);
    tie(sig, dsigdphipsi) =
        tmol::numeric::bspline::ndspline<2, 3, D, Real, Int>::interpolate(
            coeffs_sig, phipsi_idx);

    // energy
    Real baseline = tmol::portable::math::log(1. / (6. * sqrt(2. * pi)))
                    - tmol::portable::math::log(1. / (sig * sqrt(2. * pi)));
    Real offset = omegaang.V * 180.0 / pi - mu;
    if (offset < -180.0) {
      offset += 360.0;
    }
    Real logprob = offset * offset / (2 * sig * sig);
    V = baseline + logprob;

    // derivatives
    dVdomega = (180.0 / pi) * offset / (sig * sig);

    Real dVdmu = -offset / (sig * sig);
    Real dVdsig = 1 / (sig)-2.0 * logprob / sig;

    dVdphi = dVdmu * dmudphipsi[0] + dVdsig * dsigdphipsi[0];
    dVdpsi = dVdmu * dmudphipsi[1] + dVdsig * dsigdphipsi[1];

    tmol::portable::row_ref(dV_dphiatm, 0) = dVdphi * phiang.dV_dI / bbstep[0];
    tmol::portable::row_ref(dV_dphiatm, 1) = dVdphi * phiang.dV_dJ / bbstep[0];
    tmol::portable::row_ref(dV_dphiatm, 2) = dVdphi * phiang.dV_dK / bbstep[0];
    tmol::portable::row_ref(dV_dphiatm, 3) = dVdphi * phiang.dV_dL / bbstep[0];

    tmol::portable::row_ref(dV_dpsiatm, 0) = dVdpsi * psiang.dV_dI / bbstep[1];
    tmol::portable::row_ref(dV_dpsiatm, 1) = dVdpsi * psiang.dV_dJ / bbstep[1];
    tmol::portable::row_ref(dV_dpsiatm, 2) = dVdpsi * psiang.dV_dK / bbstep[1];
    tmol::portable::row_ref(dV_dpsiatm, 3) = dVdpsi * psiang.dV_dL / bbstep[1];

    tmol::portable::row_ref(dV_domegaatm, 0) = dVdomega * omegaang.dV_dI;
    tmol::portable::row_ref(dV_domegaatm, 1) = dVdomega * omegaang.dV_dJ;
    tmol::portable::row_ref(dV_domegaatm, 2) = dVdomega * omegaang.dV_dK;
    tmol::portable::row_ref(dV_domegaatm, 3) = dVdomega * omegaang.dV_dL;
  }

  return {V, dV_dphiatm, dV_dpsiatm, dV_domegaatm};
}

template <tmol::Device D, typename Real, typename Int>
def omega_bbdep_V(
    CoordQuad phi,
    CoordQuad psi,
    CoordQuad omega,
    TensorAccessor<Real, 2, D> coeffs_mu,
    TensorAccessor<Real, 2, D> coeffs_sig,
    Real2 bbstart,
    Real2 bbstep,
    Real K) -> Real {
  const Real pi = tmol::portable::math::pi<Real>();
  Real omegaang = dihedral_angle<Real>::V(
      omega.row(0), omega.row(1), omega.row(2), omega.row(3));
  if (omegaang > -0.5 * tmol::portable::math::pi<Real>()
      && omegaang < 0.5 * tmol::portable::math::pi<Real>()) {
    return omega_V(omega, K);
  }

  Real2 phipsi_idx;
  phipsi_idx[0] =
      (dihedral_angle<Real>::V(phi.row(0), phi.row(1), phi.row(2), phi.row(3))
       - bbstart[0])
      / bbstep[0];
  phipsi_idx[1] =
      (dihedral_angle<Real>::V(psi.row(0), psi.row(1), psi.row(2), psi.row(3))
       - bbstart[1])
      / bbstep[1];

  Real mu = tmol::numeric::bspline::ndspline<2, 3, D, Real, Int>::interpolate_V(
      coeffs_mu, phipsi_idx);
  Real sig =
      tmol::numeric::bspline::ndspline<2, 3, D, Real, Int>::interpolate_V(
          coeffs_sig, phipsi_idx);
  Real baseline = tmol::portable::math::log(1. / (6. * sqrt(2. * pi)))
                  - tmol::portable::math::log(1. / (sig * sqrt(2. * pi)));
  Real offset = omegaang * 180.0 / pi - mu;
  if (offset < -180.0) {
    offset += 360.0;
  }
  return baseline + offset * offset / (2 * sig * sig);
}

#undef Real2
#undef CoordQuad

#undef def
}  // namespace potentials
}  // namespace backbone_torsion
}  // namespace score
}  // namespace tmol
