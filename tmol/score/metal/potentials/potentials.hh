#pragma once

#include <cmath>

#include <Eigen/Core>

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/utility/tensor/TensorAccessor.h>

#include "params.hh"

namespace tmol {
namespace score {
namespace metal {
namespace potentials {

using namespace tmol::score::common;

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

// Radial and lateral harmonic on one occupied site, plus its well depth.
// virt_ind < 0 marks an untemplated site, which has no lateral term.
template <typename Real, tmol::Device D>
TMOL_DEVICE_FUNC void accumulate_metal_site(
    TView<Vec<Real, 3>, 1, D> coords,
    int metal_ind,
    int virt_ind,
    int donor_ind,
    MetalSiteParams<Real> const& p,
    TView<Vec<Real, 3>, 2, D> dV_dx,
    bool compute_derivs,
    Real dTdV,
    Real* V) {
  Vec<Real, 3> const metal = coords[metal_ind];
  Vec<Real, 3> const delta = coords[donor_ind] - metal;
  Real const r = delta.norm();
  Real const stretch = r - p.d0;
  Real const k_radial = 1 / (p.radial_sd * p.radial_sd);
  Real score = stretch * stretch * k_radial + p.well_depth;

  Vec<Real, 3> dE_ddelta = Vec<Real, 3>::Zero();
  if (r > 0) {
    dE_ddelta = (2 * stretch * k_radial / r) * delta;
  }
  Vec<Real, 3> dE_dray = Vec<Real, 3>::Zero();
  if (virt_ind >= 0) {
    Vec<Real, 3> const ray = coords[virt_ind] - metal;
    Real const ray_len = ray.norm();
    Vec<Real, 3> const u = ray / ray_len;
    Real const along = delta.dot(u);
    Vec<Real, 3> const off_ray = delta - along * u;
    Real const k_lateral = 1 / (p.lateral_sd * p.lateral_sd);
    score += off_ray.squaredNorm() * k_lateral;
    dE_ddelta += (2 * k_lateral) * off_ray;
    // |off_ray|^2 = |delta|^2 - (delta.u)^2, projected through du/dray
    dE_dray = (-2 * k_lateral * along / ray_len) * off_ray;
  }

  if (V != nullptr) {
    accumulate<D, Real>::add(*V, score);
  }
  if (compute_derivs) {
    accumulate<D, Vec<Real, 3>>::add(dV_dx[0][donor_ind], dE_ddelta * dTdV);
    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][metal_ind], (-dE_ddelta - dE_dray) * dTdV);
    if (virt_ind >= 0) {
      accumulate<D, Vec<Real, 3>>::add(dV_dx[0][virt_ind], dE_dray * dTdV);
    }
  }
}

// Harmonic on one metal-virtual or virtual-virtual separation.
template <typename Real, tmol::Device D>
TMOL_DEVICE_FUNC void accumulate_metal_fan_pair(
    TView<Vec<Real, 3>, 1, D> coords,
    int a_ind,
    int b_ind,
    MetalFanParams<Real> const& p,
    TView<Vec<Real, 3>, 2, D> dV_dx,
    bool compute_derivs,
    Real dTdV,
    Real* V) {
  Vec<Real, 3> const sep = coords[a_ind] - coords[b_ind];
  Real const dist = sep.norm();
  Real const k = 1 / (p.sd * p.sd);
  Real const stretch = dist - p.l0;
  if (V != nullptr) {
    accumulate<D, Real>::add(*V, stretch * stretch * k);
  }
  if (compute_derivs && dist > 0) {
    Vec<Real, 3> const dE_da = (2 * stretch * k / dist) * sep;
    accumulate<D, Vec<Real, 3>>::add(dV_dx[0][a_ind], dE_da * dTdV);
    accumulate<D, Vec<Real, 3>>::add(dV_dx[0][b_ind], -dE_da * dTdV);
  }
}

// Wall on the separation of two metals bridged by one donor: the floor is
// where the metal-donor-metal angle reaches its minimum with both metals at
// their ideal donor distances d1 and d2; the width maps the angular width onto
// the separation there.
template <typename Real, tmol::Device D>
TMOL_DEVICE_FUNC void accumulate_metal_bridge(
    TView<Vec<Real, 3>, 1, D> coords,
    int a_ind,
    int b_ind,
    Real d1,
    Real d2,
    MetalBridgeParams<Real> const& p,
    TView<Vec<Real, 3>, 2, D> dV_dx,
    bool compute_derivs,
    Real dTdV,
    Real* V) {
  if (p.floor <= 0) {
    return;  // a donor with no floor puts no wall on its metals
  }
  Real const floor_sq = d1 * d1 + d2 * d2 - 2 * d1 * d2 * std::cos(p.floor);
  if (floor_sq <= 0) {
    return;
  }
  Real const floor = std::sqrt(floor_sq);
  Real const sd = d1 * d2 * std::sin(p.floor) / floor * p.width;
  Vec<Real, 3> const sep = coords[a_ind] - coords[b_ind];
  Real const dist = sep.norm();
  if (dist >= floor) {
    return;
  }
  Real const k = 1 / (sd * sd);
  Real const short_by = floor - dist;
  if (V != nullptr) {
    accumulate<D, Real>::add(*V, short_by * short_by * k);
  }
  if (compute_derivs && dist > 0) {
    Vec<Real, 3> const dE_da = (-2 * short_by * k / dist) * sep;
    accumulate<D, Vec<Real, 3>>::add(dV_dx[0][a_ind], dE_da * dTdV);
    accumulate<D, Vec<Real, 3>>::add(dV_dx[0][b_ind], -dE_da * dTdV);
  }
}

}  // namespace potentials
}  // namespace metal
}  // namespace score
}  // namespace tmol
