#pragma once

#include <tuple>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/score/common/vec.hh>

namespace tmol {
namespace score {
namespace common {

template <typename Real>
Real distance_V(const Vec<3, Real>& a, const Vec<3, Real>& b) {
  return (a - b).norm();
}

template <typename Real>
std::tuple<Real, Vec<3, Real>, Vec<3, Real>> distance_V_dV(
    const Vec<3, Real>& a, const Vec<3, Real>& b) {
  Vec<3, Real> delta = (a - b);
  Real V = delta.norm();

  return {V, delta / V, -delta / V};
}

template <typename Real>
Real interior_angle_V(const Vec<3, Real>& v1, const Vec<3, Real>& v2) {
  auto c = v1.cross(v2);
  auto z_unit = c.normalized();

  auto v1_norm = v1.norm();
  auto v2_norm = v2.norm();

  return 2 * std::atan2(c.dot(z_unit), v1_norm * v2_norm + v1.dot(v2));
}

template <typename Real>
std::tuple<Real, Vec<3, Real>, Vec<3, Real>> interior_angle_V_dV(
    const Vec<3, Real>& v1, const Vec<3, Real>& v2) {
  auto c = v1.cross(v2);
  auto z_unit = c.normalized();

  auto v1_norm = v1.norm();
  auto v2_norm = v2.norm();

  return {2 * std::atan2(c.dot(z_unit), v1_norm * v2_norm + v1.dot(v2)),
          (v1 / v1_norm).cross(z_unit) / v1_norm,
          -(v2 / v2_norm).cross(z_unit) / v2_norm};
}

template <typename Real>
Real interior_angle_V(
    const Vec<3, Real>& A, const Vec<3, Real>& B, const Vec<3, Real>& C) {
  Vec<3, Real> BA = A - B;
  Vec<3, Real> BC = C - B;
  return interior_angle_V(BA, BC);
}

template <typename Real>
std::tuple<Real, Vec<3, Real>, Vec<3, Real>, Vec<3, Real>> interior_angle_V_dV(
    const Vec<3, Real>& A, const Vec<3, Real>& B, const Vec<3, Real>& C) {
  Vec<3, Real> BA = A - B;
  Vec<3, Real> BC = C - B;
  auto [V, dV_dBA, dV_dBC] = interior_angle_V_dV(BA, BC);
  return {V, dV_dBA, -(dV_dBA + dV_dBC), dV_dBC};
}

}  // namespace common
}  // namespace score
}  // namespace tmol
