#pragma once

#include <tuple>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/score/common/vec.hh>

namespace tmol {
namespace score {
namespace common {

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

}  // namespace common
}  // namespace score
}  // namespace tmol
