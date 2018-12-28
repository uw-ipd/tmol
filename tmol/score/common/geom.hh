#pragma once

#include <tuple>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace tmol {
namespace score {
namespace common {

using namespace std;

template <int N, typename Real>
using Vec = Eigen::Matrix<Real, N, 1>;

#define Real3 Vec<3, Real>

template <typename Real>
Real distance_V(Real3 A, Real3 B) {
  return (A - B).norm();
}

template <typename Real>
auto distance_V_dV(Real3 A, Real3 B) -> tuple<Real, Real3, Real3> {
  Real3 delta = (A - B);
  Real V = delta.norm();

  return {V, delta / V, -delta / V};
}

template <typename Real>
Real interior_angle_V(Real3 A, Real3 B) {
  auto CR = A.cross(B);
  auto z_unit = CR.normalized();

  auto A_norm = A.norm();
  auto B_norm = B.norm();

  return 2 * std::atan2(CR.dot(z_unit), A_norm * B_norm + A.dot(B));
}

template <typename Real>
auto interior_angle_V_dV(Real3 A, Real3 B) -> tuple<Real, Real3, Real3> {
  auto CR = A.cross(B);
  auto z_unit = CR.normalized();

  auto A_norm = A.norm();
  auto B_norm = B.norm();

  return {2 * std::atan2(CR.dot(z_unit), A_norm * B_norm + A.dot(B)),
          (A / A_norm).cross(z_unit) / A_norm,
          -(B / B_norm).cross(z_unit) / B_norm};
}

template <typename Real>
Real pt_interior_angle_V(Real3 A, Real3 B, Real3 C) {
  Real3 BA = A - B;
  Real3 BC = C - B;
  return interior_angle_V(BA, BC);
}

template <typename Real>
auto pt_interior_angle_V_dV(Real3 A, Real3 B, Real3 C)
    -> tuple<Real, Real3, Real3, Real3> {
  Real3 BA = A - B;
  Real3 BC = C - B;
  auto [V, dV_dBA, dV_dBC] = interior_angle_V_dV(BA, BC);
  return {V, dV_dBA, -(dV_dBA + dV_dBC), dV_dBC};
}

template <typename Real>
Real cos_interior_angle_V(Real3 A, Real3 B) {
  return A.dot(B) / (A.norm() * B.norm());
}

template <typename Real>
auto cos_interior_angle_V_dV(Real3 A, Real3 B) -> tuple<Real, Real3, Real3> {
  auto A_norm = A.norm();
  auto B_norm = B.norm();
  auto AB_norm = A.norm() * B.norm();

  auto cosAB = (A.dot(B) / AB_norm);

  return {cosAB,
          -cosAB * A / (A_norm * A_norm) + B / AB_norm,
          -cosAB * B / (B_norm * B_norm) + A / AB_norm};
}

template <typename Real>
Real cos_pt_interior_angle_V(Real3 A, Real3 B, Real3 C) {
  Real3 BA = A - B;
  Real3 BC = C - B;
  return cos_interior_angle_V(BA, BC);
}

template <typename Real>
auto cos_pt_interior_angle_V_dV(Real3 A, Real3 B, Real3 C)
    -> tuple<Real, Real3, Real3, Real3> {
  Real3 BA = A - B;
  Real3 BC = C - B;
  auto [V, dV_dBA, dV_dBC] = cos_interior_angle_V_dV(BA, BC);
  return {V, dV_dBA, -(dV_dBA + dV_dBC), dV_dBC};
}

#undef Real3

}  // namespace common
}  // namespace score
}  // namespace tmol
