#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace tmol {
namespace score {
namespace common {

template <typename Real>
EIGEN_DEVICE_FUNC Eigen::Matrix<Real, 3, 1> coord_from_shared(
    Real *coord_array, int atom_ind) {
  Eigen::Matrix<Real, 3, 1> local_coord;
  for (int i = 0; i < 3; ++i) {
    local_coord[i] = coord_array[3 * atom_ind + i];
  }
  return local_coord;
}

}  // namespace common
}  // namespace score
}  // namespace tmol
