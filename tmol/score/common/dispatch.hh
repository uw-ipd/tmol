#pragma once

#include <Eigen/Core>

#include <tmol/utility/tensor/TensorAccessor.h>

namespace tmol {
namespace score {
namespace common {

template <tmol::Device D>
struct ExhaustiveDispatch {
  ExhaustiveDispatch(int n_i, int n_j);

  template <typename Real>
  int scan(
      Real threshold_distance,
      TView<Eigen::Matrix<Real, 3, 1>, 1, D> coords_i,
      TView<Eigen::Matrix<Real, 3, 1>, 1, D> coords_j);

  template <typename ScoreFunc>
  void score(ScoreFunc f);
};

template <tmol::Device D>
struct ExhaustiveTriuDispatch {
  ExhaustiveTriuDispatch(int n_i, int n_j);

  template <typename Real>
  int scan(
      Real threshold_distance,
      TView<Eigen::Matrix<Real, 3, 1>, 1, D> coords_i,
      TView<Eigen::Matrix<Real, 3, 1>, 1, D> coords_j);

  template <typename ScoreFunc>
  void score(ScoreFunc f);
};

}  // namespace common
}  // namespace score
}  // namespace tmol
