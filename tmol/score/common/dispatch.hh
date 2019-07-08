#pragma once

#include <Eigen/Core>

#include <tmol/utility/cuda/CUDAStream.hh>
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
      TView<Eigen::Matrix<Real, 3, 1>, 1, D> coords_j,
      utility::cuda::CUDAStream stream);

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
      TView<Eigen::Matrix<Real, 3, 1>, 1, D> coords_j,
      utility::cuda::CUDAStream stream);

  template <typename ScoreFunc>
  void score(ScoreFunc f);
};

template <tmol::Device D>
struct NaiveDispatch {
  NaiveDispatch(int n_i, int n_j);

  template <typename Real>
  int scan(
      Real threshold_distance,
      TView<Eigen::Matrix<Real, 3, 1>, 1, D> coords_i,
      TView<Eigen::Matrix<Real, 3, 1>, 1, D> coords_j,
      utility::cuda::CUDAStream stream);

  template <typename ScoreFunc>
  void score(ScoreFunc f);
};

template <tmol::Device D>
struct NaiveTriuDispatch {
  NaiveTriuDispatch(int n_i, int n_j);

  template <typename Real>
  int scan(
      Real threshold_distance,
      TView<Eigen::Matrix<Real, 3, 1>, 1, D> coords_i,
      TView<Eigen::Matrix<Real, 3, 1>, 1, D> coords_j,
      utility::cuda::CUDAStream stream);

  template <typename ScoreFunc>
  void score(ScoreFunc f);
};

}  // namespace common
}  // namespace score
}  // namespace tmol
