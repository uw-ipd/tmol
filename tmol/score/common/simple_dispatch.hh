#pragma once

#include <Eigen/Core>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/cuda/stream.hh>

namespace tmol {
namespace score {
namespace common {

template <tmol::Device D>
struct AABBDispatch {
  template <typename Real, typename Func>
  void forall_pairs(
      Real threshold_distance,
      TView<Eigen::Matrix<Real, 3, 1>, 1, D> coords_i,
      TView<Eigen::Matrix<Real, 3, 1>, 1, D> coords_j,
      Func f,
      utility::cuda::CUDAStream stream);

  template <typename Real, typename Int, typename Func>
  void forall_idx_pairs(
      Real threshold_distance,
      TView<Eigen::Matrix<Real, 3, 1>, 1, D> coords_i,
      TView<Eigen::Matrix<Real, 3, 1>, 1, D> coords_j,
      TView<Int, 1, D> coord_idx_i,
      TView<Int, 1, D> coord_idx_j,
      Func f,
      utility::cuda::CUDAStream stream);
};

template <tmol::Device D>
struct AABBTriuDispatch {
  template <typename Real, typename Func>
  int forall_pairs(
      Real threshold_distance,
      TView<Eigen::Matrix<Real, 3, 1>, 1, D> coords_i,
      TView<Eigen::Matrix<Real, 3, 1>, 1, D> coords_j,
      Func f,
      utility::cuda::CUDAStream stream);

  template <typename Real, typename Int, typename Func>
  void forall_idx_pairs(
      Real threshold_distance,
      TView<Eigen::Matrix<Real, 3, 1>, 1, D> coords_i,
      TView<Eigen::Matrix<Real, 3, 1>, 1, D> coords_j,
      TView<Int, 1, D> coord_idx_i,
      TView<Int, 1, D> coord_idx_j,
      Func f,
      utility::cuda::CUDAStream stream);
};

}  // namespace common
}  // namespace score
}  // namespace tmol
