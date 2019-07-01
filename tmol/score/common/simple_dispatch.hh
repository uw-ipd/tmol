#pragma once

#include <ATen/cuda/CUDAStream.h>
#include <Eigen/Core>

#include <tmol/utility/tensor/TensorAccessor.h>

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
      at::cuda::CUDAStream* stream = nullptr);

  template <typename Real, typename Int, typename Func>
  void forall_idx_pairs(
      Real threshold_distance,
      TView<Eigen::Matrix<Real, 3, 1>, 1, D> coords_i,
      TView<Eigen::Matrix<Real, 3, 1>, 1, D> coords_j,
      TView<Int, 1, D> coord_idx_i,
      TView<Int, 1, D> coord_idx_j,
      Func f,
      at::cuda::CUDAStream* stream = nullptr);
};

template <tmol::Device D>
struct AABBTriuDispatch {
  template <typename Real, typename Func>
  int forall_pairs(
      Real threshold_distance,
      TView<Eigen::Matrix<Real, 3, 1>, 1, D> coords_i,
      TView<Eigen::Matrix<Real, 3, 1>, 1, D> coords_j,
      Func f,
      at::cuda::CUDAStream* stream = nullptr);

  template <typename Real, typename Int, typename Func>
  void forall_idx_pairs(
      Real threshold_distance,
      TView<Eigen::Matrix<Real, 3, 1>, 1, D> coords_i,
      TView<Eigen::Matrix<Real, 3, 1>, 1, D> coords_j,
      TView<Int, 1, D> coord_idx_i,
      TView<Int, 1, D> coord_idx_j,
      Func f,
      at::cuda::CUDAStream* stream = nullptr);
};

}  // namespace common
}  // namespace score
}  // namespace tmol
