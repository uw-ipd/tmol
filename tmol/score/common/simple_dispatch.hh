#pragma once

#include <Eigen/Core>

#include <tmol/utility/tensor/TensorAccessor.h>

namespace tmol {
namespace score {
namespace common {

template <tmol::Device D>
struct AABBDispatch {
  template <typename Real, typename Func>
  static void forall_pairs(
      Real threshold_distance,
      TView<Eigen::Matrix<Real, 3, 1>, 1, D> coords_i,
      TView<Eigen::Matrix<Real, 3, 1>, 1, D> coords_j,
      Func f);

  template <typename Real, typename Func>
  static void forall_stacked_pairs(
      Real threshold_distance,
      TView<Eigen::Matrix<Real, 3, 1>, 2, D> coords_i,
      TView<Eigen::Matrix<Real, 3, 1>, 2, D> coords_j,
      Func f);

  template <typename Real, typename Int, typename Func>
  static void forall_idx_pairs(
      Real threshold_distance,
      TView<Eigen::Matrix<Real, 3, 1>, 1, D> coords_i,
      TView<Eigen::Matrix<Real, 3, 1>, 1, D> coords_j,
      TView<Int, 1, D> coord_idx_i,
      TView<Int, 1, D> coord_idx_j,
      Func f);

  template <typename Real, typename Int, typename Func>
  static void forall_stacked_idx_pairs(
      Real threshold_distance,
      TView<Eigen::Matrix<Real, 3, 1>, 2, D> coords_i,
      TView<Eigen::Matrix<Real, 3, 1>, 2, D> coords_j,
      TView<Int, 2, D> coord_idx_i,
      TView<Int, 2, D> coord_idx_j,
      Func f);
};

template <tmol::Device D>
struct AABBTriuDispatch {
  template <typename Real, typename Func>
  static void forall_pairs(
      Real threshold_distance,
      TView<Eigen::Matrix<Real, 3, 1>, 1, D> coords_i,
      TView<Eigen::Matrix<Real, 3, 1>, 1, D> coords_j,
      Func f);

  template <typename Real, typename Func>
  static void forall_stacked_pairs(
      Real threshold_distance,
      TView<Eigen::Matrix<Real, 3, 1>, 2, D> coords_i,
      TView<Eigen::Matrix<Real, 3, 1>, 2, D> coords_j,
      Func f);

  template <typename Real, typename Int, typename Func>
  static void forall_idx_pairs(
      Real threshold_distance,
      TView<Eigen::Matrix<Real, 3, 1>, 1, D> coords_i,
      TView<Eigen::Matrix<Real, 3, 1>, 1, D> coords_j,
      TView<Int, 1, D> coord_idx_i,
      TView<Int, 1, D> coord_idx_j,
      Func f);

  template <typename Real, typename Int, typename Func>
  static void forall_stacked_idx_pairs(
      Real threshold_distance,
      TView<Eigen::Matrix<Real, 3, 1>, 2, D> coords_i,
      TView<Eigen::Matrix<Real, 3, 1>, 2, D> coords_j,
      TView<Int, 2, D> coord_idx_i,
      TView<Int, 2, D> coord_idx_j,
      Func f);
};

}  // namespace common
}  // namespace score
}  // namespace tmol
