
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cppitertools/product.hpp>
#include <cppitertools/range.hpp>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorUtil.h>

#include <tmol/score/common/tuple.hh>

#include "simple_dispatch.hh"

namespace tmol {
namespace score {
namespace common {

using iter::product;
using iter::range;
using tmol::TView;

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <>
struct AABBDispatch<tmol::Device::CPU> {
  static const tmol::Device D = tmol::Device::CPU;

  template <typename Real, typename Func>
  static void forall_pairs(
      Real threshold_distance,
      TView<Eigen::Matrix<Real, 3, 1>, 1, D> coords_i,
      TView<Eigen::Matrix<Real, 3, 1>, 1, D> coords_j,
      Func f,
      at::cuda::CUDAStream* stream = nullptr) {
    const Eigen::AlignedBox<Real, 3> tbox(
        Vec<Real, 3>(
            -threshold_distance, -threshold_distance, -threshold_distance),
        Vec<Real, 3>(
            threshold_distance, threshold_distance, threshold_distance));
    int n_i = coords_i.size(0);
    int n_j = coords_j.size(0);

    for (int i = 0; i < n_i; ++i) {
      for (int j = 0; j < n_j; ++j) {
        if (tbox.contains(coords_i[i] - coords_j[j])) {
          f(i, j);
        }
      }
    }
  }

  template <typename Real, typename Int, typename Func>
  static void forall_idx_pairs(
      Real threshold_distance,
      TView<Eigen::Matrix<Real, 3, 1>, 1, D> coords_i,
      TView<Eigen::Matrix<Real, 3, 1>, 1, D> coords_j,
      TView<Int, 1, D> coord_idx_i,
      TView<Int, 1, D> coord_idx_j,
      Func f,
      at::cuda::CUDAStream* stream = nullptr) {
    const Eigen::AlignedBox<Real, 3> tbox(
        Vec<Real, 3>(
            -threshold_distance, -threshold_distance, -threshold_distance),
        Vec<Real, 3>(
            threshold_distance, threshold_distance, threshold_distance));
    int n_i = coord_idx_i.size(0);
    int n_j = coord_idx_j.size(0);

    for (int i = 0; i < n_i; ++i) {
      for (int j = 0; j < n_j; ++j) {
        if (tbox.contains(
                coords_i[coord_idx_i[i]] - coords_j[coord_idx_j[j]])) {
          f(i, j);
        }
      }
    }
  }
};

template <>
struct AABBTriuDispatch<tmol::Device::CPU> {
  static const tmol::Device D = tmol::Device::CPU;

  template <typename Real, typename Func>
  static void forall_pairs(
      Real threshold_distance,
      TView<Eigen::Matrix<Real, 3, 1>, 1, D> coords_i,
      TView<Eigen::Matrix<Real, 3, 1>, 1, D> coords_j,
      Func f,
      at::cuda::CUDAStream* stream = nullptr) {
    const Eigen::AlignedBox<Real, 3> tbox(
        Vec<Real, 3>(
            -threshold_distance, -threshold_distance, -threshold_distance),
        Vec<Real, 3>(
            threshold_distance, threshold_distance, threshold_distance));
    int n_i = coords_i.size(0);
    int n_j = coords_j.size(0);

    for (int i = 0; i < n_i; ++i) {
      for (int j = i; j < n_j; ++j) {
        if (tbox.contains(coords_i[i] - coords_j[j])) {
          f(i, j);
        }
      }
    }
  }

  template <typename Real, typename Int, typename Func>
  static void forall_idx_pairs(
      Real threshold_distance,
      TView<Eigen::Matrix<Real, 3, 1>, 1, D> coords_i,
      TView<Eigen::Matrix<Real, 3, 1>, 1, D> coords_j,
      TView<Int, 1, D> coord_idx_i,
      TView<Int, 1, D> coord_idx_j,
      Func f,
      at::cuda::CUDAStream* stream = nullptr) {
    const Eigen::AlignedBox<Real, 3> tbox(
        Vec<Real, 3>(
            -threshold_distance, -threshold_distance, -threshold_distance),
        Vec<Real, 3>(
            threshold_distance, threshold_distance, threshold_distance));
    int n_i = coord_idx_i.size(0);
    int n_j = coord_idx_j.size(0);

    for (int i = 0; i < n_i; ++i) {
      for (int j = i; j < n_j; ++j) {
        if (tbox.contains(
                coords_i[coord_idx_i[i]] - coords_j[coord_idx_j[j]])) {
          f(i, j);
        }
      }
    }
  }
};

}  // namespace common
}  // namespace score
}  // namespace tmol
