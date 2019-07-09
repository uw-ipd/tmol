#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <memory>

#include <moderngpu/transform.hxx>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/score/common/tuple.hh>
#include <tmol/utility/cuda/context.hh>

#include "simple_dispatch.hh"

namespace tmol {
namespace score {
namespace common {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <>
struct AABBDispatch<tmol::Device::CUDA> {
  static const tmol::Device D = tmol::Device::CUDA;

  template <typename Real, typename Func>
  static void forall_pairs(
      Real threshold_distance,
      TView<Eigen::Matrix<Real, 3, 1>, 1, D> coords_i,
      TView<Eigen::Matrix<Real, 3, 1>, 1, D> coords_j,
      Func f,
      utility::cuda::CUDAStream stream) {
    const Eigen::AlignedBox<Real, 3> tbox(
        Vec<Real, 3>(
            -threshold_distance, -threshold_distance, -threshold_distance),
        Vec<Real, 3>(
            threshold_distance, threshold_distance, threshold_distance));
    int n_i = coords_i.size(0);
    int n_j = coords_j.size(0);

    auto context = utility::cuda::context_from_stream(stream);

    mgpu::transform(
        [=] MGPU_DEVICE(int index) {
          int i = index / n_j;
          int j = index % n_j;

          if (tbox.contains(coords_i[i] - coords_j[j])) {
            f(i, j);
          }
        },
        n_i * n_j,
        context);
  }

  template <typename Real, typename Int, typename Func>
  static void forall_idx_pairs(
      Real threshold_distance,
      TView<Eigen::Matrix<Real, 3, 1>, 1, D> coords_i,
      TView<Eigen::Matrix<Real, 3, 1>, 1, D> coords_j,
      TView<Int, 1, D> coord_idx_i,
      TView<Int, 1, D> coord_idx_j,
      Func f,
      utility::cuda::CUDAStream stream) {
    const Eigen::AlignedBox<Real, 3> tbox(
        Vec<Real, 3>(
            -threshold_distance, -threshold_distance, -threshold_distance),
        Vec<Real, 3>(
            threshold_distance, threshold_distance, threshold_distance));
    int n_i = coord_idx_i.size(0);
    int n_j = coord_idx_j.size(0);

    auto context = utility::cuda::context_from_stream(stream);

    mgpu::transform(
        [=] MGPU_DEVICE(int index) {
          int i = index / n_j;
          int j = index % n_j;

          if (tbox.contains(
                  coords_i[coord_idx_i[i]] - coords_j[coord_idx_j[j]])) {
            f(i, j);
          }
        },
        n_i * n_j,
        context);
  }
};

template <>
struct AABBTriuDispatch<tmol::Device::CUDA> {
  static const tmol::Device D = tmol::Device::CUDA;

  template <typename Real, typename Fun>
  static void forall_pairs(
      Real threshold_distance,
      TView<Eigen::Matrix<Real, 3, 1>, 1, D> coords_i,
      TView<Eigen::Matrix<Real, 3, 1>, 1, D> coords_j,
      Fun f,
      utility::cuda::CUDAStream stream) {
    const Eigen::AlignedBox<Real, 3> tbox(
        Vec<Real, 3>(
            -threshold_distance, -threshold_distance, -threshold_distance),
        Vec<Real, 3>(
            threshold_distance, threshold_distance, threshold_distance));
    int n_i = coords_i.size(0);
    int n_j = coords_j.size(0);

    auto context = utility::cuda::context_from_stream(stream);

    mgpu::transform(
        [=] MGPU_DEVICE(int index) {
          int i = index / n_j;
          int j = index % n_j;

          if (j < i) {
            return;
          }

          if (tbox.contains(coords_i[i] - coords_j[j])) {
            f(i, j);
          }
        },
        n_i * n_j,
        context);
  }

  template <typename Real, typename Int, typename Func>
  static void forall_idx_pairs(
      Real threshold_distance,
      TView<Eigen::Matrix<Real, 3, 1>, 1, D> coords_i,
      TView<Eigen::Matrix<Real, 3, 1>, 1, D> coords_j,
      TView<Int, 1, D> coord_idx_i,
      TView<Int, 1, D> coord_idx_j,
      Func f,
      utility::cuda::CUDAStream stream) {
    const Eigen::AlignedBox<Real, 3> tbox(
        Vec<Real, 3>(
            -threshold_distance, -threshold_distance, -threshold_distance),
        Vec<Real, 3>(
            threshold_distance, threshold_distance, threshold_distance));
    int n_i = coord_idx_i.size(0);
    int n_j = coord_idx_j.size(0);

    auto context = utility::cuda::context_from_stream(stream);
    mgpu::transform(
        [=] MGPU_DEVICE(int index) {
          int i = index / n_j;
          int j = index % n_j;

          if (j < i) {
            return;
          }

          if (tbox.contains(
                  coords_i[coord_idx_i[i]] - coords_j[coord_idx_j[j]])) {
            f(i, j);
          }
        },
        n_i * n_j,
        context);
  }
};

}  // namespace common
}  // namespace score
}  // namespace tmol
