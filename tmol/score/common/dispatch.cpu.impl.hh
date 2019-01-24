#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <tuple>

#include <cppitertools/product.hpp>
#include <cppitertools/range.hpp>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>

#include "dispatch.hh"

namespace tmol {
namespace score {
namespace common {

using iter::product;
using iter::range;
using std::tie;
using std::tuple;
using tmol::new_tensor;
using tmol::TView;

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <>
struct ExhaustiveDispatch<tmol::Device::CPU> {
  static const tmol::Device D = tmol::Device::CPU;

  ExhaustiveDispatch(int n_i, int n_j) : n_i(n_i), n_j(n_j) {}

  template <typename Real>
  int scan(
      Real threshold_distance,
      TView<Vec<Real, 3>, 1, D> coords_i,
      TView<Vec<Real, 3>, 1, D> coords_j) {
    return n_i * n_j;
  }

  template <typename funct_t>
  void score(funct_t f) {
    int oind = 0;
    for (int i = 0; i < n_i; i++) {
      for (int j = 0; j < n_j; j++) {
        f(oind, i, j);
        oind++;
      }
    }
  }

  int n_i, n_j;
};

template <>
struct ExhaustiveTriuDispatch<tmol::Device::CPU> {
  static const tmol::Device D = tmol::Device::CPU;

  ExhaustiveTriuDispatch(int n_i, int n_j) : n_i(n_i), n_j(n_j) {}

  template <typename Real>
  int scan(
      Real threshold_distance,
      TView<Vec<Real, 3>, 1, D> coords_i,
      TView<Vec<Real, 3>, 1, D> coords_j) {
    int n_hit = 0;

    for (int i = 0; i < n_i; i++) {
      n_hit += n_j - i;
    }

    return n_hit;
  }

  template <typename funct_t>
  void score(funct_t f) {
    int oind = 0;
    for (int i = 0; i < n_i; i++) {
      for (int j = i; j < n_j; j++) {
        f(oind, i, j);
        oind++;
      }
    }
  }

  int n_i, n_j;
};

template <>
struct NaiveDispatch<tmol::Device::CPU> {
  static const tmol::Device D = tmol::Device::CPU;

  NaiveDispatch(int n_i, int n_j) : n_i(n_i), n_j(n_j), n_ind(0) {
    tie(inds_t, inds) = new_tensor<int, 2, D>({n_i * n_j, 2});
  }

  template <typename Real>
  int scan(
      Real threshold_distance,
      TView<Vec<Real, 3>, 1, D> coords_i,
      TView<Vec<Real, 3>, 1, D> coords_j) {
    const Eigen::AlignedBox<Real, 3> tbox(
        Vec<Real, 3>(
            -threshold_distance, -threshold_distance, -threshold_distance),
        Vec<Real, 3>(
            threshold_distance, threshold_distance, threshold_distance));

    n_ind = 0;

    for (int i = 0; i < n_i; ++i) {
      for (int j = 0; j < n_j; ++j) {
        if (tbox.contains(coords_i[i] - coords_j[j])) {
          inds[n_ind][0] = i;
          inds[n_ind][1] = j;
          n_ind++;
        }
      }
    }

    return n_ind;
  }

  template <typename funct_t>
  void score(funct_t f) {
    for (int o = 0; o < n_ind; o++) {
      f(o, inds[o][0], inds[o][1]);
    }
  }

  int n_i;
  int n_j;

  int n_ind;
  at::Tensor inds_t;
  TView<int, 2, D> inds;
};

template <>
struct NaiveTriuDispatch<tmol::Device::CPU> {
  static const tmol::Device D = tmol::Device::CPU;

  NaiveTriuDispatch(int n_i, int n_j) : n_i(n_i), n_j(n_j), n_ind(0) {
    tie(inds_t, inds) = new_tensor<int, 2, D>({n_i * n_j, 2});
  }

  template <typename Real>
  int scan(
      Real threshold_distance,
      TView<Vec<Real, 3>, 1, D> coords_i,
      TView<Vec<Real, 3>, 1, D> coords_j) {
    const Eigen::AlignedBox<Real, 3> tbox(
        Vec<Real, 3>(
            -threshold_distance, -threshold_distance, -threshold_distance),
        Vec<Real, 3>(
            threshold_distance, threshold_distance, threshold_distance));

    n_ind = 0;

    for (int i = 0; i < n_i; ++i) {
      for (int j = i; j < n_j; ++j) {
        if (tbox.contains(coords_i[i] - coords_j[j])) {
          inds[n_ind][0] = i;
          inds[n_ind][1] = j;
          n_ind++;
        }
      }
    }

    return n_ind;
  }

  template <typename funct_t>
  void score(funct_t f) {
    for (int o = 0; o < n_ind; o++) {
      f(o, inds[o][0], inds[o][1]);
    }
  }

  int n_i;
  int n_j;

  int n_ind;
  at::Tensor inds_t;
  TView<int, 2, D> inds;
};

}  // namespace common
}  // namespace score
}  // namespace tmol
