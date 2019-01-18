#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <tuple>

#include <cppitertools/product.hpp>
#include <cppitertools/range.hpp>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>

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

struct TrivialDispatch {
  TrivialDispatch(float threshold_distance_, int n_i, int n_j)
      : threshold_distance(threshold_distance_) {
    tie(hits_t, hits) = new_tensor<uint8_t, 2>({n_i, n_j});
  }

  template <typename Real>
  int scan(TView<Vec<Real, 3>, 1> coords_i, TView<Vec<Real, 3>, 1> coords_j) {
    return hits.size(0) * hits.size(1);
  }

  template <typename funct_t>
  void score(funct_t f) {
    int oind = 0;
    for (auto [i, j] : product(range(hits.size(0)), range(hits.size(1)))) {
      f(oind, i, j);
      oind++;
    }
  }

  float threshold_distance;
  at::Tensor hits_t;
  TView<uint8_t, 2> hits;
};

struct NaiveDispatch {
  NaiveDispatch(float threshold_distance_, int n_i, int n_j)
      : threshold_distance(threshold_distance_) {
    tie(hits_t, hits) = new_tensor<int, 2>({n_i, n_j});
  }

  template <typename Real>
  int scan(TView<Vec<Real, 3>, 1> coords_i, TView<Vec<Real, 3>, 1> coords_j) {
    const Eigen::AlignedBox<Real, 3> tbox(
        Vec<Real, 3>(
            -threshold_distance, -threshold_distance, -threshold_distance),
        Vec<Real, 3>(
            threshold_distance, threshold_distance, threshold_distance));

    int oind = 0;

    for (auto [i, j] : product(range(hits.size(0)), range(hits.size(1)))) {
      if (tbox.contains(coords_i[i] - coords_j[j])) {
        hits[i][j] = oind;
        oind++;
      } else {
        hits[i][j] = -1;
      }
    }

    return oind;
  }

  template <typename funct_t>
  void score(funct_t f) {
    for (auto [i, j] : product(range(hits.size(0)), range(hits.size(1)))) {
      if (hits[i][j] >= 0) {
        f(hits[i][j], i, j);
      }
    }
  }

  float threshold_distance;
  at::Tensor hits_t;
  TView<int, 2> hits;
};

struct NaiveTriuDispatch {
  NaiveTriuDispatch(float threshold_distance_, int n_i, int n_j)
      : threshold_distance(threshold_distance_) {
    tie(hits_t, hits) = new_tensor<int, 2>({n_i, n_j});
  }

  template <typename Real>
  int scan(TView<Vec<Real, 3>, 1> coords_i, TView<Vec<Real, 3>, 1> coords_j) {
    const Eigen::AlignedBox<Real, 3> tbox(
        Vec<Real, 3>(
            -threshold_distance, -threshold_distance, -threshold_distance),
        Vec<Real, 3>(
            threshold_distance, threshold_distance, threshold_distance));

    int oind = 0;

    for (auto i : range(hits.size(0))) {
      for (auto j : range(i, hits.size(1))) {
        if (tbox.contains(coords_i[i] - coords_j[j])) {
          hits[i][j] = oind;
          oind++;
        } else {
          hits[i][j] = -1;
        }
      }
    }

    return oind;
  }

  template <typename funct_t>
  void score(funct_t f) {
    for (auto i : range(hits.size(0))) {
      for (auto j : range(i, hits.size(1))) {
        if (hits[i][j] >= 0) {
          f(hits[i][j], i, j);
        }
      }
    }
  }

  float threshold_distance;
  at::Tensor hits_t;
  TView<int, 2> hits;
};

}  // namespace common
}  // namespace score
}  // namespace tmol
