#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cppitertools/product.hpp>
#include <cppitertools/range.hpp>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorUtil.h>

#include <tmol/score/common/tuple.hh>

#include "dispatch.hh"

namespace tmol {
namespace score {
namespace common {

using iter::product;
using iter::range;
using tmol::TView;

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <>
struct ExhaustiveDispatch<tmol::Device::CPU> {
  static const tmol::Device D = tmol::Device::CPU;

  int n_i, n_j;

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
};

template <>
struct ExhaustiveTriuDispatch<tmol::Device::CPU> {
  static const tmol::Device D = tmol::Device::CPU;

  int n_i, n_j;

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
};

// template <>
// struct NaiveDispatch<tmol::Device::CPU> {
//   static const tmol::Device D = tmol::Device::CPU;
//
//   int n_i;
//   int n_j;
//
//   int n_ind;
//   TPack<int, 2, D> inds_t;
//
//   NaiveDispatch(int n_i, int n_j)
//       : n_i(n_i),
//         n_j(n_j),
//         n_ind(0),
//         inds_t(TPack<int, 2, D>::empty({n_i * n_j, 2})) {}
//
//   template <typename Real>
//   int scan(
//       Real threshold_distance,
//       TView<Vec<Real, 3>, 1, D> coords_i,
//       TView<Vec<Real, 3>, 1, D> coords_j) {
//     const Eigen::AlignedBox<Real, 3> tbox(
//         Vec<Real, 3>(
//             -threshold_distance, -threshold_distance, -threshold_distance),
//         Vec<Real, 3>(
//             threshold_distance, threshold_distance, threshold_distance));
//
//     n_ind = 0;
//
//     for (int i = 0; i < n_i; ++i) {
//       for (int j = 0; j < n_j; ++j) {
//         if (tbox.contains(coords_i[i] - coords_j[j])) {
//           inds_t.view[n_ind][0] = i;
//           inds_t.view[n_ind][1] = j;
//           n_ind++;
//         }
//       }
//     }
//
//     return n_ind;
//   }
//
//   template <typename funct_t>
//   void score(funct_t f) {
//     for (int o = 0; o < n_ind; o++) {
//       f(o, inds_t.view[o][0], inds_t.view[o][1]);
//     }
//   }
// };

// template <>
// struct NaiveTriuDispatch<tmol::Device::CPU> {
//   static const tmol::Device D = tmol::Device::CPU;
//
//   int n_i;
//   int n_j;
//
//   int n_ind;
//   TPack<int, 2, D> inds_t;
//
//   NaiveTriuDispatch(int n_i, int n_j)
//       : n_i(n_i),
//         n_j(n_j),
//         n_ind(0),
//         inds_t(TPack<int, 2, D>::empty({n_i * n_j, 2})) {}
//
//   template <typename Real>
//   int scan(
//       Real threshold_distance,
//       TView<Vec<Real, 3>, 1, D> coords_i,
//       TView<Vec<Real, 3>, 1, D> coords_j) {
//     const Eigen::AlignedBox<Real, 3> tbox(
//         Vec<Real, 3>(
//             -threshold_distance, -threshold_distance, -threshold_distance),
//         Vec<Real, 3>(
//             threshold_distance, threshold_distance, threshold_distance));
//
//     n_ind = 0;
//
//     for (int i = 0; i < n_i; ++i) {
//       for (int j = i; j < n_j; ++j) {
//         if (tbox.contains(coords_i[i] - coords_j[j])) {
//           inds_t.view[n_ind][0] = i;
//           inds_t.view[n_ind][1] = j;
//           n_ind++;
//         }
//       }
//     }
//
//     return n_ind;
//   }
//
//   template <typename funct_t>
//   void score(funct_t f) {
//     for (int o = 0; o < n_ind; o++) {
//       f(o, inds_t.view[o][0], inds_t.view[o][1]);
//     }
//   }
// };

}  // namespace common
}  // namespace score
}  // namespace tmol
