#pragma once

// dispatch.mps.impl.hh
//
// ExhaustiveDispatch<Device::MPS> and ExhaustiveTriuDispatch<Device::MPS>.
//
// Phase-1 implementation: serial CPU loops over unified-memory MPS tensors,
// identical to the CPU specialisation.  Replace the score() body with Metal
// kernel calls in Phase 2.

#ifdef __NVCC__
error_this_file_must_not_be_compiled_by_nvcc();
#endif

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/score/common/tuple.hh>

#include "dispatch.hh"

namespace tmol {
namespace score {
namespace common {

using tmol::TView;

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <>
struct ExhaustiveDispatch<tmol::Device::MPS> {
  static const tmol::Device D = tmol::Device::MPS;

  int n_i, n_j;

  ExhaustiveDispatch(int n_i, int n_j) : n_i(n_i), n_j(n_j) {}

  template <typename Real>
  int scan(
      Real /*threshold_distance*/,
      TView<Vec<Real, 3>, 1, D> /*coords_i*/,
      TView<Vec<Real, 3>, 1, D> /*coords_j*/) {
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
struct ExhaustiveTriuDispatch<tmol::Device::MPS> {
  static const tmol::Device D = tmol::Device::MPS;

  int n_i, n_j;

  ExhaustiveTriuDispatch(int n_i, int n_j) : n_i(n_i), n_j(n_j) {}

  template <typename Real>
  int scan(
      Real /*threshold_distance*/,
      TView<Vec<Real, 3>, 1, D> /*coords_i*/,
      TView<Vec<Real, 3>, 1, D> /*coords_j*/) {
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

}  // namespace common
}  // namespace score
}  // namespace tmol
