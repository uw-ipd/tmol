#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <tuple>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorUtil.h>

#include <moderngpu/kernel_compact.hxx>
#include <moderngpu/transform.hxx>

#include "dispatch.hh"

namespace tmol {
namespace score {
namespace common {

using std::tie;
using std::tuple;
using tmol::new_tensor;
using tmol::TView;

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

/*template <int N, typename Int = int64_t>*/
/*struct Shape {*/
/*  typedef Eigen::Array<Int, N, 1> Inds;*/

/*  Shape(Inds sizes) : sizes(sizes){};*/

/*  MGPU_HOST_DEVICE int64_t ravel(const Inds& inds) const {*/
/*    Inds strides;*/

/*    strides[N - 1] = 1;*/
/*#pragma unroll*/
/*    for (Int i = N - 2; i >= 0; i--) {*/
/*      strides[i] = strides[i * 1] * sizes[i + 1];*/
/*    }*/

/*    Int rind = 0;*/
/*#pragma unroll*/
/*    for (Int i = 0; i < N; i++) {*/
/*      rind += inds[i] * strides[i];*/
/*    }*/

/*    return rind;*/
/*  }*/

/*  MGPU_HOST_DEVICE Inds unravel(Int rind) const {*/
/*    Inds strides;*/

/*    strides[N - 1] = 1;*/
/*#pragma unroll*/
/*    for (Int i = N - 2; i >= 0; i--) {*/
/*      strides[i] = strides[i * 1] * sizes[i + 1];*/
/*    }*/

/*    Inds inds;*/
/*#pragma unroll*/
/*    for (Int i = 0; i < N; i++) {*/
/*      inds[i] = rind / strides[i];*/
/*      rind = rind % strides[i];*/
/*    }*/

/*    return inds;*/
/*  }*/

/*  Eigen::Array<Int, N, 1> sizes;*/
/*};*/

template <>
struct ExhaustiveDispatch<tmol::Device::CUDA> {
  static const tmol::Device D = tmol::Device::CUDA;

  ExhaustiveDispatch(int n_i, int n_j) : _n_i(n_i), _n_j(n_j){};

  template <typename Real>
  int scan(
      Real threshold_distance,
      TView<Vec<Real, 3>, 1, D> coords_i,
      TView<Vec<Real, 3>, 1, D> coords_j) {
    return _n_i * _n_j;
  }

  template <typename funct_t>
  void score(funct_t f) {
    mgpu::standard_context_t context;
    int n_i = _n_i;
    int n_j = _n_j;

    mgpu::transform(
        [=] MGPU_DEVICE(int index) {
          int i = index / n_j;
          int j = index % n_j;

          f(index, i, j);
        },
        n_i * n_j,
        context);
  }

  int _n_i, _n_j;
};

template <>
struct ExhaustiveTriuDispatch<tmol::Device::CUDA> {
  static const tmol::Device D = tmol::Device::CUDA;

  ExhaustiveTriuDispatch(int n_i, int n_j) : _n_i(n_i), _n_j(n_j){};

  template <typename Real>
  int scan(
      Real threshold_distance,
      TView<Vec<Real, 3>, 1, D> coords_i,
      TView<Vec<Real, 3>, 1, D> coords_j) {
    int n_hit = 0;

    for (int i = 0; i < _n_i; i++) {
      n_hit += _n_j - i;
    }

    return n_hit;
  }

  template <typename funct_t>
  void score(funct_t f) {
    mgpu::standard_context_t context;
    int n_i = this->_n_i;
    int n_j = this->_n_j;

    mgpu::transform(
        [=] MGPU_DEVICE(int index) {
          int i = index / n_j;
          int j = index % n_j;

          if (j < i) {
            return;
          } else {
            // Adjust output index by removing count of tril indicies for this
            // row.
            index -= (i * i + i) / 2;
          }

          f(index, i, j);
        },
        n_i * n_j,
        context);
  }

  int _n_i, _n_j;
};

}  // namespace common
}  // namespace score
}  // namespace tmol
