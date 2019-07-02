#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <memory>

#include <moderngpu/kernel_compact.hxx>
#include <moderngpu/transform.hxx>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/score/common/tuple.hh>

#include "dispatch.hh"

namespace tmol {
namespace score {
namespace common {

inline mgpu::standard_context_t context_from_stream(
    at::cuda::CUDAStream* stream) {
  if (stream) {
    return mgpu::standard_context_t(stream->stream());
  } else {
    return mgpu::standard_context_t();
  }
}

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

  int _n_i, _n_j;
  std::unique_ptr<mgpu::standard_context_t> _context;

  ExhaustiveDispatch(int n_i, int n_j) : _n_i(n_i), _n_j(n_j){};

  template <typename Real>
  int scan(
      Real threshold_distance,
      TView<Vec<Real, 3>, 1, D> coords_i,
      TView<Vec<Real, 3>, 1, D> coords_j,
      at::cuda::CUDAStream* stream = nullptr) {
    _context.reset(new mgpu::standard_context_t(context_from_stream(stream)));
    return _n_i * _n_j;
  }

  template <typename funct_t>
  void score(funct_t f) {
    int n_i = _n_i;
    int n_j = _n_j;

    mgpu::transform(
        [=] MGPU_DEVICE(int index) {
          int i = index / n_j;
          int j = index % n_j;

          f(index, i, j);
        },
        n_i * n_j,
        *_context);
  }
};

template <>
struct ExhaustiveTriuDispatch<tmol::Device::CUDA> {
  static const tmol::Device D = tmol::Device::CUDA;

  int _n_i, _n_j;
  std::unique_ptr<mgpu::standard_context_t> _context;

  ExhaustiveTriuDispatch(int n_i, int n_j) : _n_i(n_i), _n_j(n_j){};

  template <typename Real>
  int scan(
      Real threshold_distance,
      TView<Vec<Real, 3>, 1, D> coords_i,
      TView<Vec<Real, 3>, 1, D> coords_j,
      at::cuda::CUDAStream* stream = nullptr) {
    _context.reset(new mgpu::standard_context_t(context_from_stream(stream)));
    int i = _n_i - 1;
    return (_n_i * _n_j) - (i * i + i) / 2;
  }

  template <typename funct_t>
  void score(funct_t f) {
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
        *_context);
  }
};

template <>
struct NaiveDispatch<tmol::Device::CUDA> {
  static const tmol::Device D = tmol::Device::CUDA;
  typedef mgpu::stream_compact_t<mgpu::empty_t> TransformCompact;

  int _n_i, _n_j;
  std::unique_ptr<mgpu::standard_context_t> _context;
  std::unique_ptr<TransformCompact> _compact;

  NaiveDispatch(int n_i, int n_j) : _n_i(n_i), _n_j(n_j){};

  template <typename Real>
  int scan(
      Real threshold_distance,
      TView<Vec<Real, 3>, 1, D> coords_i,
      TView<Vec<Real, 3>, 1, D> coords_j,
      at::cuda::CUDAStream* stream = nullptr) {
    _context.reset(new mgpu::standard_context_t(context_from_stream(stream)));
    _compact.reset(new TransformCompact(_n_i * _n_j, *_context));

    int n_j = _n_j;

    return _compact->upsweep([=] MGPU_DEVICE(int index) {
      Eigen::AlignedBox<Real, 3> tbox(
          Vec<Real, 3>(
              -threshold_distance, -threshold_distance, -threshold_distance),
          Vec<Real, 3>(
              threshold_distance, threshold_distance, threshold_distance));

      int i = index / n_j;
      int j = index % n_j;

      return tbox.contains(coords_i[i] - coords_j[j]);
    });
  }

  template <typename funct_t>
  void score(funct_t f) {
    int n_j = _n_j;

    _compact->downsweep([=] MGPU_DEVICE(int dest_index, int src_index) {
      int i = src_index / n_j;
      int j = src_index % n_j;

      f(dest_index, i, j);
    });
  }
};

template <>
struct NaiveTriuDispatch<tmol::Device::CUDA> {
  static const tmol::Device D = tmol::Device::CUDA;
  typedef mgpu::stream_compact_t<mgpu::empty_t> TransformCompact;

  int _n_i, _n_j;
  std::unique_ptr<mgpu::standard_context_t> _context;
  std::unique_ptr<TransformCompact> _compact;

  NaiveTriuDispatch(int n_i, int n_j) : _n_i(n_i), _n_j(n_j){};

  template <typename Real>
  int scan(
      Real threshold_distance,
      TView<Vec<Real, 3>, 1, D> coords_i,
      TView<Vec<Real, 3>, 1, D> coords_j,
      at::cuda::CUDAStream stream = nullptr) {
    _context.reset(new mgpu::standard_context_t(context_from_stream(stream)));
    _compact.reset(new TransformCompact(_n_i * _n_j, *_context));

    return _compact->upsweep([=, n_j = _n_j] MGPU_DEVICE(int index) {
      int i = index / n_j;
      int j = index % n_j;

      if (j < i) {
        return false;
      }

      Eigen::AlignedBox<Real, 3> tbox(
          Vec<Real, 3>(
              -threshold_distance, -threshold_distance, -threshold_distance),
          Vec<Real, 3>(
              threshold_distance, threshold_distance, threshold_distance));

      return tbox.contains(coords_i[i] - coords_j[j]);
    });
  }

  template <typename funct_t>
  void score(funct_t f) {
    _compact->downsweep(
        [=, n_j = _n_j] MGPU_DEVICE(int dest_index, int src_index) {
          int i = src_index / n_j;
          int j = src_index % n_j;

          f(dest_index, i, j);
        });
  }
};

}  // namespace common
}  // namespace score
}  // namespace tmol
