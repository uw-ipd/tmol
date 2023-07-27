#pragma once

#include <Eigen/Core>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorUtil.h>

namespace tmol {
namespace score {
namespace dunbrack {
namespace potentials {

template <typename T, int N>
using Vec = Eigen::Matrix<T, N, 1>;

template <typename Real>
struct DunbrackParameters {
  // Real K;
};

template <typename Real>
struct DunbrackGlobalParams {
  // Real K;
};

template <typename Real, tmol::Device D>
struct DunbrackGlobalParamTensors {
  // TView<Real, 1, D> K;

  template <typename Idx>
  auto operator[](Idx i) const {
    // return DunbrackGlobalParams<Real>{K[i]};
    return DunbrackGlobalParams<Real>{};
  }
};

}  // namespace potentials
}  // namespace dunbrack
}  // namespace score
}  // namespace tmol

namespace tmol {

template <typename Real>
struct enable_tensor_view<
    score::dunbrack::potentials::DunbrackParameters<Real>> {
  static const bool enabled = enable_tensor_view<Real>::enabled;
  static const at::ScalarType scalar_type() {
    return enable_tensor_view<Real>::scalar_type();
  }

  static const int nconsumed_dims = 1;
  static const int consumed_dims(int i) {
    return (i == 0)
               ? sizeof(score::dunbrack::potentials::DunbrackParameters<Real>)
                     / sizeof(Real)
               : 0;
  }

  typedef typename enable_tensor_view<Real>::PrimitiveType PrimitiveType;
};

template <typename Real>
struct enable_tensor_view<
    score::dunbrack::potentials::DunbrackGlobalParams<Real>> {
  static const bool enabled = enable_tensor_view<Real>::enabled;
  static const at::ScalarType scalar_type() {
    return enable_tensor_view<Real>::scalar_type();
  }

  static const int nconsumed_dims = 1;
  static const int consumed_dims(int i) {
    return (i == 0)
               ? sizeof(score::dunbrack::potentials::DunbrackGlobalParams<Real>)
                     / sizeof(Real)
               : 0;
  }

  typedef typename enable_tensor_view<Real>::PrimitiveType PrimitiveType;
};

}  // namespace tmol
