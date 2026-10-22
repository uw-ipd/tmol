#pragma once

// enable_tensor_view for Eigen element types.
//
// Kept apart from TensorUtil.h so that the tensor layer -- and through it the
// device-side math that includes it -- does not pull in Eigen. Only the
// subsystems still built on Eigen (kinematics' 4x4 transforms, the b-spline
// setup code) need this header.

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorUtil.h>

namespace tmol {

// Eigen Matrix/Vector Conversions
// Matrix is defined as <T, M, N>, consume two [M, N] dimensions.
// Vector is defined as <T, M, 1>, consume one [M] dimension.
template <typename T, int M, int N>
struct enable_tensor_view<Eigen::Matrix<T, M, N>> {
  static const bool enabled = enable_tensor_view<T>::enabled;
  static const tmol::native::ScalarType scalar_type() {
    return enable_tensor_view<T>::scalar_type();
  }
  static constexpr int nconsumed_dims = ((N > 1) ? 2 : 1);
  static int consumed_dims(int i) {
    if (i == 0) {
      return M;
    } else if (i == 1 && N > 1) {
      return N;
    } else {
      return 0;
    }
  }
  typedef typename enable_tensor_view<T>::PrimitiveType PrimitiveType;
};

template <typename T, int N>
struct enable_tensor_view<Eigen::AlignedBox<T, N>> {
  static const bool enabled = enable_tensor_view<T>::enabled;
  static const tmol::native::ScalarType scalar_type() {
    return enable_tensor_view<T>::scalar_type();
  }
  static const int nconsumed_dims = 1;
  static int consumed_dims(int i) { return (i == 0) ? N : 0; }
  typedef typename enable_tensor_view<T>::PrimitiveType PrimitiveType;
};

}  // namespace tmol
