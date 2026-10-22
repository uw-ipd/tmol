#pragma once

#include <tmol/utility/tensor/native.hh>
#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include "params_types.hh"

namespace tmol {

template <typename Int>
struct enable_tensor_view<kinematics::KinForestParams<Int>> {
  static const bool enabled = enable_tensor_view<Int>::enabled;
  static const tmol::native::ScalarType scalar_type() {
    return enable_tensor_view<Int>::scalar_type();
  }

  static const int nconsumed_dims = 1;
  static const int consumed_dims(int i) {
    return (i == 0) ? sizeof(kinematics::KinForestParams<Int>) / sizeof(Int)
                    : 0;
  }

  typedef typename enable_tensor_view<Int>::PrimitiveType PrimitiveType;
};

template <typename Int>
struct enable_tensor_view<kinematics::KinForestGenData<Int>> {
  static const bool enabled = enable_tensor_view<Int>::enabled;
  static const tmol::native::ScalarType scalar_type() {
    enable_tensor_view<Int>::scalar_type();
  }

  static const int nconsumed_dims = 1;
  static const int consumed_dims(int i) {
    return (i == 0) ? sizeof(kinematics::KinForestGenData<Int>) / sizeof(Int)
                    : 0;
  }

  typedef typename enable_tensor_view<Int>::PrimitiveType PrimitiveType;
};

}  // namespace tmol
