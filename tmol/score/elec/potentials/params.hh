#pragma once

#include <tmol/utility/tensor/native.hh>
#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include "params_types.hh"

namespace tmol {

template <typename Real>
struct enable_tensor_view<score::elec::potentials::ElecGlobalParams<Real>> {
  static const bool enabled = enable_tensor_view<Real>::enabled;
  static const tmol::native::ScalarType scalar_type() {
    return enable_tensor_view<Real>::scalar_type();
  }

  static const int nconsumed_dims = 1;
  static const int consumed_dims(int i) {
    return (i == 0) ? sizeof(score::elec::potentials::ElecGlobalParams<Real>)
                          / sizeof(Real)
                    : 0;
  }

  typedef typename enable_tensor_view<Real>::PrimitiveType PrimitiveType;
};

}  // namespace tmol
