#pragma once

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorUtil.h>

namespace tmol {
namespace pack {
namespace compiled {

template <typename Real>
struct SimAParams {
  Real hitemp;
  Real lotemp;
  Real n_outer;
  Real n_inner_scale;
  
};

} // namespace compiled
} // namespace pack
} // namespace tmol


namespace tmol {

template <typename Real>
struct enable_tensor_view<pack::compiled::SimAParams<Real>> {
  static const bool enabled = enable_tensor_view<Real>::enabled;
  static const at::ScalarType scalar_type =
      enable_tensor_view<Real>::scalar_type;

  static const int nconsumed_dims = 1;
  static const int consumed_dims(int i) {
    return (i == 0) ? sizeof(pack::compiled::SimAParams<Real>)
                          / sizeof(Real)
                    : 0;
  }

  typedef typename enable_tensor_view<Real>::PrimitiveType PrimitiveType;

};

} // namespace tmol
