#pragma once

#include <tmol/utility/tensor/native.hh>
#include "params_types.hh"

namespace tmol {

template <typename Int>
struct enable_tensor_view<
    io::details::compiled::HisAtomIndsInCanonicalOrdering<Int>> {
  static const bool enabled = enable_tensor_view<Int>::enabled;
  static const tmol::native::ScalarType scalar_type() {
    return enable_tensor_view<Int>::scalar_type();
  }
  static const int nconsumed_dims = 1;
  static const int consumed_dims(int i) {
    return (i == 0)
               ? sizeof(
                     io::details::compiled::HisAtomIndsInCanonicalOrdering<Int>)
                     / sizeof(Int)
               : 0;
  }

  typedef typename enable_tensor_view<Int>::PrimitiveType PrimitiveType;
};

}  // namespace tmol
