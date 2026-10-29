#pragma once

// Binds a view-generic kernel context to tmol::TView.
//
// Contexts are templated on a two-parameter view so one field list serves
// every backend. TView carries its device as a third parameter, so this
// adapts it; a Metal translation unit binds the same contexts to
// tmol::portable::MetalView instead and never includes this file.

#include <tmol/utility/tensor/TensorAccessor.h>

namespace tmol {
namespace score {
namespace common {

template <tmol::Device D>
struct TViewBinder {
  template <typename T, int N>
  using View = TView<T, N, D>;
};

}  // namespace common
}  // namespace score
}  // namespace tmol
