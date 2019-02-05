#pragma once

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorUtil.h>

namespace tmol {
namespace tests {
namespace utility {
namespace cpp_extension {

template <typename Real, tmol::Device D>
struct sum_tensor {
  static at::Tensor f(tmol::TView<Real, 1, D> t);
};

}
}
}
}