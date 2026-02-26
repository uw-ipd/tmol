#include <tmol/utility/tensor/TensorPack.h>
#include <cppitertools/range.hpp>

#include "hybrid.hh"

namespace tmol {
namespace tests {
namespace utility {
namespace cpp_extension {

template <typename Real>
struct sum_tensor<Real, tmol::Device::CPU> {
  static const tmol::Device D = tmol::Device::CPU;

  static at::Tensor f(tmol::TView<Real, 1, D> t) {
    auto v_t = tmol::TPack<Real, 1, D>::empty({1});
    auto v = v_t.view;

    v[0] = 0;
    for (int i = 0; i < t.size(0); i++) {
      v[0] += t[i];
    }

    return v_t.tensor;
  }
};

template struct sum_tensor<float, tmol::Device::CPU>;
template struct sum_tensor<double, tmol::Device::CPU>;
}  // namespace cpp_extension
}  // namespace utility
}  // namespace tests
}  // namespace tmol