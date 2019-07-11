#pragma once

#include <tmol/utility/tensor/TensorAccessor.h>

namespace tmol {
namespace score {
namespace common {

template <tmol::Device D>
struct zero_array {
  template <class T>
  static void EIGEN_DEVICE_FUNC go(T* data, int i, int size0, int stride0) {
    if (D == tmol::Device::CPU) {
      for (int j = 0; j < stride0; ++j) {
        data[i * stride0 + j] = 0;
      }
    } else {
      for (int j = 0; j < stride0; ++j) {
        data[j * size0 + i] = 0;
      }
    }
  }
};

}  // namespace common
}  // namespace score
}  // namespace tmol
