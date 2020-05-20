#pragma once

#include <Eigen/Core>

#include <tmol/utility/tensor/TensorAccessor.h>

namespace tmol {
namespace score {
namespace common {

template <tmol::Device D>
struct ComplexDispatch {
  template <typename Int, typename Func>
  static void forall(Int N, Func f);

  template <typename T, typename Func>
  static void exclusive_scan(TView<T, 1, D> vals, TView<T, 1, D> out, Func op);

  template <typename T, typename Func>
  static T exclusive_scan_w_final_val(
      TView<T, 1, D> vals, TView<T, 1, D> out, Func op);
};
}
}
}
