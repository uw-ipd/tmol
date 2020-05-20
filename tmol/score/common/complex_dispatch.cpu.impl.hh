#pragma once

#include <Eigen/Core>

#include <tmol/utility/tensor/TensorAccessor.h>

namespace tmol {
namespace score {
namespace common {

template <tmol::Device D>
struct ComplexDispatch {
  template <typename Int, typename Func>
  static void forall(Int N, Func f) {
    for (int ii = 0; ii < N; ++ii) {
      f(ii);
    }
  }

  template <typename T, typename Func>
  static void exclusive_scan(TView<T, 1, D> vals, TView<T, 1, D> out, Func op) {
    assert(vals.size(0) == out.size(0));
    out[0] = T(0);
    for (int ii = 1; ii < vals.size(0); ++ii) {
      out[ii] = op(out[ii - 1], vals[ii - 1]);
    }
  }

  template <typename T, typename Func>
  static void inclusive_scan(TView<T, 1, D> vals, TView<T, 1, D> out, Func op) {
    assert(vals.size(0) == out.size(0));
    out[0] = vals[0];
    for (int ii = 1; ii < vals.size(0); ++ii) {
      out[ii] = op(out[ii - 1], vals[ii]);
    }
  }

  template <typename T, typename Func>
  static T exclusive_scan_w_final_val(
      TView<T, 1, D> vals, TView<T, 1, D> out, Func op) {
    assert(vals.size(0) == out.size(0));
    out[0] = T(0);
    for (int ii = 1; ii < vals.size(0); ++ii) {
      out[ii] = op(out[ii - 1], vals[ii - 1]);
    }
    return out[out.size(0) - 1] + vals[vals.size(0) - 1];
  }

  template <typename T, typename B, typename Func>
  static void exclusive_segmented_scan(
      TView<T, 1, D> vals,
      TView<B, 1, D> seg_start,
      TView<T, 1, D> out,
      Func op) {
    assert(vals.size(0) == out.size(0));
    for (int ii = 0; ii < vals.size(0); ++ii) {
      if (seg_start[ii]) {
        out[ii] = T(0);
      } else {
        out[ii] = op(out[ii - 1], vals[ii - 1]);
      }
    }
  }
};
}
}
}
