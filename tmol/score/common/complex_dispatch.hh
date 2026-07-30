#pragma once

#include <Eigen/Core>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/context_manager.hh>

namespace tmol {
namespace score {
namespace common {

template <tmol::Device D>
struct ComplexDispatch {
  template <typename Int, typename Func>
  static void forall(ContextManager& mgr, Int N, Func f);

  template <typename T, typename Func>
  static T reduce(ContextManager& mgr, TView<T, 1, D> vals, Func op);

  template <typename T, typename Func>
  static void exclusive_scan(
      ContextManager& mgr, TView<T, 1, D> vals, TView<T, 1, D> out, Func op);

  template <typename T, typename Func>
  static void inclusive_scan(
      ContextManager& mgr, TView<T, 1, D> vals, TView<T, 1, D> out, Func op);

  template <typename T, typename Func>
  static T exclusive_scan_w_final_val(
      ContextManager& mgr, TView<T, 1, D> vals, TView<T, 1, D> out, Func op);
  template <typename T, typename B, typename Func>
  static void exclusive_segmented_scan(
      ContextManager& mgr,
      TView<T, 1, D> vals,
      TView<B, 1, D>
          seg_start,  // n-segments length array of indices where segments start
      TView<T, 1, D> out,
      Func op);

  template <typename T, typename B, typename Func>
  static void inclusive_segmented_scan(
      ContextManager& mgr,
      TView<T, 1, D> vals,
      TView<B, 1, D>
          seg_start,  // n-segments length array of indices where segments start
      TView<T, 1, D> out,
      Func op);

  static void synchronize();
};
}  // namespace common
}  // namespace score
}  // namespace tmol
