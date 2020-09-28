#pragma once

#include <tmol/extern/moderngpu/operators.hxx>
#include "test.hh"
#include "tmol/utility/tensor/TensorPack.h"

namespace tmol {
template <template <Device> class Dispatch, Device D, typename Real>
auto DispatchTest<Dispatch, D, Real>::f(TView<Vec<Real, 3>, 1, D> coords)
    -> std::tuple<TPack<int64_t, 2, D>, TPack<float, 1, D>> {
  Dispatch<D> dispatcher(coords.size(0), coords.size(0));
  auto num_scores = dispatcher.scan(6.0, coords, coords);

  auto ind_t = TPack<int64_t, 2, D>::empty({num_scores, 2});
  auto score_t = TPack<float, 1, D>::empty({num_scores});

  auto ind = ind_t.view;
  auto score = score_t.view;

  Real squared_threshold = 6.0 * 6.0;

  dispatcher.score([=] EIGEN_DEVICE_FUNC(int o, int i, int j) {
    ind[o][0] = i;
    ind[o][1] = j;

    if ((coords[i] - coords[j]).squaredNorm() < squared_threshold) {
      score[o] = 1.0;
    } else {
      score[o] = 0.0;
    }
  });

  return {ind_t, score_t};
};

template <template <Device> class Dispatch, Device D, typename Int>
auto ComplexDispatchTest<Dispatch, D, Int>::f(
    TView<Int, 1, D> vals, TView<Int, 1, D> boundaries)
    -> std::tuple<
        TPack<Int, 1, D>,  // forall
        Int,               // reduce
        TPack<Int, 1, D>,  // exclusive_scan
        TPack<Int, 1, D>,  // inclusive_scan
        TPack<Int, 1, D>,  // exclusive_scan w/ final val
        Int,               // final val from above
        TPack<Int, 1, D>>  // exclusive seg scan
{
  int const n_entries = vals.size(0);

  auto res1_tp = TPack<Int, 1, D>::zeros({n_entries});
  auto res3_tp = TPack<Int, 1, D>::zeros({n_entries});
  auto res4_tp = TPack<Int, 1, D>::zeros({n_entries});
  auto res5_tp = TPack<Int, 1, D>::zeros({n_entries});
  auto res7_tp = TPack<Int, 1, D>::zeros({n_entries});

  auto res1 = res1_tp.view;
  Int res2(0);
  auto res3 = res3_tp.view;
  auto res4 = res4_tp.view;
  auto res5 = res5_tp.view;
  Int res6(0);
  auto res7 = res7_tp.view;

  auto double_func = [=] EIGEN_DEVICE_FUNC(int index) {
    res1[index] = vals[index] * 2;
  };

  Dispatch<D>::forall(n_entries, double_func);
  res2 = Dispatch<D>::reduce(vals, mgpu::plus_t<Int>());
  Dispatch<D>::exclusive_scan(vals, res3, mgpu::plus_t<Int>());
  Dispatch<D>::inclusive_scan(vals, res4, mgpu::plus_t<Int>());
  res6 =
      Dispatch<D>::exclusive_scan_w_final_val(vals, res5, mgpu::plus_t<Int>());
  Dispatch<D>::exclusive_segmented_scan(
      vals, boundaries, res7, mgpu::plus_t<Int>());
  return {res1_tp, res2, res3_tp, res4_tp, res5_tp, res6, res7_tp};
}

}  // namespace tmol
