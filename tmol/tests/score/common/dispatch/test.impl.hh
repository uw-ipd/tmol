#pragma once

#include "test.hh"
#include "tmol/utility/cuda/stream.hh"
#include "tmol/utility/tensor/TensorPack.h"

namespace tmol {
template <template <Device> class Dispatch, Device D, typename Real>
auto DispatchTest<Dispatch, D, Real>::f(TView<Vec<Real, 3>, 1, D> coords)
    -> std::tuple<TPack<int64_t, 2, D>, TPack<float, 1, D>> {
  auto defstream = utility::cuda::get_default_stream();
  Dispatch<D> dispatcher(coords.size(0), coords.size(0));
  auto num_scores = dispatcher.scan(6.0, coords, coords, defstream);

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
}  // namespace tmol
