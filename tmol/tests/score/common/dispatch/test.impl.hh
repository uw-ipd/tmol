#pragma once

#include "test.hh"

template <template <Device> class Dispatch, Device D, typename Real>
auto DispatchTest<Dispatch, D, Real>::f(TView<Vec<Real, 3>, 1, D> coords)
    -> tuple<at::Tensor, at::Tensor> {
  using tmol::new_tensor;

  Dispatch<D> dispatcher(coords.size(0), coords.size(0));
  auto num_scores = dispatcher.scan(6.0, coords, coords);

  at::Tensor ind_t;
  TView<int64_t, 2, D> ind;
  tie(ind_t, ind) = new_tensor<int64_t, 2, D>({num_scores, 2});

  at::Tensor score_t;
  TView<float, 1, D> score;
  tie(score_t, score) = new_tensor<float, 1, D>(num_scores);

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
