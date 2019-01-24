#include <pybind11/eigen.h>
#include <torch/torch.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/tensor/pybind.h>

#include <tmol/score/common/dispatch.hh>

using std::tie;
using std::tuple;
using tmol::Device;
using tmol::TView;

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <typename Dispatch, Device D, typename Real>
auto dispatch(TView<Vec<Real, 3>, 1, D> coords)
    -> tuple<at::Tensor, at::Tensor> {
  using tmol::new_tensor;

  Dispatch dispatcher(6.0, coords.size(0), coords.size(0));
  auto num_scores = dispatcher.scan(coords, coords);

  at::Tensor ind_t;
  TView<int64_t, 2, D> ind;
  tie(ind_t, ind) = new_tensor<int64_t, 2, D>({num_scores, 2});

  at::Tensor score_t;
  TView<float, 1, D> score;
  tie(score_t, score) = new_tensor<float, 1, D>(num_scores);

  Real squared_threshold = 6.0 * 6.0;

  dispatcher.score([=](int o, int i, int j) mutable {
    ind[o][0] = i;
    ind[o][1] = j;
    if ((coords[i] - coords[j]).squaredNorm() < squared_threshold) {
      score[o] = 1;
    } else {
      score[o] = 0;
    }
  });

  return {ind_t, score_t};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace pybind11::literals;

  using tmol::score::common::TrivialDispatch;
  m.def(
      "trivial_dispatch",
      &dispatch<TrivialDispatch, Device::CPU, double>,
      "coords"_a);

  using tmol::score::common::NaiveDispatch;
  m.def(
      "naive_dispatch",
      &dispatch<NaiveDispatch, Device::CPU, double>,
      "coords"_a);
}
