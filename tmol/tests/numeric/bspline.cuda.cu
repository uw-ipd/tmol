#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <moderngpu/transform.hxx>

#include <tmol/numeric/bspline_compiled/bspline.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/pybind.h>

namespace {

template <std::size_t N, typename Real>
std::tuple<at::Tensor, at::Tensor> interpolate_cuda(
    tmol::TView<Real, N, tmol::Device::CUDA> coeffs,
    tmol::TView<Eigen::Matrix<Real, N, 1>, 1, tmol::Device::CUDA> points) {
  auto values_t =
      tmol::TPack<Real, 1, tmol::Device::CUDA>::empty({points.size(0)});
  auto derivatives_t =
      tmol::TPack<Eigen::Matrix<Real, N, 1>, 1, tmol::Device::CUDA>::empty(
          {points.size(0)});
  // Capture the views, not the packs: a TPack holds an at::Tensor, which a
  // device lambda cannot take by value. Same shape as geom.cu.
  auto values = values_t.view;
  auto derivatives = derivatives_t.view;

  mgpu::standard_context_t context;
  mgpu::transform(
      [=] MGPU_LAMBDA(int i) {
        tmol::score::common::tie(values[i], derivatives[i]) =
            tmol::numeric::bspline::
                ndspline<N, 3, tmol::Device::CUDA, Real, int32_t>::interpolate(
                    coeffs, points[i]);
      },
      points.size(0),
      context);

  return std::make_tuple(values_t.tensor, derivatives_t.tensor);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  using namespace pybind11::literals;
  module.def(
      "interpolate3", &interpolate_cuda<3, float>, "coeffs"_a, "points"_a);
  module.def(
      "interpolate4", &interpolate_cuda<4, float>, "coeffs"_a, "points"_a);
}
