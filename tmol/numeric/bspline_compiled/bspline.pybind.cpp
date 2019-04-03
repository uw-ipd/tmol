#include <pybind11/eigen.h>
#include <tmol/utility/tensor/pybind.h>
#include <torch/torch.h>

#include <tmol/numeric/bspline_compiled/bspline.hh>

namespace tmol {
namespace numeric {
namespace bspline {

template <tmol::Device D, typename Real, typename Int>
void bind(pybind11::module& m) {
  using namespace pybind11::literals;

  m.def(
      "computeCoeffs2",
      &ndspline<2, 3, D, Real, Int>::computeCoeffs,
      "data"_a);

  m.def(
      "interpolate2",
      py::overload_cast< TView<Real, 2, D>, TView<Eigen::Matrix<Real, 2, 1>, 1, D> >(
        &ndspline<2, 3, D, Real, Int>::interpolate_tv),
      "data"_a,
      "X"_a);

  m.def(
      "computeCoeffs3",
      &ndspline<3, 3, D, Real, Int>::computeCoeffs,
      "data"_a);

  m.def(
      "interpolate3",
      py::overload_cast< TView<Real, 3, D>, TView<Eigen::Matrix<Real, 3, 1>, 1, D> >(
        &ndspline<3, 3, D, Real, Int>::interpolate_tv),
      "data"_a,
      "X"_a);

  m.def(
      "computeCoeffs4",
      &ndspline<4, 3, D, Real, Int>::computeCoeffs,
      "data"_a);

  m.def(
      "interpolate4",
      py::overload_cast< TView<Real, 4, D>, TView<Eigen::Matrix<Real, 4, 1>, 1, D> >(
        &ndspline<4, 3, D, Real, Int>::interpolate_tv),
      "data"_a,
      "X"_a);

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace pybind11::literals;

  bind<tmol::Device::CPU,float,int32_t>(m);
  bind<tmol::Device::CPU,double,int32_t>(m);
}

}
}
}
