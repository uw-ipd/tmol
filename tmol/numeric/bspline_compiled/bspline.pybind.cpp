#include <pybind11/eigen.h>
#include <tmol/utility/tensor/pybind.h>
#include <torch/torch.h>

#include <tmol/numeric/bspline_compiled/bspline.hh>

namespace tmol {
namespace numeric {
namespace bspline {

template <typename Real, typename Int>
void bind(pybind11::module& m) {
  using namespace pybind11::literals;

  m.def(
      "computeCoeffs2",
      &ndspline<2, 3, tmol::Device::CPU, Real, Int>::computeCoeffs,
      "data"_a);

  m.def(
      "interpolate2",
      &ndspline<2, 3, tmol::Device::CPU, Real, Int>::interpolate,
      "data"_a,
      "X"_a);

  m.def(
      "computeCoeffs3",
      &ndspline<3, 3, tmol::Device::CPU, Real, Int>::computeCoeffs,
      "x"_a);

  m.def(
      "interpolate3",
      &ndspline<3, 3, tmol::Device::CPU, Real, Int>::interpolate,
      "data"_a,
      "X"_a);

  m.def(
      "computeCoeffs4",
      &ndspline<4, 3, tmol::Device::CPU, Real, Int>::computeCoeffs,
      "x"_a);

  m.def(
      "interpolate4",
      &ndspline<4, 3, tmol::Device::CPU, Real, Int>::interpolate,
      "data"_a,
      "X"_a);

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace pybind11::literals;

  bind<float,int32_t>(m);
  bind<double,int32_t>(m);
  bind<float,int64_t>(m);
  bind<double,int64_t>(m);
}

}
}
}