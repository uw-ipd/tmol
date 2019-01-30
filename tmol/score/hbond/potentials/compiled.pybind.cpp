#include <pybind11/eigen.h>
#include <tmol/utility/tensor/pybind.h>
#include <torch/torch.h>

#include <tmol/score/common/dispatch.hh>
#include <tmol/score/hbond/potentials/dispatch.hh>

namespace tmol {
namespace score {
namespace hbond {
namespace potentials {

template <tmol::Device Dev, typename Real, typename Int>
void bind_dispatch(pybind11::module& m) {
  using namespace pybind11::literals;

#define HBOND_PYARGS()                                                       \
  "D"_a, "H"_a, "donor_type"_a, "A"_a, "B"_a, "B0"_a, "acceptor_type"_a,     \
      "acceptor_class"_a, "acceptor_weight"_a, "donor_weight"_a,             \
      "AHdist_coeffs"_a, "AHdist_range"_a, "AHdist_bound"_a,                 \
      "cosBAH_coeffs"_a, "cosBAH_range"_a, "cosBAH_bound"_a,                 \
      "cosAHD_coeffs"_a, "cosAHD_range"_a, "cosAHD_bound"_a,                 \
      "hb_sp2_range_span"_a, "hb_sp2_BAH180_rise"_a, "hb_sp2_outer_width"_a, \
      "hb_sp3_softmax_fade"_a, "threshold_distance"_a

  m.def(
      "hbond_pair_score",
      &HBondDispatch<common::NaiveDispatch, Dev, Real, Int>::f,
      HBOND_PYARGS());

  m.def(
      "hbond_pair_score",
      &HBondDispatch<common::ExhaustiveDispatch, Dev, Real, Int>::f,
      HBOND_PYARGS());
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace pybind11::literals;

  bind_dispatch<tmol::Device::CPU, float, int32_t>(m);
  bind_dispatch<tmol::Device::CPU, float, int64_t>(m);
  bind_dispatch<tmol::Device::CPU, double, int32_t>(m);
  bind_dispatch<tmol::Device::CPU, double, int64_t>(m);

#ifdef WITH_CUDA
  bind_dispatch<tmol::Device::CUDA, float, int32_t>(m);
  bind_dispatch<tmol::Device::CUDA, double, int32_t>(m);
#endif
}
}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol
