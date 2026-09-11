#include <limits>
#include <tuple>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <tmol/score/dunbrack/potentials/potentials.hh>

template <typename Real>
auto measure_default_dihedral(Real default_angle) {
  Eigen::Vector4i atom_indices = Eigen::Vector4i::Constant(-1);
  Eigen::Matrix<Real, 4, 3> derivatives = Eigen::Matrix<Real, 4, 3>::Constant(
      std::numeric_limits<Real>::quiet_NaN());
  Real angle = std::numeric_limits<Real>::quiet_NaN();
  tmol::score::dunbrack::potentials::measure_dihedral_V_dV(
      tmol::TensorAccessor<Eigen::Matrix<Real, 3, 1>, 1, tmol::Device::CPU>(),
      atom_indices,
      default_angle,
      angle,
      derivatives);
  return std::make_tuple(angle, derivatives);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("measure_default_float", &measure_default_dihedral<float>);
  m.def("measure_default_double", &measure_default_dihedral<double>);
}
