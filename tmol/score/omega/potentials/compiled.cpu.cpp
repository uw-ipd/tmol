#include <Eigen/Core>

#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorCollection.h>
#include <tmol/score/common/geom.hh>
#include <tmol/numeric/bspline_compiled/bspline.hh>

#include <ATen/Tensor.h>

#include <tuple>

#include <pybind11/pybind11.h>

#include "potentials.hh"

namespace tmol {
namespace score {
namespace rama {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

#define CoordQuad Eigen::Matrix<Real, 4, 3>

template <tmol::Device D, typename Real, typename Int>
struct OmegaDispatch {
  static auto f(
      TView<Vec<Real, 3>, 1, D> coords,
      TView<Vec<Int, 4>, 1, D> omega_indices,
      Real K) -> std::tuple<TPack<Real, 1, D>, TPack<CoordQuad, 1, D> > {
    int num_Vs = omega_indices.size(0);

    auto Vs_t = TPack<Real, 1, D>::empty(num_Vs);
    auto Vs = Vs_t.view;

    auto dV_domegas_t = TPack<CoordQuad, 1, D>::empty(num_Vs);
    auto dV_domegas = dV_domegas_t.view;

    auto func = ([=] EIGEN_DEVICE_FUNC(int i) {
      CoordQuad omegacoords;
      omegacoords.row(0) = coords[omega_indices[i][0]];
      omegacoords.row(1) = coords[omega_indices[i][1]];
      omegacoords.row(2) = coords[omega_indices[i][2]];
      omegacoords.row(3) = coords[omega_indices[i][3]];
      tie(Vs[i], dV_domegas[i]) = omega_V_dV<D, Real, Int>(
          omegacoords, K);
    });

	for (int idx=0; idx<num_Vs; ++idx) {
		func(idx);
	}

	return {Vs_t, dV_domegas_t};
  }
};

#undef CoordQuad


template struct OmegaDispatch<tmol::Device::CPU,float,int32_t>;
template struct OmegaDispatch<tmol::Device::CPU,double,int32_t>;

}  // namespace potentials
}  // namespace rama
}  // namespace score
}  // namespace tmol
