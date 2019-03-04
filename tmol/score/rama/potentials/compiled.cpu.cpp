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

template <
	tmol::Device D,
    typename Real,
    typename Int>
struct RamaDispatch {
  static auto f(
      TView<Vec<Real, 3>, 1, D> coords,
      TView<Vec<Int, 4>, 1, D> phi_indices,
      TView<Vec<Int, 4>, 1, D> psi_indices,
      TView<Int, 1, D> parameter_indices,
      TCollection<Real, 2, D> tables,
      TView<Vec<Real, 2>, 1, D> bbstarts,
      TView<Vec<Real, 2>, 1, D> bbsteps
  ) -> std::tuple<
          TPack<Real, 1, D>,
          TPack<CoordQuad, 1, D>,
          TPack<CoordQuad, 1, D> > {
	int num_Vs = phi_indices.size(0);

    auto Vs_t = TPack<Real, 1, D>::empty(num_Vs);
    auto Vs = Vs_t.view;

    auto dV_dphis_t = TPack<CoordQuad, 1, D>::empty(num_Vs);
    auto dV_dphis = dV_dphis_t.view;
    auto dV_dpsis_t = TPack<CoordQuad, 1, D>::empty(num_Vs);
    auto dV_dpsis = dV_dpsis_t.view;
    auto tableview = tables.view;

    auto func = ([=] EIGEN_DEVICE_FUNC(int i) {
      CoordQuad phicoords;
      phicoords.row(0) = coords[phi_indices[i][0]];
      phicoords.row(1) = coords[phi_indices[i][1]];
      phicoords.row(2) = coords[phi_indices[i][2]];
      phicoords.row(3) = coords[phi_indices[i][3]];
      CoordQuad psicoords;
      psicoords.row(0) = coords[psi_indices[i][0]];
      psicoords.row(1) = coords[psi_indices[i][1]];
      psicoords.row(2) = coords[psi_indices[i][2]];
      psicoords.row(3) = coords[psi_indices[i][3]];
      Int pari = parameter_indices[i];

      tie(Vs[i], dV_dphis[i], dV_dpsis[i]) = rama_V_dV<D, Real, Int>(
          phicoords,
          psicoords,
          tableview[pari],
          bbstarts[pari],
          bbsteps[pari]);
    });

	for (int idx=0; idx<num_Vs; ++idx) {
		func(idx);
	}
/*
*/
	return {Vs_t, dV_dphis_t, dV_dpsis_t};
  }
};

#undef CoordQuad


template struct RamaDispatch<tmol::Device::CPU,float,int32_t>;
template struct RamaDispatch<tmol::Device::CPU,double,int32_t>;
template struct RamaDispatch<tmol::Device::CPU,float,int64_t>;
template struct RamaDispatch<tmol::Device::CPU,double,int64_t>;

}  // namespace potentials
}  // namespace rama
}  // namespace score
}  // namespace tmol
