#include <Eigen/Core>

#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorCollection.h>
#include <tmol/score/common/geom.hh>
#include <tmol/numeric/bspline_compiled/bspline.hh>

#include <ATen/Tensor.h>

#include <tuple>

#include <pybind11/pybind11.h>

//#include "potentials.hh"

namespace tmol {
namespace score {
namespace rama {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

#define Real3 Vec<Real, 3>
#define Real2 Vec<Real, 2>

template <
	tmol::Device D,
    typename Real,
    typename Int>
struct RamaDispatch {
  static auto f(
      TCollection<Real, 2, D> tables,
      TView<Real2, 1, D> indices
  )
      -> TPack<Real, 1, D> {
	int num_Vs = indices.size(0);

    auto Vs_t = TPack<Real, 1, D>::empty(num_Vs);
    auto Vs = Vs_t.view;

    auto dV_dIs_t = TPack<Real2, 1, D>::empty(num_Vs);
    auto dV_dIs = dV_dIs_t.view;

    auto func = ([=] EIGEN_DEVICE_FUNC(int i) {
      tmol::score::common::tie(Vs[i], dV_dIs[i]) =
          numeric::bspline::ndspline<2, 3, D, Real, Int>::interpolate(tables.view[0],indices[i]);
    });

	for (int idx=0; idx<num_Vs; ++idx) {
		func(idx);
	}

	return Vs_t;
  }
};

template struct RamaDispatch<tmol::Device::CPU,float,int32_t>;
template struct RamaDispatch<tmol::Device::CPU,double,int32_t>;
template struct RamaDispatch<tmol::Device::CPU,float,int64_t>;
template struct RamaDispatch<tmol::Device::CPU,double,int64_t>;

}  // namespace potentials
}  // namespace rama
}  // namespace score
}  // namespace tmol
