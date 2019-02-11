#include <Eigen/Core>

#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/score/common/geom.hh>

#include "potentials.hh"

namespace tmol {
namespace score {
namespace cartbonded {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <
	tmol::Device D,
    typename Real,
    typename Int>
struct CartBondedLengthDispatch {
  static auto f(
    TView<Int, 2, D> atompair_indices,
    TView<Int, 1, D> parameter_indices,
    TView<Vec<Real, 3>, 1, D> coords,
    TView<Real, 1, D> K,
    TView<Real, 1, D> x0
  )
      -> std::tuple<
          TPack<Real, 1, D>,
          TPack<Vec<Real, 3>, 1, D>,
          TPack<Vec<Real, 3>, 1, D> > {
    auto num_Vs = parameter_indices.size(0);

    auto Vs_t = TPack<Real, 1, D>::empty({num_Vs});
    auto dV_dIs_t = TPack<Vec<Real, 3>, 1, D>::empty(num_Vs);
    auto dV_dJs_t = TPack<Vec<Real, 3>, 1, D>::empty(num_Vs);

    auto Vs = Vs_t.view;
    auto dV_dIs = dV_dIs_t.view;
    auto dV_dJs = dV_dJs_t.view;

    auto f_i = ([=] EIGEN_DEVICE_FUNC(int i) {
      Int ati = atompair_indices[i][0];
      Int atj = atompair_indices[i][1];
      Int pari = parameter_indices[i];
      tie(Vs[i],dV_dIs[i],dV_dJs[i]) = cblength_V_dV(
        coords[ati], coords[atj], K[ pari ], x0[ pari ] );
    });

    for (int i = 0; i < num_Vs; i++) {
      f_i(i);
    }

    return {Vs_t, dV_dIs_t, dV_dJs_t};
  }
};

template struct CartBondedLengthDispatch<tmol::Device::CPU,float,int32_t>;
template struct CartBondedLengthDispatch<tmol::Device::CPU,double,int32_t>;
template struct CartBondedLengthDispatch<tmol::Device::CPU,float,int64_t>;
template struct CartBondedLengthDispatch<tmol::Device::CPU,double,int64_t>;

}  // namespace potentials
}  // namespace cartbonded
}  // namespace score
}  // namespace tmol
