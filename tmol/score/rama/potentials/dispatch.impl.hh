#include <Eigen/Core>

#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/numeric/bspline_compiled/bspline.hh>
#include <tmol/score/common/geom.hh>

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/tuple_operators.hh>
#include <tmol/score/common/zero.hh>

#include <ATen/Tensor.h>

#include "params.hh"
#include "potentials.hh"

namespace tmol {
namespace score {
namespace rama {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

#define CoordQuad Eigen::Matrix<Real, 4, 3>

template <
    template <tmol::Device>
    class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct RamaDispatch {
  static auto f(
      TView<Vec<Real, 3>, 1, D> coords,
      TView<RamaParameters<Int>, 1, D> params,
      TView<Real, 3, D> tables,
      TView<RamaTableParams<Real>, 1, D> table_params)
      -> std::tuple<TPack<Real, 1, D>, TPack<Vec<Real, 3>, 1, D>> {
    auto V_t = TPack<Real, 1, D>::empty({1});
    auto dV_dx_t = TPack<Vec<Real, 3>, 1, D>::empty({coords.size(0)});

    auto V = V_t.view;
    auto dV_dx = dV_dx_t.view;

    auto zero = [=] EIGEN_DEVICE_FUNC(int i) {
      if (i == 0) {
        V[i] = 0;
      }
      if (i < dV_dx.size(0)) {
	common::zero_array<D>::go((Real *) dV_dx.data(), i, dV_dx.size(0), 3);
        // for (int j = 0; j < 3; ++j) {
        //   dV_dx[i](j) = 0;
        // }
      }
    };
    Dispatch<D>::forall(std::max(1L, coords.size(0)), zero);

    auto func = ([=] EIGEN_DEVICE_FUNC(int i) {
      CoordQuad phicoords;
      CoordQuad psicoords;
      for (int j = 0; j < 4; ++j) {
        phicoords.row(j) = coords[params[i].phis[j]];
        psicoords.row(j) = coords[params[i].psis[j]];
      }

      Int idx = params[i].table_index;

      auto rama = rama_V_dV<D, Real, Int>(
          phicoords,
          psicoords,
          tables[idx],
          Eigen::Map<Vec<Real, 2>>(table_params[idx].bbstarts),
          Eigen::Map<Vec<Real, 2>>(table_params[idx].bbsteps));

      accumulate<D, Real>::add(V[0], common::get<0>(rama));
      for (int j = 0; j < 4; ++j) {
        accumulate<D, Vec<Real, 3>>::add(
            dV_dx[params[i].phis[j]], common::get<1>(rama).row(j));
        accumulate<D, Vec<Real, 3>>::add(
            dV_dx[params[i].psis[j]], common::get<2>(rama).row(j));
      }
    });

    int num_Vs = params.size(0);
    Dispatch<D>::forall(num_Vs, func);

    return {V_t, dV_dx_t};
  }
};

#undef CoordQuad

}  // namespace potentials
}  // namespace rama
}  // namespace score
}  // namespace tmol
