#include <Eigen/Core>

#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/numeric/bspline_compiled/bspline.hh>
#include <tmol/score/common/geom.hh>

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/tuple_operators.hh>

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
      TView<Vec<Real, 3>, 2, D> coords,
      TView<RamaParameters<Int>, 2, D> params,
      TView<Real, 3, D> tables,
      TView<RamaTableParams<Real>, 1, D> table_params)
      -> std::tuple<TPack<Real, 1, D>, TPack<Vec<Real, 3>, 2, D>> {
    int const nstacks = coords.size(0);
    auto V_t = TPack<Real, 1, D>::zeros({nstacks});
    auto dV_dx_t = TPack<Vec<Real, 3>, 2, D>::zeros({nstacks, coords.size(1)});

    auto V = V_t.view;
    auto dV_dx = dV_dx_t.view;

    auto func = ([=] EIGEN_DEVICE_FUNC(int ind) {
      int stack = ind / params.size(1);
      int i = ind - stack * params.size(1);

      // if stacks are of different size, then mark the entries that
      // should not be evaluated with -1
      Int idx = params[stack][i].table_index;
      if (idx == -1) return;

      CoordQuad phicoords;
      CoordQuad psicoords;
      for (int j = 0; j < 4; ++j) {
        phicoords.row(j) = coords[stack][params[stack][i].phis[j]];
        psicoords.row(j) = coords[stack][params[stack][i].psis[j]];
      }

      auto rama = rama_V_dV<D, Real, Int>(
          phicoords,
          psicoords,
          tables[idx],
          Eigen::Map<Vec<Real, 2>>(table_params[idx].bbstarts),
          Eigen::Map<Vec<Real, 2>>(table_params[idx].bbsteps));

      accumulate<D, Real>::add(V[stack], common::get<0>(rama));
      for (int j = 0; j < 4; ++j) {
        accumulate<D, Vec<Real, 3>>::add(
            dV_dx[stack][params[stack][i].phis[j]],
            common::get<1>(rama).row(j));
        accumulate<D, Vec<Real, 3>>::add(
            dV_dx[stack][params[stack][i].psis[j]],
            common::get<2>(rama).row(j));
      }
    });

    int num_Vs = params.size(1);
    Dispatch<D>::forall(nstacks * num_Vs, func);

    return {V_t, dV_dx_t};
  }
};

#undef CoordQuad

}  // namespace potentials
}  // namespace rama
}  // namespace score
}  // namespace tmol
