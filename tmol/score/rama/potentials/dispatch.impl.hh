#include <Eigen/Core>

#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/numeric/bspline_compiled/bspline.hh>
#include <tmol/score/common/geom.hh>

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/tuple_operators.hh>

#include <ATen/Tensor.h>

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
      TView<Vec<Int, 4>, 1, D> phi_indices,
      TView<Vec<Int, 4>, 1, D> psi_indices,
      TView<Int, 1, D> parameter_indices,
      TView<Real, 3, D> tables,
      TView<Vec<Real, 2>, 1, D> bbstarts,
      TView<Vec<Real, 2>, 1, D> bbsteps)
      -> std::tuple<TPack<Real, 1, D>, TPack<Vec<Real, 3>, 1, D>> {
    int num_Vs = phi_indices.size(0);

    auto V_t = TPack<Real, 1, D>::zeros({1});
    auto dV_dx_t = TPack<Vec<Real, 3>, 1, D>::zeros({coords.size(0)});

    auto V = V_t.view;
    auto dV_dx = dV_dx_t.view;

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
      auto rama = rama_V_dV<D, Real, Int>(
          phicoords, psicoords, tables[pari], bbstarts[pari], bbsteps[pari]);

      accumulate<D, Real>::add(V[0], common::get<0>(rama));
      for (int j = 0; j < 4; ++j) {
        accumulate<D, Vec<Real, 3>>::add(
            dV_dx[phi_indices[i][j]], common::get<1>(rama).row(j));
        accumulate<D, Vec<Real, 3>>::add(
            dV_dx[psi_indices[i][j]], common::get<2>(rama).row(j));
      }
    });

    Dispatch<D>::forall(num_Vs, func);

    return {V_t, dV_dx_t};
  }
};

#undef CoordQuad

}  // namespace potentials
}  // namespace rama
}  // namespace score
}  // namespace tmol
