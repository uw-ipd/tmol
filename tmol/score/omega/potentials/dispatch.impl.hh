#include <Eigen/Core>

#include <tmol/utility/tensor/TensorCollection.h>
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
namespace omega {
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
struct OmegaDispatch {
  static auto f(
      TView<Vec<Real, 3>, 1, D> coords,
      TView<OmegaParameters<Real>, 1, D> omega_indices)
      -> std::tuple<TPack<Real, 1, D>, TPack<Vec<Real, 3>, 1, D>> {
    auto V_t = TPack<Real, 1, D>::empty({1});
    auto dV_dx_t = TPack<Vec<Real, 3>, 1, D>::empty({coords.size(0)});

    auto V = V_t.view;
    auto dV_dx = dV_dx_t.view;

    auto zero = [=] EIGEN_DEVICE_FUNC(int i) {
      if (i == 0) {
        V[i] = 0;
      }
      common::zero_array<D>::go((Real *) dV_dx.data(), i, dV_dx.size(0), 3);
    };
    Dispatch<D>::forall(std::max(1L, dV_dx.size(0)), zero);

    auto func = ([=] EIGEN_DEVICE_FUNC(int i) {
      CoordQuad omegacoords;
      for (int j = 0; j < 4; ++j) {
        omegacoords.row(j) = coords[(int)omega_indices[i].atoms[j]];
      }
      auto omega = omega_V_dV<D, Real, Int>(omegacoords, omega_indices[i].K);

      accumulate<D, Real>::add(V[0], common::get<0>(omega));
      for (int j = 0; j < 4; ++j) {
        accumulate<D, Vec<Real, 3>>::add(
            dV_dx[(int)omega_indices[i].atoms[j]],
            common::get<1>(omega).row(j));
      }
    });

    int num_Vs = omega_indices.size(0);
    Dispatch<D>::forall(num_Vs, func);

    return {V_t, dV_dx_t};
  }
};

#undef CoordQuad

}  // namespace potentials
}  // namespace omega
}  // namespace score
}  // namespace tmol
