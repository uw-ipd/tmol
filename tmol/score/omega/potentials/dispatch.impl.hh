#include <Eigen/Core>

#include <tmol/utility/tensor/TensorCollection.h>
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
      TView<Vec<Real, 3>, 2, D> coords,
      TView<OmegaParameters<Real>, 2, D> omega_indices)
      -> std::tuple<TPack<Real, 1, D>, TPack<Vec<Real, 3>, 2, D>> {
    int const nstacks = coords.size(0);
    auto V_t = TPack<Real, 1, D>::zeros({nstacks});
    auto dV_dx_t = TPack<Vec<Real, 3>, 2, D>::zeros({nstacks, coords.size(1)});

    auto V = V_t.view;
    auto dV_dx = dV_dx_t.view;

    auto func = ([=] EIGEN_DEVICE_FUNC(int stack, int i) {
      if (omega_indices[stack][i].atoms[0] >= 0) {
        CoordQuad omegacoords;
        for (int j = 0; j < 4; ++j) {
          omegacoords.row(j) =
              coords[stack][(int)omega_indices[stack][i].atoms[j]];
        }
        auto omega =
            omega_V_dV<D, Real, Int>(omegacoords, omega_indices[stack][i].K);

        accumulate<D, Real>::add(V[stack], common::get<0>(omega));
        for (int j = 0; j < 4; ++j) {
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[stack][(int)omega_indices[stack][i].atoms[j]],
              common::get<1>(omega).row(j));
        }
      }
    });

    int num_Vs = omega_indices.size(1);
    Dispatch<D>::forall_stacks(nstacks, num_Vs, func);

    return {V_t, dV_dx_t};
  }
};

#undef CoordQuad

}  // namespace potentials
}  // namespace omega
}  // namespace score
}  // namespace tmol
