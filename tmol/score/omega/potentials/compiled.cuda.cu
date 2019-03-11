#include <Eigen/Core>

#include <tmol/utility/tensor/TensorCollection.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/numeric/bspline_compiled/bspline.hh>
#include <tmol/score/common/geom.hh>

#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/tuple_operators.hh>

#include <moderngpu/kernel_compact.hxx>
#include <moderngpu/transform.hxx>

#include <ATen/Tensor.h>

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
      tie(Vs[i], dV_domegas[i]) = omega_V_dV<D, Real, Int>(omegacoords, K);
    });

    mgpu::standard_context_t context;
    mgpu::transform([=] MGPU_DEVICE(int idx) { func(idx); }, num_Vs, context);

    return {Vs_t, dV_domegas_t};
  }
};

#undef CoordQuad

template struct OmegaDispatch<tmol::Device::CUDA, float, int32_t>;
template struct OmegaDispatch<tmol::Device::CUDA, double, int32_t>;

}  // namespace potentials
}  // namespace rama
}  // namespace score
}  // namespace tmol
