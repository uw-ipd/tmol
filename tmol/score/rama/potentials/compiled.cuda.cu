#include <Eigen/Core>

#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/numeric/bspline_compiled/bspline.hh>
#include <tmol/score/common/geom.hh>

#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/tuple_operators.hh>

#include <moderngpu/kernel_compact.hxx>
#include <moderngpu/transform.hxx>

#include <ATen/Tensor.h>

//#include "potentials.hh"

namespace tmol {
namespace score {
namespace rama {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

#define Real3 Vec<Real, 3>
#define Real2 Vec<Real, 2>

template <tmol::Device D, typename Real, typename Int>
struct RamaDispatch {
  static auto f(TViewCollection<Real, 2, D> tables, TView<Real2, 1, D> indices)
      -> TPack<Real, 1, D> {
    int num_Vs = indices.size(0);

    auto Vs_t = TPack<Real, 1, D>::empty(num_Vs);
    auto Vs = Vs_t.view;

    auto dV_dIs_t = TPack<Real2, 1, D>::empty(num_Vs);
    auto dV_dIs = dV_dIs_t.view;

    auto func = ([=] __device__(int i) {
      // tmol::score::common::tie(Vs[i], dV_dIs[i]) =
      //  numeric::bspline::ndspline<2, 3, D, Real,
      //  Int>::interpolate(tablesd[0],indices[i]);
    });

    mgpu::standard_context_t context;
    mgpu::transform([=] MGPU_DEVICE(int idx) { func(idx); }, num_Vs, context);

    return Vs_t;
  }
};

template struct RamaDispatch<tmol::Device::CUDA, float, int32_t>;
template struct RamaDispatch<tmol::Device::CUDA, double, int32_t>;

}  // namespace potentials
}  // namespace rama
}  // namespace score
}  // namespace tmol
