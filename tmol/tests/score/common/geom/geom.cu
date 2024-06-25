#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/tensor/pybind.h>
#include <moderngpu/transform.hxx>

#include <tmol/score/common/geom.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/tuple_operators.hh>

template <typename Real, tmol::Device D>
void bind(pybind11::module& m) {
  using namespace pybind11::literals;
  using namespace tmol;
  using namespace tmol::score::common;
  typedef Eigen::Matrix<Real, 3, 1> Real3;

  m.def(
      "distance_V",
      [](TView<Real3, 1, D> A, TView<Real3, 1, D> B) {
        TORCH_CHECK(
            A.size(0) == B.size(0),
            "Invalid sizes A: ",
            A.size(0),
            " B: ",
            B.size(0));

        auto V_t = tmol::TPack<Real, 1, D>::empty({A.size(0)});
        auto V = V_t.view;

        mgpu::standard_context_t context;

        mgpu::transform(
            [=] MGPU_LAMBDA(int i) { V[i] = distance<Real>::V(A[i], B[i]); },
            A.size(0),
            context);

        return V_t.tensor;
      },
      "A"_a,
      "B"_a);

  m.def(
      "distance_V_dV",
      [](TView<Real3, 1, D> A, TView<Real3, 1, D> B) {
        TORCH_CHECK(
            A.size(0) == B.size(0),
            "Invalid sizes A: ",
            A.size(0),
            " B: ",
            B.size(0));

        auto V_t = tmol::TPack<Real, 1, D>::empty({A.size(0)});
        auto dV_dA_t = tmol::TPack<Real3, 1, D>::empty({A.size(0)});
        auto dV_dB_t = tmol::TPack<Real3, 1, D>::empty({B.size(0)});

        auto V = V_t.view;
        auto dV_dA = dV_dA_t.view;
        auto dV_dB = dV_dB_t.view;

        mgpu::standard_context_t context;

        mgpu::transform(
            [=] MGPU_LAMBDA(int i) {
              using tmol::score::common::tie;
              tie(V[i], dV_dA[i], dV_dB[i]) =
                  distance<Real>::V_dV(A[i], B[i]).astuple();
            },
            A.size(0),
            context);

        return std::make_tuple(V_t.tensor, dV_dA_t.tensor, dV_dB_t.tensor);
      },
      "A"_a,
      "B"_a);
};

PYBIND11_MODULE(geom, m) {
  bind<double, tmol::Device::CUDA>(m);
  bind<float, tmol::Device::CUDA>(m);
}
