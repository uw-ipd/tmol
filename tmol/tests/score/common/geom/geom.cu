#include <pybind11/pybind11.h>
#include <torch/torch.h>

#include <tmol/utility/tensor/TensorAccessor.h>
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
        AT_CHECK(
            A.size(0) == B.size(0),
            "Invalid sizes A: ",
            A.size(0),
            " B: ",
            B.size(0));

        at::Tensor V_t;
        TView<Real, 1, D> V;
        std::tie(V_t, V) = tmol::new_tensor<Real, 1, D>({A.size(0)});

        mgpu::standard_context_t context(false);

        mgpu::transform(
            [=] MGPU_LAMBDA(int i) { V[i] = distance_V(A[i], B[i]); },
            A.size(0),
            context);

        return V_t;
      },
      "A"_a,
      "B"_a);

  m.def(
      "distance_V_dV",
      [](TView<Real3, 1, D> A, TView<Real3, 1, D> B) {
        AT_CHECK(
            A.size(0) == B.size(0),
            "Invalid sizes A: ",
            A.size(0),
            " B: ",
            B.size(0));

        at::Tensor V_t;
        TView<Real, 1, D> V;
        std::tie(V_t, V) = tmol::new_tensor<Real, 1, D>({A.size(0)});

        at::Tensor dV_dA_t;
        TView<Real3, 1, D> dV_dA;
        std::tie(dV_dA_t, dV_dA) = tmol::new_tensor<Real3, 1, D>({A.size(0)});

        at::Tensor dV_dB_t;
        TView<Real3, 1, D> dV_dB;
        std::tie(dV_dB_t, dV_dB) = tmol::new_tensor<Real3, 1, D>({B.size(0)});

        mgpu::standard_context_t context(false);

        mgpu::transform(
            [=] MGPU_LAMBDA(int i) {
              using tmol::score::common::tie;
              tie(V[i], dV_dA[i], dV_dB[i]) = distance_V_dV(A[i], B[i]);
            },
            A.size(0),
            context);

        return std::make_tuple(V_t, dV_dA_t, dV_dB_t);
      },
      "A"_a,
      "B"_a);
};

PYBIND11_MODULE(geom, m) {
  bind<double, tmol::Device::CUDA>(m);
  bind<float, tmol::Device::CUDA>(m);
}
