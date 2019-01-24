#include <pybind11/pybind11.h>

#include <moderngpu/kernel_reduce.hxx>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/tensor/pybind.h>

using tmol::TView;

template <typename Real, tmol::Device D>
struct sum {};

template <typename Real>
struct sum<Real, tmol::Device::CPU> {
  static const tmol::Device D = tmol::Device::CPU;

  static auto f(TView<Real, 1, D> t) -> at::Tensor {
    at::Tensor v_t;
    TView<Real, 1, D> v;
    std::tie(v_t, v) = tmol::new_tensor<Real, 1, D>({1});

    v[0] = 0;
    for (int i = 0; i < t.size(0); i++) {
      v[0] += t[i];
    }

    return v_t;
  }
};

template <typename Real>
struct sum<Real, tmol::Device::CUDA> {
  static const tmol::Device D = tmol::Device::CUDA;

  static auto f(TView<Real, 1, D> t) -> at::Tensor {
    at::Tensor v_t;
    TView<Real, 1, D> v;
    std::tie(v_t, v) = tmol::new_tensor<Real, 1, D>({1});

    mgpu::standard_context_t context;

    mgpu::transform_reduce(
        [=] MGPU_DEVICE(int i) { return t[i]; },
        t.size(0),
        &v[0],
        mgpu::plus_t<Real>(),
        context);

    return v_t;
  }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace pybind11::literals;
  m.def("sum", &sum<float, tmol::Device::CPU>::f, "t"_a);
  m.def("sum", &sum<float, tmol::Device::CUDA>::f, "t"_a);
}
