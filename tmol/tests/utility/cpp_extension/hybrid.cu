#include <moderngpu/kernel_reduce.hxx>

#include "hybrid.hh"

template <typename Real>
struct sum<Real, tmol::Device::CUDA> {
  static const tmol::Device D = tmol::Device::CUDA;

  static at::Tensor f(tmol::TView<Real, 1, D> t) {
    at::Tensor v_t;
    tmol::TView<Real, 1, D> v;
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

template struct sum<float, tmol::Device::CUDA>;
template struct sum<double, tmol::Device::CUDA>;
