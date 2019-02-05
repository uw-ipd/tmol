#include <moderngpu/kernel_reduce.hxx>

#include <tmol/utility/tensor/TensorPack.h>

#include "hybrid.hh"

namespace tmol {
namespace tests {
namespace utility {
namespace cpp_extension {

template <typename Real>
struct sum_tensor<Real, tmol::Device::CUDA> {
  static const tmol::Device D = tmol::Device::CUDA;

  static at::Tensor f(tmol::TView<Real, 1, D> t) {
    auto v_t = tmol::TPack<Real, 1, D>::empty({1});
    auto v = v_t.view;

    mgpu::standard_context_t context;

    mgpu::transform_reduce(
        [=] MGPU_DEVICE(int i) { return t[i]; },
        t.size(0),
        &v[0],
        mgpu::plus_t<Real>(),
        context);

    return v_t.tensor;
  }
};

template struct sum_tensor<float, tmol::Device::CUDA>;
template struct sum_tensor<double, tmol::Device::CUDA>;

}
}
}
}