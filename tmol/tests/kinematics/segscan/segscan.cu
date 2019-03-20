#include <pybind11/pybind11.h>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/tensor/pybind.h>
#include <tmol/kinematics/compiled/kernel_segscan.cuh>

namespace tmol {
namespace tests {
namespace utility {
namespace cpp_extension {

using tmol::TPack;
using tmol::TView;

template <typename Real, tmol::Device D>
struct segscan {};

template <typename Real>
struct segscan<Real, tmol::Device::CPU> {
  static const tmol::Device D = tmol::Device::CPU;

  static auto f(TView<Real, 1, D> t, TView<int32_t, 1, D> segs)
      -> TPack<Real, 1, D> {
    auto nelts = t.size(0);
    auto nsegs = segs.size(0);

    auto res_t = tmol::TPack<Real, 1, D>::zeros({nelts});
    auto res = res_t.view;

    for (int i = 0; i <= nsegs; ++i) {
      int lb = (i == 0) ? 0 : segs[i - 1];
      int ub = (i == nsegs) ? nelts : segs[i];
      for (int j = lb; j < ub; ++j) {
        res[j] = (j == lb) ? t[j] : res[j - 1] + t[j];
      }
    }

    return res_t;
  }
};

template <typename Real>
struct segscan<Real, tmol::Device::CUDA> {
  static const tmol::Device D = tmol::Device::CUDA;

  static auto f(TView<Real, 1, D> t, TView<int32_t, 1, D> segs)
      -> TPack<Real, 1, D> {
    auto nelts = t.size(0);
    auto nsegs = segs.size(0);

    auto res_t = tmol::TPack<Real, 1, D>::zeros({nelts});
    auto res = res_t.view;

    Real init = 0;

    auto data_loader = [=] MGPU_DEVICE(int index, int seg, int rank) {
      return t[index];
    };

    mgpu::standard_context_t context;
    tmol::kinematics::kernel_segscan<mgpu::launch_params_t<256, 3> >(
        data_loader,
        nelts,
        segs.data(),
        nsegs,
        res.data(),
        mgpu::plus_t<Real>(),
        init,
        context);

    return res_t;
  }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace pybind11::literals;
  m.def("segscan", &segscan<float, tmol::Device::CUDA>::f, "t"_a, "segs"_a);
  m.def("segscan", &segscan<float, tmol::Device::CPU>::f, "t"_a, "segs"_a);
}

}  // namespace cpp_extension
}  // namespace utility
}  // namespace tests
}  // namespace tmol