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

template <typename Real, tmol::Device D, mgpu::scan_type_t scan_type>
struct segscan {};

template <typename Real, mgpu::scan_type_t scan_type>
struct segscan<Real, tmol::Device::CPU, scan_type> {
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
        if (j == lb) {
          res[j] = scan_type == mgpu::scan_type_inc ? t[j] : 0;
        } else {
          res[j] =
              res[j - 1] + (scan_type == mgpu::scan_type_inc ? t[j] : t[j - 1]);
        }
      }
    }

    return res_t;
  }
};

template <typename Real, mgpu::scan_type_t scan_type>
struct segscan<Real, tmol::Device::CUDA, scan_type> {
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
    tmol::kinematics::kernel_segscan<mgpu::launch_params_t<256, 3>, scan_type>(
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
  m.def(
      "segscan_incl",
      &segscan<float, tmol::Device::CUDA, mgpu::scan_type_inc>::f,
      "t"_a,
      "segs"_a);
  m.def(
      "segscan_incl",
      &segscan<float, tmol::Device::CPU, mgpu::scan_type_inc>::f,
      "t"_a,
      "segs"_a);
  m.def(
      "segscan_excl",
      &segscan<float, tmol::Device::CUDA, mgpu::scan_type_exc>::f,
      "t"_a,
      "segs"_a);
  m.def(
      "segscan_excl",
      &segscan<float, tmol::Device::CPU, mgpu::scan_type_exc>::f,
      "t"_a,
      "segs"_a);
}

}  // namespace cpp_extension
}  // namespace utility
}  // namespace tests
}  // namespace tmol