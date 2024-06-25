#include <pybind11/pybind11.h>
#include <Eigen/Core>

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

// template <typename Real>
// using HomogeneousTransform = Eigen::Matrix<Real, 4, 4>;

template <typename Real>
struct HTRawBuffer {
  Real data[16];
};

// the composite operation for the forward pass: apply a transform
//   qt1/qt2 -> HT1/HT2 -> HT1*HT2 -> qt12' -> norm(qt12')
template <typename Real>
struct ht_sum : public std::binary_function<
                    HTRawBuffer<Real>,
                    HTRawBuffer<Real>,
                    HTRawBuffer<Real>> {
  MGPU_HOST_DEVICE
  HTRawBuffer<Real> operator()(
      HTRawBuffer<Real> ht1, HTRawBuffer<Real> ht2) const {
    HTRawBuffer<Real> res;
    for (int i = 0; i < 16; ++i) {
      res.data[i] = ht1.data[i] + ht2.data[i];
    }
    return res;
  }
};

}  // namespace cpp_extension
}  // namespace utility
}  // namespace tests
}  // namespace tmol

namespace tmol {

template <typename Real>
struct enable_tensor_view<tests::utility::cpp_extension::HTRawBuffer<Real>> {
  static const bool enabled = enable_tensor_view<Real>::enabled;
  static const at::ScalarType scalar_type() {
    enable_tensor_view<Real>::scalar_type();
  }

  static const int nconsumed_dims = 1;
  static const int consumed_dims(int i) {
    return (i == 0) ? sizeof(tests::utility::cpp_extension::HTRawBuffer<Real>)
                          / sizeof(Real)
                    : 0;
  }

  typedef typename enable_tensor_view<Real>::PrimitiveType PrimitiveType;
};

}  // namespace tmol

namespace pybind11 {
namespace detail {

template <typename T>
struct npy_format_descriptor_name<
    tmol::tests::utility::cpp_extension::HTRawBuffer<T>> {
  static constexpr auto name =
      _("HTRawBuffer(") + npy_format_descriptor_name<T>::name + _(")");
};

}  // namespace detail
}  // namespace pybind11

namespace tmol {
namespace tests {
namespace utility {
namespace cpp_extension {

struct float_ident {
  static float identity() { return 0.f; }
};

template <typename Real>
struct ht_zeros {
  static HTRawBuffer<Real> identity() {
    HTRawBuffer<Real> res;
    for (int i = 0; i < 16; ++i) {
      res.data[i] = 0;
    }
    return res;
  }
};

auto getScanBufferSize(int scansize, int nt, int vt) -> mgpu::tuple<int, int> {
  float scanleft = std::ceil(((float)scansize) / (nt * vt));
  int lbsSize = (int)scanleft + 1;
  int carryoutSize = (int)scanleft;
  while (scanleft > 1) {
    scanleft = std::ceil(scanleft / nt);
    carryoutSize += (int)scanleft;
  }

  return {carryoutSize, lbsSize};
}

struct plus_op {
  float operator()(float x, float y) const { return x + y; }
};

struct weird_op {
  MGPU_HOST_DEVICE float operator()(float x, float y) const { return y; }
};

template <
    typename Real,
    typename Ident,
    typename OP,
    tmol::Device D,
    mgpu::scan_type_t scan_type,
    int nt,
    int vt>
// template <typename Real, tmol::Device D, mgpu::scan_type_t scan_type, int nt,
// int vt>
struct segscan {};

template <
    typename Real,
    typename Ident,
    typename OP,
    mgpu::scan_type_t scan_type,
    int nt,
    int vt>
// template <typename Real, mgpu::scan_type_t scan_type, int nt, int vt>
struct segscan<Real, Ident, OP, tmol::Device::CPU, scan_type, nt, vt> {
  static const tmol::Device D = tmol::Device::CPU;

  static auto f(TView<Real, 1, D> t, TView<int32_t, 1, D> segs)
      -> TPack<Real, 1, D> {
    auto nelts = t.size(0);
    auto nsegs = segs.size(0);

    auto res_t = tmol::TPack<Real, 1, D>::zeros({nelts});
    auto res = res_t.view;
    OP op;

    for (int i = 0; i <= nsegs; ++i) {
      int lb = (i == 0) ? 0 : segs[i - 1];
      int ub = (i == nsegs) ? nelts : segs[i];
      for (int j = lb; j < ub; ++j) {
        if (j == lb) {
          res[j] = scan_type == mgpu::scan_type_inc ? t[j] : Ident::identity();
          // res[j] = scan_type == mgpu::scan_type_inc ? t[j] : 0;
        } else {
          res[j] = op(
              res[j - 1], (scan_type == mgpu::scan_type_inc ? t[j] : t[j - 1]));
        }
      }
    }

    return res_t;
  }
};

template <
    typename Real,
    typename Ident,
    typename OP,
    mgpu::scan_type_t scan_type,
    int nt,
    int vt>
// template <typename Real, mgpu::scan_type_t scan_type, int nt, int vt>
struct segscan<Real, Ident, OP, tmol::Device::CUDA, scan_type, nt, vt> {
  static const tmol::Device D = tmol::Device::CUDA;

  static auto f(TView<Real, 1, D> t, TView<int32_t, 1, D> segs)
      -> TPack<Real, 1, D> {
    auto nelts = t.size(0);
    auto nsegs = segs.size(0);

    auto res_t = tmol::TPack<Real, 1, D>::zeros({nelts});
    auto res = res_t.view;

    int carryoutBuff, lbsBuff;
    mgpu::tie(carryoutBuff, lbsBuff) = getScanBufferSize(nelts + nsegs, nt, vt);
    auto scanCarryout_t = TPack<Real, 1, D>::empty({carryoutBuff});
    auto scanCarryout = scanCarryout_t.view;
    auto scanCodes_t = TPack<int, 1, D>::empty({carryoutBuff});
    auto scanCodes = scanCodes_t.view;
    auto LBS_t = TPack<int, 1, D>::empty({lbsBuff});
    auto LBS = LBS_t.view;

    Real init = Ident::identity();
    // Real init = 0;

    auto data_loader = [=] MGPU_DEVICE(int index, int seg, int rank) {
      return t[index];
    };

    mgpu::standard_context_t context(false);
    tmol::kinematics::kernel_segscan<mgpu::launch_params_t<nt, vt>, scan_type>(
        data_loader,
        nelts,
        segs.data(),
        nsegs,
        res.data(),
        scanCarryout.data(),
        scanCodes.data(),
        LBS.data(),
        OP(),
        init,
        context);

    return res_t;
  }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace pybind11::literals;
  m.def(
      "segscan_incl",
      &segscan<
          float,
          float_ident,
          mgpu::plus_t<float>,
          tmol::Device::CUDA,
          mgpu::scan_type_inc,
          256,
          3>::f,
      "t"_a,
      "segs"_a);
  m.def(
      "segscan_incl",
      &segscan<
          float,
          float_ident,
          plus_op,
          tmol::Device::CPU,
          mgpu::scan_type_inc,
          256,
          3>::f,
      "t"_a,
      "segs"_a);
  m.def(
      "segscan_excl",
      &segscan<
          float,
          float_ident,
          mgpu::plus_t<float>,
          tmol::Device::CUDA,
          mgpu::scan_type_exc,
          256,
          3>::f,
      "t"_a,
      "segs"_a);
  m.def(
      "segscan_excl",
      &segscan<
          float,
          float_ident,
          plus_op,
          tmol::Device::CPU,
          mgpu::scan_type_exc,
          256,
          3>::f,
      "t"_a,
      "segs"_a);
  m.def(
      "segscan_incl_128_2",
      &segscan<
          float,
          float_ident,
          mgpu::plus_t<float>,
          tmol::Device::CUDA,
          mgpu::scan_type_inc,
          128,
          2>::f,
      "t"_a,
      "segs"_a);
  m.def(
      "segscan_incl_128_2",
      &segscan<
          float,
          float_ident,
          plus_op,
          tmol::Device::CPU,
          mgpu::scan_type_inc,
          128,
          2>::f,
      "t"_a,
      "segs"_a);
  m.def(
      "segscan_excl_128_2",
      &segscan<
          float,
          float_ident,
          mgpu::plus_t<float>,
          tmol::Device::CUDA,
          mgpu::scan_type_exc,
          128,
          2>::f,
      "t"_a,
      "segs"_a);
  m.def(
      "segscan_excl_128_2",
      &segscan<
          float,
          float_ident,
          plus_op,
          tmol::Device::CPU,
          mgpu::scan_type_exc,
          128,
          2>::f,
      "t"_a,
      "segs"_a);
  m.def(
      "weird_segscan_incl_128_2",
      &segscan<
          float,
          float_ident,
          weird_op,
          tmol::Device::CUDA,
          mgpu::scan_type_inc,
          128,
          2>::f,
      "t"_a,
      "segs"_a);
  m.def(
      "weird_segscan_incl_128_2",
      &segscan<
          float,
          float_ident,
          weird_op,
          tmol::Device::CPU,
          mgpu::scan_type_inc,
          128,
          2>::f,
      "t"_a,
      "segs"_a);
  m.def(
      "ht_segscan_incl",
      &segscan<
          HTRawBuffer<float>,
          ht_zeros<float>,
          ht_sum<float>,
          tmol::Device::CUDA,
          mgpu::scan_type_inc,
          128,
          2>::f,
      "t"_a,
      "segs"_a);
  m.def(
      "ht_segscan_incl",
      &segscan<
          HTRawBuffer<float>,
          ht_zeros<float>,
          ht_sum<float>,
          tmol::Device::CPU,
          mgpu::scan_type_inc,
          128,
          2>::f,
      "t"_a,
      "segs"_a);
}

}  // namespace cpp_extension
}  // namespace utility
}  // namespace tests
}  // namespace tmol
