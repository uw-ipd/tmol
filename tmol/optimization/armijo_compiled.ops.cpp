#include <torch/script.h>
#include <torch/torch.h>

#include <tmol/optimization/armijo_compiled.hh>
#include <tmol/score/common/device_operations.hh>
#include <tmol/utility/function_dispatch/aten.hh>
#include <tmol/utility/tensor/TensorCast.h>

namespace tmol {
namespace optimization {

using torch::Tensor;
ContextManager mgr;

Tensor armijo_start(Tensor searching, Tensor alpha0) {
  Tensor alpha;
  TMOL_DISPATCH_FLOATING_DEVICE(
      alpha0.options(), "armijo_start", ([&] {
        using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;
        alpha =
            ArmijoCompiledDispatch<score::common::DeviceOperations, Dev, Real>::
                start(mgr, TCAST(searching), TCAST(alpha0))
                    .tensor;
      }));
  return alpha;
}

std::tuple<Tensor, Tensor, Tensor, Tensor> armijo_classify(
    Tensor searching,
    Tensor alpha,
    Tensor phi,
    Tensor phi0,
    Tensor derphi0,
    double sigma_increase,
    double sigma_decrease) {
  Tensor accepted, phi_accepted, status, active;
  TMOL_DISPATCH_FLOATING_DEVICE(
      alpha.options(), "armijo_classify", ([&] {
        using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;
        auto result =
            ArmijoCompiledDispatch<score::common::DeviceOperations, Dev, Real>::
                classify(
                    mgr,
                    TCAST(searching),
                    TCAST(alpha),
                    TCAST(phi),
                    TCAST(phi0),
                    TCAST(derphi0),
                    Real(sigma_increase),
                    Real(sigma_decrease));
        accepted = std::get<0>(result).tensor;
        phi_accepted = std::get<1>(result).tensor;
        status = std::get<2>(result).tensor;
        active = std::get<3>(result).tensor;
      }));
  return {accepted, phi_accepted, status, active};
}

Tensor armijo_trial(
    Tensor status, Tensor alpha, Tensor accepted, double factor) {
  Tensor trial;
  TMOL_DISPATCH_FLOATING_DEVICE(
      alpha.options(), "armijo_trial", ([&] {
        using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;
        trial =
            ArmijoCompiledDispatch<score::common::DeviceOperations, Dev, Real>::
                trial(
                    mgr,
                    TCAST(status),
                    TCAST(alpha),
                    TCAST(accepted),
                    Real(factor))
                    .tensor;
      }));
  return trial;
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> armijo_update(
    Tensor status,
    Tensor trial,
    Tensor accepted,
    Tensor phi_accepted,
    Tensor phi,
    Tensor phi_trial,
    Tensor phi0,
    Tensor derphi0,
    double sigma_decrease,
    double minstep) {
  Tensor next_accepted, next_phi_accepted, next_status, failed, active;
  TMOL_DISPATCH_FLOATING_DEVICE(
      trial.options(), "armijo_update", ([&] {
        using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;
        auto result =
            ArmijoCompiledDispatch<score::common::DeviceOperations, Dev, Real>::
                update(
                    mgr,
                    TCAST(status),
                    TCAST(trial),
                    TCAST(accepted),
                    TCAST(phi_accepted),
                    TCAST(phi),
                    TCAST(phi_trial),
                    TCAST(phi0),
                    TCAST(derphi0),
                    Real(sigma_decrease),
                    Real(minstep));
        next_accepted = std::get<0>(result).tensor;
        next_phi_accepted = std::get<1>(result).tensor;
        next_status = std::get<2>(result).tensor;
        failed = std::get<3>(result).tensor;
        active = std::get<4>(result).tensor;
      }));
  return {next_accepted, next_phi_accepted, next_status, failed, active};
}

std::tuple<Tensor, Tensor> armijo_finalize(
    Tensor status,
    Tensor derphi0,
    Tensor accepted,
    Tensor start,
    Tensor searching,
    double minstep) {
  Tensor step, failed;
  TMOL_DISPATCH_FLOATING_DEVICE(
      accepted.options(), "armijo_finalize", ([&] {
        using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;
        auto result =
            ArmijoCompiledDispatch<score::common::DeviceOperations, Dev, Real>::
                finalize(
                    mgr,
                    TCAST(status),
                    TCAST(derphi0),
                    TCAST(accepted),
                    TCAST(start),
                    TCAST(searching),
                    Real(minstep));
        step = std::get<0>(result).tensor;
        failed = std::get<1>(result).tensor;
      }));
  return {step, failed};
}

TORCH_LIBRARY(tmol_optimization, m) {
  m.def("armijo_start", &armijo_start);
  m.def("armijo_classify", &armijo_classify);
  m.def("armijo_trial", &armijo_trial);
  m.def("armijo_update", &armijo_update);
  m.def("armijo_finalize", &armijo_finalize);
}

}  // namespace optimization
}  // namespace tmol
