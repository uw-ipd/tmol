#include <tmol/utility/tensor/native_library.hh>

#include <tmol/dock/lbfgs_compiled.hh>
#include <tmol/score/common/device_operations.hh>
#include <tmol/utility/function_dispatch/aten.hh>
#include <tmol/utility/tensor/TensorCast.h>

namespace tmol {
namespace dock {

using tmol::native::Tensor;
ContextManager mgr;

std::tuple<Tensor, Tensor, Tensor> lbfgs_direction(
    Tensor x,
    Tensor g,
    Tensor s_hist,
    Tensor y_hist,
    Tensor rho,
    Tensor gamma,
    Tensor shrink,
    int64_t n_trials) {
  TORCH_CHECK(s_hist.size(1) <= MAX_HISTORY, "L-BFGS history is too long");
  Tensor trial_x, alpha, gd;
  TMOL_DISPATCH_FLOATING_DEVICE(x.options(), "lbfgs_direction", ([&] {
    using Real = scalar_t;
    constexpr tmol::Device Dev = device_t;
    auto result = LBFGSDispatch<score::common::DeviceOperations, Dev, Real>::
        direction(
            mgr,
            TCAST(x),
            TCAST(g),
            TCAST(s_hist),
            TCAST(y_hist),
            TCAST(rho),
            TCAST(gamma),
            TCAST(shrink),
            n_trials);
    trial_x = std::get<0>(result).tensor;
    alpha = std::get<1>(result).tensor;
    gd = std::get<2>(result).tensor;
  }));
  return {trial_x, alpha, gd};
}

// Updates its state arguments in place; returns the active mask for chaining.
Tensor lbfgs_update(
    Tensor x,
    Tensor f,
    Tensor g,
    Tensor s_hist,
    Tensor y_hist,
    Tensor rho,
    Tensor gamma,
    Tensor shrink,
    Tensor small,
    Tensor active,
    Tensor trial_x,
    Tensor trial_f,
    Tensor trial_g,
    Tensor alpha,
    Tensor gd,
    double c1,
    double atol,
    int64_t patience) {
  TMOL_DISPATCH_FLOATING_DEVICE(x.options(), "lbfgs_update", ([&] {
    using Real = scalar_t;
    constexpr tmol::Device Dev = device_t;
    LBFGSDispatch<score::common::DeviceOperations, Dev, Real>::update(
        mgr,
        TCAST(x),
        TCAST(f),
        TCAST(g),
        TCAST(s_hist),
        TCAST(y_hist),
        TCAST(rho),
        TCAST(gamma),
        TCAST(shrink),
        TCAST(small),
        TCAST(active),
        TCAST(trial_x),
        TCAST(trial_f),
        TCAST(trial_g),
        TCAST(alpha),
        TCAST(gd),
        Real(c1),
        Real(atol),
        patience);
  }));
  return active;
}

TMOL_LIBRARY(tmol_dock, m) {
  m.def("lbfgs_direction", TMOL_FN(&lbfgs_direction));
  m.def("lbfgs_update", TMOL_FN(&lbfgs_update));
}

}  // namespace dock
}  // namespace tmol
