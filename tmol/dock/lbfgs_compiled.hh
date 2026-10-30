#pragma once

#include <tuple>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/context_manager.hh>

namespace tmol {
namespace dock {

constexpr int MAX_HISTORY = 32;

// Fused state transitions of the pose-batched L-BFGS (see _batched_lbfgs.py).
// Each work item is one pose; every pose runs a fixed sequence of arithmetic,
// so the two launches per iteration can be captured in a CUDA graph.
template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real>
struct LBFGSDispatch {
  // Two-loop recursion, descent safeguard and the trial points x + a_t d for
  // a_t = scale * 2^-t, t < n_trials. Resets the history (rho) of poses whose
  // direction is not a descent direction. Returns (trial_x [t, pose, dof],
  // alpha [t, pose], g.d [pose]).
  static std::tuple<TPack<Real, 3, D>, TPack<Real, 2, D>, TPack<Real, 1, D>>
  direction(
      ContextManager& mgr,
      TView<Real, 2, D> x,
      TView<Real, 2, D> g,
      TView<Real, 3, D> s_hist,
      TView<Real, 3, D> y_hist,
      TView<Real, 2, D> rho,
      TView<Real, 1, D> gamma,
      TView<Real, 1, D> shrink,
      int64_t n_trials);

  // Armijo acceptance (largest satisfying step) and the in-place update of the
  // iterate, history, step shrink, convergence counters and active mask.
  static void update(
      ContextManager& mgr,
      TView<Real, 2, D> x,
      TView<Real, 1, D> f,
      TView<Real, 2, D> g,
      TView<Real, 3, D> s_hist,
      TView<Real, 3, D> y_hist,
      TView<Real, 2, D> rho,
      TView<Real, 1, D> gamma,
      TView<Real, 1, D> shrink,
      TView<int64_t, 1, D> small,
      TView<bool, 1, D> active,
      TView<Real, 3, D> trial_x,
      TView<Real, 2, D> trial_f,
      TView<Real, 3, D> trial_g,
      TView<Real, 2, D> alpha,
      TView<Real, 1, D> gd,
      Real c1,
      Real atol,
      int64_t patience);
};

}  // namespace dock
}  // namespace tmol
