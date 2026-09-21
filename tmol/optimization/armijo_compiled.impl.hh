#pragma once

#include <cmath>

#include <tmol/optimization/armijo_compiled.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/launch_box_macros.hh>

namespace tmol {
namespace optimization {

// Keep these values synchronized with _lbfgs_armijo.py.  The kernels preserve
// the Python implementation's sequential state transitions while replacing
// each group of elementwise tensor expressions with one launch.
constexpr int64_t LS_DONE = 0;
constexpr int64_t LS_INCREASE = 1;
constexpr int64_t LS_BACKTRACK = 2;
constexpr int64_t LS_FAILED = 3;

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real>
TPack<Real, 1, D> ArmijoCompiledDispatch<DeviceDispatch, D, Real>::start(
    ContextManager& mgr,
    TView<bool, 1, D> searching,
    TView<Real, 1, D> alpha0) {
  int const n_segments = alpha0.size(0);
  auto alpha_t = TPack<Real, 1, D>::empty({n_segments});
  auto alpha = alpha_t.view;

  LAUNCH_BOX_128;
  DeviceDispatch<D>::template forall<launch_t>(
      mgr, n_segments, [=] TMOL_DEVICE_FUNC(int i) {
        alpha[i] = searching[i] ? alpha0[i] : Real(0);
      });
  return alpha_t;
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real>
std::tuple<
    TPack<Real, 1, D>,
    TPack<Real, 1, D>,
    TPack<int64_t, 1, D>,
    TPack<bool, 1, D>>
ArmijoCompiledDispatch<DeviceDispatch, D, Real>::classify(
    ContextManager& mgr,
    TView<bool, 1, D> searching,
    TView<Real, 1, D> alpha,
    TView<Real, 1, D> phi,
    TView<Real, 1, D> phi0,
    TView<Real, 1, D> derphi0,
    Real sigma_increase,
    Real sigma_decrease) {
  int const n_segments = alpha.size(0);
  auto accepted_t = TPack<Real, 1, D>::empty({n_segments});
  auto phi_accepted_t = TPack<Real, 1, D>::empty({n_segments});
  auto status_t = TPack<int64_t, 1, D>::empty({n_segments});
  auto active_t = TPack<bool, 1, D>::empty({n_segments});
  auto accepted = accepted_t.view;
  auto phi_accepted = phi_accepted_t.view;
  auto status = status_t.view;
  auto active = active_t.view;

  LAUNCH_BOX_128;
  DeviceDispatch<D>::template forall<launch_t>(
      mgr, n_segments, [=] TMOL_DEVICE_FUNC(int i) {
        bool const linear =
            phi[i] <= phi0[i] + alpha[i] * sigma_increase * derphi0[i];
        bool const sufficient =
            phi[i] <= phi0[i] + alpha[i] * sigma_decrease * derphi0[i];
        bool const took = searching[i] && (linear || sufficient);

        accepted[i] = took ? alpha[i] : Real(0);
        phi_accepted[i] = took ? phi[i] : phi0[i];
        int64_t const status_i = !searching[i] ? LS_DONE
                                 : linear      ? LS_INCREASE
                                 : !sufficient ? LS_BACKTRACK
                                               : LS_DONE;
        status[i] = status_i;
        active[i] = status_i == LS_INCREASE || status_i == LS_BACKTRACK;
      });
  return {accepted_t, phi_accepted_t, status_t, active_t};
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real>
TPack<Real, 1, D> ArmijoCompiledDispatch<DeviceDispatch, D, Real>::trial(
    ContextManager& mgr,
    TView<int64_t, 1, D> status,
    TView<Real, 1, D> alpha,
    TView<Real, 1, D> accepted,
    Real factor) {
  int const n_segments = alpha.size(0);
  auto result_t = TPack<Real, 1, D>::empty({n_segments});
  auto result = result_t.view;

  LAUNCH_BOX_128;
  DeviceDispatch<D>::template forall<launch_t>(
      mgr, n_segments, [=] TMOL_DEVICE_FUNC(int i) {
        result[i] = status[i] == LS_INCREASE    ? alpha[i] / factor
                    : status[i] == LS_BACKTRACK ? alpha[i] * factor * factor
                                                : accepted[i];
      });
  return result_t;
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real>
std::tuple<
    TPack<Real, 1, D>,
    TPack<Real, 1, D>,
    TPack<int64_t, 1, D>,
    TPack<bool, 1, D>,
    TPack<bool, 1, D>>
ArmijoCompiledDispatch<DeviceDispatch, D, Real>::update(
    ContextManager& mgr,
    TView<int64_t, 1, D> status,
    TView<Real, 1, D> trial,
    TView<Real, 1, D> accepted,
    TView<Real, 1, D> phi_accepted,
    TView<Real, 1, D> phi,
    TView<Real, 1, D> phi_trial,
    TView<Real, 1, D> phi0,
    TView<Real, 1, D> derphi0,
    Real sigma_decrease,
    Real minstep) {
  int const n_segments = trial.size(0);
  auto next_accepted_t = TPack<Real, 1, D>::empty({n_segments});
  auto next_phi_accepted_t = TPack<Real, 1, D>::empty({n_segments});
  auto next_status_t = TPack<int64_t, 1, D>::empty({n_segments});
  auto failed_t = TPack<bool, 1, D>::empty({n_segments});
  auto active_t = TPack<bool, 1, D>::empty({n_segments});
  auto next_accepted = next_accepted_t.view;
  auto next_phi_accepted = next_phi_accepted_t.view;
  auto next_status = next_status_t.view;
  auto failed = failed_t.view;
  auto active = active_t.view;

  LAUNCH_BOX_128;
  DeviceDispatch<D>::template forall<launch_t>(
      mgr, n_segments, [=] TMOL_DEVICE_FUNC(int i) {
        Real accepted_i = accepted[i];
        Real phi_accepted_i = phi_accepted[i];
        int64_t status_i = status[i];
        bool failed_i = false;

        if (status_i == LS_INCREASE) {
          if (phi_trial[i] < phi[i]) {
            accepted_i = trial[i];
            phi_accepted_i = phi_trial[i];
          }
          status_i = LS_DONE;
        } else if (status_i == LS_BACKTRACK) {
          bool const armijo =
              phi_trial[i] <= phi0[i] + trial[i] * sigma_decrease * derphi0[i];
          if (armijo) {
            accepted_i = trial[i];
            phi_accepted_i = phi_trial[i];
            status_i = LS_DONE;
          } else if (trial[i] < minstep) {
            if (phi_trial[i] < phi0[i]) {
              accepted_i = trial[i];
              phi_accepted_i = phi_trial[i];
              status_i = LS_DONE;
            } else {
              accepted_i = Real(0);
              phi_accepted_i = phi0[i];
              status_i = LS_FAILED;
              failed_i = true;
            }
          }
        }

        next_accepted[i] = accepted_i;
        next_phi_accepted[i] = phi_accepted_i;
        next_status[i] = status_i;
        failed[i] = failed_i;
        active[i] = status_i == LS_INCREASE || status_i == LS_BACKTRACK;
      });
  return {
      next_accepted_t, next_phi_accepted_t, next_status_t, failed_t, active_t};
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real>
std::tuple<TPack<Real, 1, D>, TPack<bool, 1, D>>
ArmijoCompiledDispatch<DeviceDispatch, D, Real>::finalize(
    ContextManager& mgr,
    TView<int64_t, 1, D> status,
    TView<Real, 1, D> derphi0,
    TView<Real, 1, D> accepted,
    TView<Real, 1, D> start,
    TView<bool, 1, D> searching,
    Real minstep) {
  int const n_segments = accepted.size(0);
  auto step_t = TPack<Real, 1, D>::empty({n_segments});
  auto failed_t = TPack<bool, 1, D>::empty({n_segments});
  auto step = step_t.view;
  auto failed = failed_t.view;

  LAUNCH_BOX_128;
  DeviceDispatch<D>::template forall<launch_t>(
      mgr, n_segments, [=] TMOL_DEVICE_FUNC(int i) {
        bool const failed_i = status[i] == LS_FAILED;
        Real selected = accepted[i];
        if (failed_i) {
          Real magnitude = -derphi0[i];
          magnitude = magnitude < minstep ? minstep : magnitude;
          Real retry = Real(1) / sqrt(magnitude);
          selected = retry < Real(1) ? retry : Real(1);
        }
        step[i] = searching[i] ? selected : start[i];
        failed[i] = failed_i;
      });
  return {step_t, failed_t};
}

}  // namespace optimization
}  // namespace tmol
