#pragma once

#include <algorithm>
#include <cmath>

#include <tmol/dock/lbfgs_compiled.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/launch_box_macros.hh>

namespace tmol {
namespace dock {

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real>
std::tuple<TPack<Real, 3, D>, TPack<Real, 2, D>, TPack<Real, 1, D>>
LBFGSDispatch<DeviceDispatch, D, Real>::direction(
    ContextManager& mgr,
    TView<Real, 2, D> x,
    TView<Real, 2, D> g,
    TView<Real, 3, D> s_hist,
    TView<Real, 3, D> y_hist,
    TView<Real, 2, D> rho,
    TView<Real, 1, D> gamma,
    TView<Real, 1, D> shrink,
    int64_t n_trials) {
  int const n_poses = x.size(0);
  int const n_dofs = x.size(1);
  int const history = s_hist.size(1);
  auto trial_x_t = TPack<Real, 3, D>::empty({n_trials, n_poses, n_dofs});
  auto alpha_t = TPack<Real, 2, D>::empty({n_trials, n_poses});
  auto gd_t = TPack<Real, 1, D>::empty({n_poses});
  auto trial_x = trial_x_t.view;
  auto alpha = alpha_t.view;
  auto gd_out = gd_t.view;

  LAUNCH_BOX_32;
  DeviceDispatch<D>::template forall_independent<launch_t>(
      mgr, n_poses, [=] TMOL_DEVICE_FUNC(int i) {
        // d is built in the first trial slot, then scaled into every trial.
        auto d = [&](int k) -> Real& { return trial_x[0][i][k]; };
        Real a[MAX_HISTORY];
        for (int k = 0; k < n_dofs; ++k) d(k) = g[i][k];
        for (int j = history - 1; j >= 0; --j) {
          Real dot = 0;
          for (int k = 0; k < n_dofs; ++k) dot += s_hist[i][j][k] * d(k);
          a[j] = rho[i][j] * dot;
          for (int k = 0; k < n_dofs; ++k) d(k) -= a[j] * y_hist[i][j][k];
        }
        for (int k = 0; k < n_dofs; ++k) d(k) *= gamma[i];
        for (int j = 0; j < history; ++j) {
          Real dot = 0;
          for (int k = 0; k < n_dofs; ++k) dot += y_hist[i][j][k] * d(k);
          Real const b = rho[i][j] * dot;
          for (int k = 0; k < n_dofs; ++k) d(k) += s_hist[i][j][k] * (a[j] - b);
        }
        Real gdv = 0;
        for (int k = 0; k < n_dofs; ++k) {
          d(k) = -d(k);
          gdv += g[i][k] * d(k);
        }
        // Not a descent direction: steepest descent, and forget the curvature.
        if (gdv >= 0) {
          gdv = 0;
          for (int k = 0; k < n_dofs; ++k) {
            d(k) = -g[i][k];
            gdv += g[i][k] * d(k);
          }
          for (int j = 0; j < history; ++j) rho[i][j] = 0;
        }
        bool fresh = true;
        for (int j = 0; j < history; ++j) fresh = fresh && rho[i][j] == 0;
        // Without curvature information the first step has unit length.
        Real gmax = 1;
        for (int k = 0; k < n_dofs; ++k) {
          gmax = std::max(gmax, std::abs(g[i][k]));
        }
        Real const scale = fresh ? shrink[i] / gmax : Real(1);
        gd_out[i] = gdv;
        // t = 0 last: it overwrites the direction held in its own slot.
        for (int t = n_trials - 1; t >= 0; --t) {
          Real const step = scale * std::pow(Real(0.5), Real(t));
          alpha[t][i] = step;
          for (int k = 0; k < n_dofs; ++k) {
            trial_x[t][i][k] = x[i][k] + step * trial_x[0][i][k];
          }
        }
      });
  return {trial_x_t, alpha_t, gd_t};
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real>
void LBFGSDispatch<DeviceDispatch, D, Real>::update(
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
    int64_t patience) {
  int const n_poses = x.size(0);
  int const n_dofs = x.size(1);
  int const history = s_hist.size(1);
  int const n_trials = trial_f.size(0);

  LAUNCH_BOX_32;
  DeviceDispatch<D>::template forall_independent<launch_t>(
      mgr, n_poses, [=] TMOL_DEVICE_FUNC(int i) {
        if (!active[i]) return;
        // Largest step satisfying sufficient decrease: what sequential
        // backtracking from the first trial would accept.
        int first = -1;
        for (int t = 0; t < n_trials && first < 0; ++t) {
          if (trial_f[t][i] <= f[i] + c1 * alpha[t][i] * gd[i]) first = t;
        }
        if (first < 0) {
          // No acceptable step: restart from steepest descent, with a smaller
          // step.
          for (int j = 0; j < history; ++j) rho[i][j] = 0;
          shrink[i] = shrink[i] / 64;
          small[i] = 0;
          return;
        }
        shrink[i] = 1;
        Real sy = 0, ss = 0, yy = 0;
        for (int k = 0; k < n_dofs; ++k) {
          Real const s = trial_x[first][i][k] - x[i][k];
          Real const y = trial_g[first][i][k] - g[i][k];
          sy += s * y;
          ss += s * s;
          yy += y * y;
          for (int j = 0; j + 1 < history; ++j) {
            s_hist[i][j][k] = s_hist[i][j + 1][k];
            y_hist[i][j][k] = y_hist[i][j + 1][k];
          }
          s_hist[i][history - 1][k] = s;
          y_hist[i][history - 1][k] = y;
        }
        for (int j = 0; j + 1 < history; ++j) rho[i][j] = rho[i][j + 1];
        bool const useful = sy > Real(1e-10) * std::sqrt(ss) * std::sqrt(yy);
        rho[i][history - 1] =
            useful ? Real(1) / std::max(sy, Real(1e-30)) : Real(0);
        if (useful) gamma[i] = sy / std::max(yy, Real(1e-30));

        Real const new_f = trial_f[first][i];
        small[i] = std::abs(new_f - f[i]) <= atol ? small[i] + 1 : 0;
        for (int k = 0; k < n_dofs; ++k) {
          x[i][k] = trial_x[first][i][k];
          g[i][k] = trial_g[first][i][k];
        }
        f[i] = new_f;
        if (small[i] >= patience) active[i] = false;
      });
}

}  // namespace dock
}  // namespace tmol
