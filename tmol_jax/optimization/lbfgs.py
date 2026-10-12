"""Limited-memory BFGS minimizer — pure JAX port.

Upstream reference: ``tmol/optimization/lbfgs_armijo.py`` (the
``LBFGS_Armijo`` class). Algorithmically we follow the standard
Nocedal two-loop recursion (``Numerical Optimization``, Algorithm 7.4):
given the past ``history`` pairs of ``(s_i, y_i) = (x_i+1 - x_i,
g_i+1 - g_i)``, compute the quasi-Newton step as
``d = -H_k * g_k`` where ``H_k`` is the implicit inverse Hessian.

Design notes for the JAX port
----------------------------
* We evaluate gradients through ``jax.value_and_grad(fn)`` so callers
  only supply the scalar-output energy function. The caller is free
  to ``jit`` the returned ``f_and_grad`` themselves, or to pass a
  pre-jitted version — :func:`lbfgs_minimize` wraps the closure with
  ``jax.jit`` internally as a convenience and caches it across
  iterations.
* The convergence loop is a plain Python ``while`` (not
  ``lax.while_loop``). Termination is *dynamic* (driven by gradient
  norms, energy changes, line-search failure, etc.) and trying to
  staticize it would force us to unroll the iteration budget —
  defeating the point of early stopping.
* The history is stored as two fixed-size ring buffers of shape
  ``(history, n)``. We only ever read the valid prefix, so later
  unused rows are harmless. This mirrors the upstream circular-buffer
  design and keeps every ``lbfgs_minimize`` call reusing the same
  compile cache (shapes don't change after the first iteration).
* The line search is strong-Wolfe (see :mod:`~.line_search`); we
  prefer it over the Armijo / Rosetta-style increment-or-halve rule
  because the curvature condition keeps ``y.s > 0``, which is what
  the Hessian approximation needs for positive definiteness.
* The gradient mask (``grad_mask``) lets the caller freeze a subset
  of coordinates/DOFs by zeroing out their gradient before stepping.
  When provided, every direction is also masked — equivalent to
  restricting the quasi-Newton step to the movable subspace.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp

from tmol_jax.optimization.line_search import strong_wolfe_line_search


def _two_loop_recursion(
    g: jnp.ndarray,
    s_history: jnp.ndarray,
    y_history: jnp.ndarray,
    rho_history: jnp.ndarray,
    hist_count: int,
) -> jnp.ndarray:
    """Nocedal two-loop recursion returning ``d = -H * g``.

    Operates on the valid prefix of the history (indices
    ``0 .. hist_count-1``); later rows are ignored. If ``hist_count == 0``
    the method degenerates to steepest descent (``d = -g``).
    """
    q = -g
    if hist_count == 0:
        return q

    s_valid = s_history[:hist_count]
    y_valid = y_history[:hist_count]
    rho_valid = rho_history[:hist_count]

    # First loop: right-to-left in history order, rolling back the action
    # of each secant pair. ``alpha`` is saved for the forward sweep.
    alphas = jnp.zeros((hist_count,), dtype=q.dtype)
    for i in range(hist_count - 1, -1, -1):
        alpha_i = rho_valid[i] * jnp.dot(s_valid[i], q)
        alphas = alphas.at[i].set(alpha_i)
        q = q - alpha_i * y_valid[i]

    # Initial inverse-Hessian scaling. Nocedal recommends
    # ``gamma = <s_k, y_k> / <y_k, y_k>`` using the most recent pair.
    y_last = y_valid[-1]
    s_last = s_valid[-1]
    ys = jnp.dot(s_last, y_last)
    yy = jnp.dot(y_last, y_last)
    gamma = jnp.where(yy > 0.0, ys / yy, jnp.asarray(1.0, dtype=q.dtype))
    r = gamma * q

    # Second loop: left-to-right forward sweep.
    for i in range(hist_count):
        beta_i = rho_valid[i] * jnp.dot(y_valid[i], r)
        r = r + (alphas[i] - beta_i) * s_valid[i]
    return r


def lbfgs_minimize(
    fn: Callable[[jnp.ndarray], jnp.ndarray],
    x0: jnp.ndarray,
    max_iter: int = 100,
    history: int = 5,
    gtol: float = 1e-5,
    ftol: float = 1e-6,
    grad_mask: Optional[jnp.ndarray] = None,
    line_search_max_iter: int = 25,
    c1: float = 1e-4,
    c2: float = 0.9,
    verbose: bool = False,
    value_and_grad: Optional[Callable[[jnp.ndarray], Any]] = None,
) -> Dict[str, Any]:
    """Minimize ``fn(x)`` with L-BFGS + strong-Wolfe line search.

    Parameters
    ----------
    fn
        Scalar-output callable taking a flat ``jax.Array`` and returning
        a scalar. The function is differentiated via ``jax.grad``; the
        returned ``value_and_grad`` variant is wrapped in ``jax.jit``
        internally to avoid recompilation across iterations.
    x0
        Initial flat iterate.
    max_iter
        Maximum outer iterations.
    history
        L-BFGS memory depth (number of ``(s, y)`` pairs kept).
    gtol
        Convergence tolerance on the infinity norm of the gradient.
    ftol
        Relative function-value change tolerance (``|df| <= ftol *
        (|f| + |f_prev| + 1e-10)``).
    grad_mask
        Optional ``bool`` or ``float`` mask of the same shape as ``x0``.
        Gradients are multiplied by the mask before every step and the
        resulting search direction is masked too — equivalent to
        restricting optimization to the coordinates where ``mask=True``.
    line_search_max_iter, c1, c2
        Passed through to :func:`strong_wolfe_line_search`.
    verbose
        If ``True``, print a one-line progress report per iteration.
    value_and_grad
        Optional pre-jitted ``x -> (value, grad)`` callable. When
        supplied, the internal ``jax.jit(jax.value_and_grad(fn))``
        wrapper is bypassed; ``fn`` is only used for display and is not
        re-differentiated. This lets callers (e.g.
        :func:`~tmol_jax.optimization.minimizers.run_cart_min`) reuse a
        score-function-level jit cache instead of paying a per-call
        compilation cost.

    Returns
    -------
    dict
        ``{"x", "fun", "grad", "n_iter", "converged", "status"}``.
        ``status`` is a short human-readable string describing why the
        loop terminated.
    """
    if max_iter < 0:
        raise ValueError("max_iter must be nonnegative")
    if history < 1:
        raise ValueError("history must be positive")
    if line_search_max_iter < 1:
        raise ValueError("line_search_max_iter must be positive")
    if gtol < 0 or ftol < 0:
        raise ValueError("gtol and ftol must be nonnegative")
    if not (0.0 < c1 < c2 < 1.0):
        raise ValueError("expected 0 < c1 < c2 < 1")
    x = jnp.asarray(x0)
    if x.ndim != 1 or not jnp.issubdtype(x.dtype, jnp.floating):
        raise TypeError("x0 must be a one-dimensional floating array")
    dtype = x.dtype
    n = int(x.shape[0])

    # Mask handling: cast once to avoid a per-iteration dtype cast cost.
    if grad_mask is None:
        mask = None
    else:
        mask = jnp.asarray(grad_mask, dtype=dtype)
        if mask.shape != x.shape:
            raise ValueError("grad_mask must have the same shape as x0")

    # Gradient pipeline. We jit the fused (value, grad) so the energy
    # function compiles once; subsequent iterations reuse the trace.
    # Callers can pass a pre-jitted ``value_and_grad`` (e.g. produced by
    # ``ScoreFunction.jitted_value_and_grad``) to reuse an existing
    # compile cache and skip the internal wrapping entirely.
    if value_and_grad is None:
        value_and_grad_fn = jax.jit(jax.value_and_grad(fn))
    else:
        value_and_grad_fn = value_and_grad

    def f_and_grad(y: jnp.ndarray):
        fy, gy = value_and_grad_fn(y)
        if mask is not None:
            gy = gy * mask
        return fy, gy

    # Initial evaluation. ``f_and_grad`` masks the gradient internally.
    f_val, grad = f_and_grad(x)
    if not bool(jnp.isfinite(f_val)) or not bool(jnp.isfinite(grad).all()):
        raise FloatingPointError("initial objective value and gradient must be finite")

    # Ring-buffer storage for history.
    s_history = jnp.zeros((history, n), dtype=dtype)
    y_history = jnp.zeros((history, n), dtype=dtype)
    rho_history = jnp.zeros((history,), dtype=dtype)
    hist_count = 0

    n_iter = 0
    converged = False
    status = "max_iter"

    # Initial gradient-norm check: skip the whole loop if we're already there.
    if float(jnp.max(jnp.abs(grad))) <= gtol:
        return {
            "x": x,
            "fun": f_val,
            "grad": grad,
            "n_iter": 0,
            "converged": True,
            "status": "initial_gtol",
        }

    # Initial step scaling heuristic: ``alpha0 = min(1, 1/||grad||_1)``.
    # This keeps the first trial step modest on a cold start.
    g_norm1 = float(jnp.sum(jnp.abs(grad)))
    alpha0 = min(1.0, 1.0 / g_norm1) if g_norm1 > 0.0 else 1.0

    for it in range(1, max_iter + 1):
        n_iter = it

        # Quasi-Newton search direction.
        d = _two_loop_recursion(grad, s_history, y_history, rho_history, hist_count)
        if mask is not None:
            d = d * mask

        # Safety: if the direction is not a descent direction, flip to
        # steepest descent. Matches the "reset Hessian" logic upstream.
        dphi0 = float(jnp.sum(grad * d))
        if not (dphi0 < 0.0):
            d = -grad
            if mask is not None:
                d = d * mask
            dphi0 = float(jnp.sum(grad * d))
            # If masked gradient is zero, nothing to do.
            if dphi0 == 0.0:
                converged = True
                status = "zero_grad"
                break

        alpha, f_new, g_new = strong_wolfe_line_search(
            f_and_grad,
            x,
            d,
            f_val,
            grad,
            alpha0=alpha0,
            c1=c1,
            c2=c2,
            max_iter=line_search_max_iter,
        )

        if alpha == 0.0:
            # Line search failed. Try a final steepest-descent attempt.
            d = -grad
            if mask is not None:
                d = d * mask
            alpha, f_new, g_new = strong_wolfe_line_search(
                f_and_grad,
                x,
                d,
                f_val,
                grad,
                alpha0=min(1.0, alpha0),
                c1=c1,
                c2=c2,
                max_iter=line_search_max_iter,
            )
            if alpha == 0.0:
                status = "line_search_failed"
                break

        # Update position and gradient. Apply mask to ensure no drift
        # accumulates on frozen coords.
        step = alpha * d
        x_new = x + step
        if mask is not None:
            # Rebuild x_new by keeping frozen entries at their previous value.
            x_new = jnp.where(mask != 0.0, x_new, x)

        # History update: store (s, y) where y must satisfy y.s > 0 to
        # keep the Hessian positive definite. Skip the update otherwise,
        # matching upstream (and Nocedal's recommendation).
        s = x_new - x
        y = g_new - grad
        if mask is not None:
            s = s * mask
            y = y * mask
        ys = float(jnp.sum(y * s))
        if ys > 1e-10:
            if hist_count < history:
                s_history = s_history.at[hist_count].set(s)
                y_history = y_history.at[hist_count].set(y)
                rho_history = rho_history.at[hist_count].set(1.0 / ys)
                hist_count += 1
            else:
                # Ring-buffer shift. Roll so the newest pair ends up at
                # index ``history - 1`` and the oldest drops off.
                s_history = jnp.concatenate([s_history[1:], s[None, :]], axis=0)
                y_history = jnp.concatenate([y_history[1:], y[None, :]], axis=0)
                rho_history = jnp.concatenate(
                    [rho_history[1:], jnp.asarray([1.0 / ys], dtype=dtype)], axis=0
                )

        if verbose:
            print(
                f"  iter {it:4d}  E={float(f_new):12.4f}"
                f"  max_grad={float(jnp.max(jnp.abs(g_new))):10.4e}"
                f"  step={alpha:.4e}  hist={hist_count}"
            )

        # Slide state forward.
        x = x_new
        f_prev = f_val
        f_val = f_new
        grad = g_new

        # After the first accepted step, strong-Wolfe defaults to alpha0=1
        # (the quasi-Newton natural unit step).
        alpha0 = 1.0

        # ------------------------------------------------------------
        # Convergence checks (gradient, relative f-change).
        # ------------------------------------------------------------
        max_grad = float(jnp.max(jnp.abs(grad)))
        if max_grad <= gtol:
            converged = True
            status = "gtol"
            break

        df = abs(float(f_val) - float(f_prev))
        denom = abs(float(f_val)) + abs(float(f_prev)) + 1e-10
        if 2.0 * df <= ftol * denom:
            converged = True
            status = "ftol"
            break

    return {
        "x": x,
        "fun": f_val,
        "grad": grad,
        "n_iter": n_iter,
        "converged": converged,
        "status": status,
    }


__all__ = ["lbfgs_minimize"]
