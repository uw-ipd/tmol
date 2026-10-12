"""Coordinate minimization against rendered, topology-stable JAX scorers."""

from __future__ import annotations

from typing import Any, Optional

import jax
import jax.numpy as jnp

from .lbfgs import lbfgs_minimize


def _coordinate_mask(pose_stack: Any, move_mask: Optional[Any]) -> jax.Array:
    """Return a strict xyz movement mask, always excluding padding atoms."""
    coords_shape = pose_stack.coords.shape
    atom_shape = coords_shape[:2]
    if move_mask is None:
        mask = jnp.broadcast_to(pose_stack.atom_is_real[..., None], coords_shape)
    else:
        mask = jnp.asarray(move_mask, dtype=jnp.bool_)
        if mask.shape == atom_shape:
            mask = jnp.broadcast_to(mask[..., None], coords_shape)
        elif mask.shape != coords_shape:
            raise ValueError(
                "move_mask must have shape "
                f"{atom_shape} or {coords_shape}, got {mask.shape}"
            )
        mask &= pose_stack.atom_is_real[..., None]
    return mask


def run_cart_min(
    pose_stack: Any,
    score_function: Any,
    move_mask: Optional[Any] = None,
    *,
    max_iter: int = 100,
    history: int = 5,
    gtol: float = 1e-5,
    ftol: float = 1e-6,
    line_search_max_iter: int = 25,
    verbose: bool = False,
    return_result: bool = False,
    constraint_set: Optional[Any] = None,
):
    """Minimize Cartesian coordinates with L-BFGS and strong-Wolfe search.

    The score function is rendered once because topology and atom layout stay
    fixed. ``move_mask`` may select atoms (``[poses, atoms]``) or individual
    xyz components (``[poses, atoms, 3]``); padding is always frozen. The
    returned pose is new and the input is unchanged.

    Set ``return_result=True`` to receive ``(pose, diagnostics)`` where the
    diagnostics mapping includes the final value, gradient, iteration count,
    convergence flag, and termination status.
    """
    coords = jnp.asarray(pose_stack.coords)
    if not jnp.issubdtype(coords.dtype, jnp.floating):
        raise TypeError("pose coordinates must be floating point")
    if not bool(jnp.isfinite(coords).all()):
        raise ValueError("pose coordinates must be finite")
    if any(
        getattr(term, "use_neighbor_list", False)
        for term in getattr(score_function, "terms", ())
    ):
        raise ValueError("Cartesian minimization requires a dense score function")

    try:
        scorer = score_function.render_whole_pose_scoring_module(
            pose_stack, constraint_set=constraint_set
        )
    except AttributeError as error:
        raise TypeError(
            "score_function must provide the TMol JAX render API"
        ) from error

    mask = _coordinate_mask(pose_stack, move_mask)
    shape = coords.shape
    x0 = coords.reshape(-1)
    flat_mask = mask.reshape(-1)

    def energy(x_flat):
        return scorer(x_flat.reshape(shape)).sum()

    value_and_grad = jax.jit(jax.value_and_grad(energy))
    initial_energy = float(energy(x0))
    result = lbfgs_minimize(
        energy,
        x0,
        max_iter=max_iter,
        history=history,
        gtol=gtol,
        ftol=ftol,
        grad_mask=flat_mask,
        line_search_max_iter=line_search_max_iter,
        verbose=verbose,
        value_and_grad=value_and_grad,
    )
    final_coords = result["x"].reshape(shape).astype(coords.dtype)
    final_coords = jnp.where(mask, final_coords, coords)
    if not bool(jnp.isfinite(final_coords).all()) or not bool(
        jnp.isfinite(result["fun"])
    ):
        raise FloatingPointError("minimization produced non-finite output")
    final_energy = float(result["fun"])
    tolerance = 1e-6 * max(abs(initial_energy), 1.0)
    if final_energy > initial_energy + tolerance:
        raise RuntimeError(
            "minimization increased the energy: "
            f"{initial_energy:.7g} -> {final_energy:.7g}"
        )

    minimized = pose_stack.replace(coords=final_coords)
    if return_result:
        diagnostics = dict(result)
        diagnostics["initial_fun"] = initial_energy
        return minimized, diagnostics
    return minimized


__all__ = ["run_cart_min"]
