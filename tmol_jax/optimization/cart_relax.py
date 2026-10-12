"""Conservative Cartesian repulsive-ramp relaxation without packing."""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

from tmol_jax.score.constraint import create_mainchain_coordinate_constraints
from tmol_jax.score.score_types import ScoreType

from .minimizers import run_cart_min

DEFAULT_REPULSIVE_RAMP = (0.02, 0.25, 0.55, 1.0)


def _validated_ramp(schedule: Sequence[float]) -> tuple[float, ...]:
    ramp = tuple(float(value) for value in schedule)
    if not ramp:
        raise ValueError("ramp_schedule must not be empty")
    if any(not math.isfinite(value) or not 0.0 < value <= 1.0 for value in ramp):
        raise ValueError("ramp values must be finite and in (0, 1]")
    if any(right < left for left, right in zip(ramp, ramp[1:])):
        raise ValueError("ramp_schedule must be nondecreasing")
    if ramp[-1] != 1.0:
        raise ValueError("ramp_schedule must end at full repulsive weight (1.0)")
    return ramp


def run_cart_relax(
    pose_stack: Any,
    score_function: Any,
    move_mask: Optional[Any] = None,
    *,
    n_cycles: int = 1,
    ramp_schedule: Sequence[float] = DEFAULT_REPULSIVE_RAMP,
    constrain_mainchain: bool = False,
    constraint_k: float = 1.0,
    constraint_tol: float = 0.0,
    max_iter: int = 100,
    history: int = 5,
    gtol: float = 1e-5,
    ftol: float = 1e-6,
    line_search_max_iter: int = 25,
    verbose: bool = False,
    return_result: bool = False,
):
    """Run no-pack Cartesian minimization across an LJ-repulsive ramp.

    This is not FastRelax: it performs no rotamer packing and has no
    kinematic minimization. Each stage minimizes the same full score with only
    ``fa_ljrep`` scaled. The returned pose is the best structure observed under
    the full-weight objective, including optional mainchain constraints.

    The caller's score function is never mutated. Mainchain constraints target
    the input N, CA, and C coordinates and are weighted locally at 1.0 when the
    caller's constraint weight is zero.
    """
    if not isinstance(n_cycles, int) or isinstance(n_cycles, bool) or n_cycles < 1:
        raise ValueError("n_cycles must be a positive integer")
    ramp = _validated_ramp(ramp_schedule)
    try:
        working_score = score_function.clone()
    except AttributeError as error:
        raise TypeError("score_function must provide the TMol JAX clone API") from error

    full_repulsive_weight = working_score.get_weight(ScoreType.fa_ljrep)
    if full_repulsive_weight == 0.0:
        raise ValueError("Cartesian relaxation requires a nonzero fa_ljrep weight")

    constraint_set = None
    if constrain_mainchain:
        if not math.isfinite(constraint_k) or constraint_k < 0:
            raise ValueError("constraint_k must be finite and nonnegative")
        if not math.isfinite(constraint_tol) or constraint_tol < 0:
            raise ValueError("constraint_tol must be finite and nonnegative")
        constraint_set = create_mainchain_coordinate_constraints(
            pose_stack, k=constraint_k, tol=constraint_tol
        )
        if working_score.get_weight(ScoreType.constraint) == 0.0:
            working_score.set_weight(ScoreType.constraint, 1.0)

    full_score = working_score.clone()
    full_scorer = full_score.render_whole_pose_scoring_module(
        pose_stack, constraint_set=constraint_set
    )

    def full_energy(pose):
        value = float(full_scorer(pose.coords).sum())
        if not math.isfinite(value):
            raise FloatingPointError("relaxation produced a non-finite full score")
        return value

    current = pose_stack
    best = pose_stack
    best_energy = full_energy(pose_stack)
    initial_energy = best_energy
    best_stage = -1
    stage_results = []

    for cycle in range(n_cycles):
        for ramp_index, fraction in enumerate(ramp):
            working_score.set_weight(
                ScoreType.fa_ljrep, fraction * full_repulsive_weight
            )
            current, min_result = run_cart_min(
                current,
                working_score,
                move_mask,
                max_iter=max_iter,
                history=history,
                gtol=gtol,
                ftol=ftol,
                line_search_max_iter=line_search_max_iter,
                verbose=verbose,
                return_result=True,
                constraint_set=constraint_set,
            )
            candidate_energy = full_energy(current)
            stage_results.append(
                {
                    "cycle": cycle,
                    "ramp_index": ramp_index,
                    "repulsive_fraction": fraction,
                    "scaled_fun": float(min_result["fun"]),
                    "full_fun": candidate_energy,
                    "n_iter": int(min_result["n_iter"]),
                    "converged": bool(min_result["converged"]),
                    "status": min_result["status"],
                }
            )
            if candidate_energy < best_energy:
                best = current
                best_energy = candidate_energy
                best_stage = len(stage_results) - 1

    diagnostics = {
        "initial_fun": initial_energy,
        "fun": best_energy,
        "best_stage": best_stage,
        "stages": tuple(stage_results),
    }
    return (best, diagnostics) if return_result else best


__all__ = ["DEFAULT_REPULSIVE_RAMP", "run_cart_relax"]
