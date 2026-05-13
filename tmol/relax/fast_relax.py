import attr
import torch
import time
import warnings

from tmol.pose.pose_stack import PoseStack
from tmol.score.score_function import ScoreFunction
from typing import Optional, Union

from tmol.kinematics.move_map import CartesianMoveMap, MoveMap
from tmol.kinematics.fold_forest import FoldForest
from tmol.pack.pack_rotamers import pack_rotamers
from tmol.pack.packer_task import PackerPalette, PackerTask
from tmol.pack.rotamer.fixed_aa_chi_sampler import FixedAAChiSampler
from tmol.pack.rotamer.include_current_sampler import IncludeCurrentSampler
from tmol.score.score_types import ScoreType
from tmol.optimization.minimizers import run_cart_min, run_kin_min

# Default schedule from Jack Maguire's tuned MonomerRelax2019.txt.
# Each entry specifies fa_rep scale fractions for the packing and minimization
# stages of a single pack-min step.  These fractions are multiplied by the
# starting fa_rep weight to obtain the absolute weights used during each stage.
#
# Entries may be:
#   - A single float: used as both the pack and min fraction.
#     e.g.  0.559  =>  pack at 0.559 * fa_rep_start, min at 0.559 * fa_rep_start
#   - A dict with keys "fa_rep_pack_frac", "fa_rep_min_frac", "cst_frac": allows the
#     packing and minimization stages to use different fa_rep fractions and to
#     specify a constraint weight.
#     e.g.  {"fa_rep_pack_frac": 0.040, "fa_rep_min_frac": 0.051, "cst_frac": 0.5}
#
# The default behavior in Rosetta3 is to ramp constraints unless you explicitly
# say that you should not, and that is preserved here.
#
# Both forms may be mixed freely within a single schedule list.
DEFAULT_RELAX_SCHEDULE = [
    {"fa_rep_pack_frac": 0.040, "fa_rep_min_frac": 0.051, "cst_frac": 1.0},
    {"fa_rep_pack_frac": 0.265, "fa_rep_min_frac": 0.280, "cst_frac": 0.5},
    {"fa_rep_pack_frac": 0.559, "fa_rep_min_frac": 0.581, "cst_frac": 0.0},
    {"fa_rep_pack_frac": 1.000, "fa_rep_min_frac": 1.000, "cst_frac": 0.0},
]


def _normalize_schedule(
    schedule, constrain: bool = False, ramp_constraints: bool = False
):
    """Normalize a relax schedule into a list of (pack_frac, min_frac, constraint) tuples.

    Accepts a list of schedule entries in either simple or complex format:

    Simple format (float):
        A single number used as both the pack and min fa_rep fraction and a
        constraint weight determined by the constrain and ramp_constraints flags.
        Example: ``0.559`` becomes ``(0.559, 0.559, 0.0)``.

    Complex format (dict):
        A dict with keys ``"fa_rep_pack_frac"``, ``"fa_rep_min_frac"``, and ``"cst_frac"``
        specifying separate fa_rep fractions for the packing and minimization stages.
        Example: ``{"fa_rep_pack_frac": 0.040, "fa_rep_min_frac": 0.051, "cst_frac": 0.5}``
        becomes ``(0.040, 0.051, 0.5)``.

    Both formats may be freely mixed within a single schedule list.

    Args:
        schedule: List of floats and/or dicts.

    Returns:
        List of (pack_frac, min_frac, cst_frac) tuples.

    Raises:
        ValueError: If an entry is neither a float/int nor a dict with the
            required keys.
    """

    def constraint_fraction(step_index):
        """Determine the constraint fraction for a particular step.

        If ramping, ramp constraint frations down from 1.0 to 0.0 over
        the first half of the schedule, then keep at 0.0.
        """
        if not constrain:
            return 0.0
        if not ramp_constraints:
            return 1.0
        n_steps = len(schedule)
        if step_index > n_steps // 2:
            return 0
        else:
            return 1 - step_index / (n_steps / 2)

    normalized = []
    for i, entry in enumerate(schedule):
        if isinstance(entry, (int, float)):
            constraint = constraint_fraction(i)
            normalized.append((float(entry), float(entry), constraint))
        elif isinstance(entry, dict):
            pack_frac = float(entry["fa_rep_pack_frac"])
            min_frac = float(entry["fa_rep_min_frac"])
            constraint = (
                constraint_fraction(i)
                if "cst_frac" not in entry
                else float(entry["cst_frac"])
            )
            normalized.append((pack_frac, min_frac, constraint))
        else:
            raise ValueError(
                f"Schedule entry must be a number or a dict with keys"
                f" 'fa_rep_pack_frac', 'fa_rep_min_frac', and optionally 'cst_frac', got {type(entry)}"
            )
    return normalized


def _default_min_fn(pose_stack, sfxn, *, fold_forest, move_map, verbose):
    """Default minimization function: kinematic (torsion-space) LBFGS."""
    return run_kin_min(
        pose_stack,
        sfxn,
        fold_forest,
        move_map,
        verbose=verbose,
        optimizer_kwargs={"verbose": verbose},
    )


def _default_cart_min_fn(pose_stack, sfxn, *, fold_forest, move_map, verbose):
    """Default Cartesian minimization function for use as fast_relax min_fn.

    Extracts ``coord_mask`` from ``move_map`` if it is a
    :class:`~tmol.kinematics.move_map.CartesianMoveMap`; otherwise all atoms
    are free to move.  ``fold_forest`` is accepted but ignored.

    Example usage with :func:`fast_relax`::

        fast_relax(pose_stack, sfxn, palette, CartesianMoveMap(), fold_forest,
                   min_fn=_default_cart_min_fn, ...)
    """
    coord_mask = move_map.coord_mask if isinstance(move_map, CartesianMoveMap) else None
    return run_cart_min(
        pose_stack,
        sfxn,
        coord_mask=coord_mask,
        verbose=verbose,
        optimizer_kwargs={"verbose": verbose},
    )


def fast_relax(
    pose_stack: PoseStack,
    sfxn: ScoreFunction,
    packer_pallete: PackerPalette,
    move_map: Union[MoveMap, CartesianMoveMap],
    fold_forest: FoldForest,
    *,
    task_operations=None,
    num_repeats=5,
    ramp_constraints: Optional[bool] = None,  # default True
    schedule=None,
    min_fn=None,
    verbose: bool = False,
):
    """Run the FastRelax protocol: repeated rounds of rotamer packing and
    gradient minimization with a ramped fa_rep weight schedule.

    This implements Jack Maguire's tuned MonomerRelax2019 protocol.  Each
    repeat consists of several pack-min steps at increasing fa_rep fractions,
    followed by an accept-to-best check.

    Args:
        pose_stack: The input poses to relax.
        sfxn: Score function used for packing and minimization. If you wish
            to use constraints during relax, then the weight on the "constraint"
            score type must already have a non-zero value.
        packer_pallete: Palette defining the residue types available to the
            packer.
        move_map: Specifies which DOFs are free to move during minimization.
        fold_forest: Fold forest defining the kinematic connectivity.
        task_operations: List of callables that configure a PackerTask.  Each
            callable receives a PackerTask and modifies it in place (e.g. to
            restrict to repacking or add chi samplers).  If None, a default
            operation is created that restricts to repacking with Dunbrack
            rotamers and includes the current rotamer.
        ramp_constraints: If True, decrease the constraing weight over the first
            half of the weight-ramping schedule from its starting value to 0.
            If False, use the starting constraint weight for the entirety
            of relax. The weight on the "constraint" term in the input sfxn
            will be restored to its starting value at the end of relax.
            Default: True. A warning message is printed if you specify
            ramp_constraints=True but the "constraint" weight is 0.
        num_repeats: Number of times to repeat the full schedule of pack-min
            steps (default: 5).
        schedule: The fa_rep / constraint ramp schedule — a list of per-step entries
            controlling the fa_rep weight and constraints used during packing and
            minimization. Each entry is either:

            - A **float**: used as the fa_rep fraction for both packing and
              minimization.  E.g. ``0.559`` means both stages run at
              ``0.559 * fa_rep_start`` (specifying nothing for the constraints).
            - A **dict** with keys ``"fa_rep_pack_frac"``,
              ``"fa_rep_min_frac"`` and optionally ``"cst_frac"``:
              allows different fractions for the two
              stages and an optional constraint fraction.  E.g. ``{"fa_rep_pack_frac": 0.040, "fa_rep_min_frac":
              0.051, "cst_frac": 0.5}``.

            Both formats may be mixed within a single list.  All fractions are
            multiplied by the starting ``fa_rep`` weight / ``constraint`` weight
            from ``sfxn``.

            If None, ``DEFAULT_RELAX_SCHEDULE`` is used (the 4-step ramp from
            MonomerRelax2019).
        min_fn: Callable used to minimize the pose at each pack-min step.
            Must have the signature::

                min_fn(pose_stack, sfxn, *, fold_forest, move_map, verbose)
                    -> PoseStack

            The ``fold_forest`` and ``move_map`` are passed as keyword
            arguments so that Cartesian minimizers can accept and ignore them
            via ``**kwargs``.

            If None, the default kinematic (torsion-space) LBFGS minimizer
            is used (``run_kin_min``).

            Examples::

                # Cartesian minimization:
                min_fn=lambda ps, sfxn, **kw: run_cart_min(ps, sfxn)

                # Kinematic minimization with torch's LBFGS + strong Wolfe:
                def my_min(ps, sfxn, *, fold_forest, move_map, **kw):
                    return run_kin_min(
                        ps, sfxn, fold_forest, move_map,
                        optimizer_cls=torch.optim.LBFGS,
                        optimizer_kwargs={
                            "line_search_fn": "strong_wolfe",
                        },
                    )

        verbose: Print timing information for each step.

    Returns:
        The relaxed PoseStack (best-scoring across all repeats).
    """
    if min_fn is None:
        min_fn = _default_min_fn
    if schedule is None:
        schedule = DEFAULT_RELAX_SCHEDULE

    # Logic for using / adding constraints
    constraint_weight_start = sfxn.get_weight(ScoreType.constraint)
    use_constraints = constraint_weight_start != 0
    if not use_constraints and ramp_constraints:
        print(
            "Warning: ramp_constraints is True but sfxn's 'constraint' weight is 0; no constraints will be used."
        )
    if ramp_constraints is None:
        ramp_constraints = True

    steps = _normalize_schedule(schedule, use_constraints, ramp_constraints)

    if len(steps) == 0:
        raise ValueError("Relax schedule must contain at least one step.")

    # Warn if the final step doesn't restore fa_rep to its full weight.
    final_min_frac = steps[-1][1]
    if abs(final_min_frac - 1.0) > 1e-6:
        warnings.warn(
            f"Final schedule step has fa_rep min fraction {final_min_frac:.4f},"
            f" not 1.0!",
            stacklevel=2,
        )

    if task_operations is None:
        # Create a default task operation that
        # 1. builds rotamers using the Dunbrack rotamer library
        # 2. restricts to repacking
        # 3. includes the current rotamer

        torch_device = pose_stack.device
        from tmol.pack.rotamer.dunbrack.dunbrack_chi_sampler import (
            create_dunbrack_sampler_from_database,
        )
        import tmol.database

        default_database = tmol.database.ParameterDatabase.get_default()
        dun_sampler = create_dunbrack_sampler_from_database(
            default_database, torch_device
        )

        def default_op(task):
            task.restrict_to_repacking()
            task.or_bump_check(True)

            fixed_sampler = FixedAAChiSampler()
            task.add_conformer_sampler(dun_sampler)
            task.add_conformer_sampler(fixed_sampler)
            task.add_conformer_sampler(IncludeCurrentSampler())

        task_operations = [default_op]

    fa_rep_start = float(sfxn.get_weight(ScoreType.fa_ljrep))

    wpsm = sfxn.render_whole_pose_scoring_module(pose_stack)
    best_score = wpsm(pose_stack.coords)
    best_ps = pose_stack.clone()

    ps = pose_stack
    for _ in range(num_repeats):
        for pack_frac, min_frac, cst_frac in steps:
            ps = relax_pack_min_step(
                pose_stack=ps,
                sfxn=sfxn,
                fold_forest=fold_forest,
                move_map=move_map,
                packer_pallete=packer_pallete,
                fa_rep_pack_weight=pack_frac * fa_rep_start,
                fa_rep_min_weight=min_frac * fa_rep_start,
                cst_weight=cst_frac * constraint_weight_start,
                task_operations=task_operations,
                min_fn=min_fn,
                verbose=verbose,
            )

        best_ps, best_score = accept_best(sfxn, best_ps, best_score, ps, verbose)
        ps = best_ps.clone()
    if use_constraints:
        # Restore original constraint weight to the score function
        sfxn.set_weight(ScoreType.constraint, constraint_weight_start)
    return ps


def relax_pack_min_step(
    pose_stack,
    sfxn,
    fold_forest,
    move_map: Union[MoveMap, CartesianMoveMap],
    packer_pallete,
    fa_rep_pack_weight,
    fa_rep_min_weight,
    cst_weight,
    task_operations,
    min_fn,
    verbose,
):

    if verbose and torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    task = PackerTask(pose_stack, packer_pallete)
    for op in task_operations:
        op(task)

    sfxn.set_weight(ScoreType.fa_ljrep, fa_rep_pack_weight)
    sfxn.set_weight(ScoreType.constraint, cst_weight)
    if verbose:
        print(
            f"packing with fa_rep of {fa_rep_pack_weight: .2f} and constraint weight of {cst_weight: .2f}"
        )
    if verbose and torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time1 = time.perf_counter()
    packed_pose_stack = pack_rotamers(pose_stack, sfxn, task, verbose)

    sfxn.set_weight(ScoreType.fa_ljrep, fa_rep_min_weight)
    if verbose:
        print(
            f"minimizing with fa_rep of {fa_rep_min_weight: .2f} and constraint weight of {cst_weight: .2f}"
        )
    if verbose and torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time2 = time.perf_counter()
    minimized_pose_stack = min_fn(
        packed_pose_stack,
        sfxn,
        fold_forest=fold_forest,
        move_map=move_map,
        verbose=verbose,
    )
    if verbose and torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time3 = time.perf_counter()

    if verbose:
        print(
            f"pack-min {end_time3 - start_time: .2f} task-init {end_time1 - start_time: .2f}"
            + f" packing {end_time2 - end_time1: .2f} min {end_time3 - end_time2: .2f}"
        )

    return minimized_pose_stack


def accept_best(
    sfxn: ScoreFunction,
    best_pose_stack: PoseStack,
    best_pose_score: torch.Tensor,
    candidate_pose_stack: PoseStack,
    verbose=False,
):
    wpsm = sfxn.render_whole_pose_scoring_module(candidate_pose_stack)
    candidate_score = wpsm(candidate_pose_stack.coords)
    better_mask = candidate_score < best_pose_score

    def select_better(tensor_name):
        tensor = getattr(best_pose_stack, tensor_name)
        new_tensor = tensor.detach().clone()
        new_tensor[better_mask] = getattr(candidate_pose_stack, tensor_name)[
            better_mask
        ]
        return new_tensor

    if better_mask.any():
        if verbose:
            print("accepting new best scores")
            print(f" old best score: {best_pose_score[better_mask]}")
            print(f" new best score: {candidate_score[better_mask]}")

        new_coords = select_better("coords")
        new_block_coord_offset = select_better("block_coord_offset")
        new_block_coord_offset64 = select_better("block_coord_offset64")
        new_block_type_ind = select_better("block_type_ind")
        new_block_type_ind64 = select_better("block_type_ind64")
        new_best_pose_stack = attr.evolve(
            best_pose_stack,
            coords=new_coords,
            block_coord_offset=new_block_coord_offset,
            block_coord_offset64=new_block_coord_offset64,
            block_type_ind=new_block_type_ind,
            block_type_ind64=new_block_type_ind64,
        )
        new_best_pose_score = best_pose_score.detach().clone()
        new_best_pose_score[better_mask] = candidate_score[better_mask]
        return new_best_pose_stack, new_best_pose_score
    else:  # no change
        return best_pose_stack, best_pose_score
