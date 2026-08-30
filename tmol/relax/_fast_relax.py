import time
import warnings
from collections.abc import Callable, Sequence
from typing import Protocol

import attr
import torch

from tmol.pose import PoseStack
from tmol.score import (
    ScoreFunction,
    ScoreType,
)
from tmol.kinematics import (
    CartesianMoveMap,
    MoveMap,
    FoldForest,
)
from tmol.pack import (
    pack_rotamers,
    PackerPalette,
    PackerTask,
)
from tmol.pack.rotamer import (
    FixedAAChiSampler,
    IncludeCurrentSampler,
)
from tmol.optimization import run_cart_min, run_kin_min
from tmol.types import Tensor
from tmol.utility._device import synchronize_device

RelaxScheduleEntry = float | int | dict[str, float]
PackerTaskOperation = Callable[[PackerTask], None]


class RelaxMinimizer(Protocol):
    """Callable contract for a FastRelax minimization stage."""

    def __call__(
        self,
        pose_stack: PoseStack,
        sfxn: ScoreFunction,
        *,
        fold_forest: FoldForest,
        move_map: MoveMap | CartesianMoveMap,
        verbose: bool,
    ) -> PoseStack:
        """Minimize and return the updated poses."""


# Jack Maguire's tuned MonomerRelax2019 schedule. Fractions scale the score
# function's initial fa_rep and constraint weights at each pack-minimize stage.
DEFAULT_RELAX_SCHEDULE = [
    {"fa_rep_pack_frac": 0.040, "fa_rep_min_frac": 0.051, "cst_frac": 1.0},
    {"fa_rep_pack_frac": 0.265, "fa_rep_min_frac": 0.280, "cst_frac": 0.5},
    {"fa_rep_pack_frac": 0.559, "fa_rep_min_frac": 0.581, "cst_frac": 0.0},
    {"fa_rep_pack_frac": 1.000, "fa_rep_min_frac": 1.000, "cst_frac": 0.0},
]


def _normalize_schedule(
    schedule: Sequence[RelaxScheduleEntry],
    constrain: bool = False,
    ramp_constraints: bool = False,
) -> list[tuple[float, float, float]]:
    """Normalize user schedule entries into pack, minimize, and constraint weights.

    Args:
        schedule: Numbers, used for both fa_rep stages, or dictionaries with
            ``fa_rep_pack_frac``, ``fa_rep_min_frac``, and optional ``cst_frac``.
        constrain: Whether the score function has an active constraint term.
        ramp_constraints: Whether to ramp its weight to zero.

    Returns:
        ``(pack_fraction, minimize_fraction, constraint_fraction)`` per step.

    Raises:
        ValueError: If an entry is neither a float/int nor a dict with the
            required keys.
    """

    def constraint_fraction(step_index: int) -> float:
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
            return 0.0
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


def _default_kin_min_fn(
    pose_stack: PoseStack,
    sfxn: ScoreFunction,
    *,
    fold_forest: FoldForest,
    move_map: MoveMap | CartesianMoveMap,
    verbose: bool,
) -> PoseStack:
    """Default minimization function: kinematic (torsion-space) LBFGS."""
    return run_kin_min(
        pose_stack,
        sfxn,
        fold_forest,
        move_map,
        verbose=verbose,
        optimizer_kwargs={"verbose": verbose},
    )


def _default_cart_min_fn(
    pose_stack: PoseStack,
    sfxn: ScoreFunction,
    *,
    fold_forest: FoldForest,
    move_map: MoveMap | CartesianMoveMap,
    verbose: bool,
) -> PoseStack:
    """Run Cartesian LBFGS using a Cartesian move map's coordinate mask."""
    coord_mask = move_map.coord_mask if isinstance(move_map, CartesianMoveMap) else None
    return run_cart_min(
        pose_stack,
        sfxn,
        coord_mask=coord_mask,
        verbose=verbose,
        optimizer_kwargs={"verbose": verbose},
    )


def fast_relax(  # noqa: C901
    pose_stack: PoseStack,
    sfxn: ScoreFunction,
    packer_pallete: PackerPalette,
    move_map: MoveMap | CartesianMoveMap,
    fold_forest: FoldForest,
    *,
    task_operations: Sequence[PackerTaskOperation] | None = None,
    num_repeats: int = 2,
    ramp_constraints: bool | None = None,  # default True
    schedule: Sequence[RelaxScheduleEntry] | None = None,
    min_fn: RelaxMinimizer | None = None,
    verbose: bool = False,
) -> PoseStack:
    """Relax poses through repeated side-chain packing and minimization.

    Each repeat applies the MonomerRelax2019 fa_rep ramp and retains the
    lowest-scoring conformation independently for every pose in the batch.

    Args:
        pose_stack: Input poses to relax.
        sfxn: Packing and minimization score function. Constraints are active
            only when its constraint weight is nonzero.
        packer_pallete: Residue types available to the packer.
        move_map: Specifies which DOFs are free to move during minimization.
        fold_forest: Fold forest defining the kinematic connectivity.
        task_operations: In-place task configuration callbacks. By default,
            restrict to repacking with Dunbrack, fixed-AA, and current rotamers.
        num_repeats: Number of complete pack-minimize ramps.
        ramp_constraints: Ramp an active constraint weight to zero. Defaults
            to true; the input weight is restored after relaxation.
        schedule: Numeric fa_rep fractions or dictionaries with separate pack,
            minimize, and optional constraint fractions. Defaults to
            ``DEFAULT_RELAX_SCHEDULE``.
        min_fn: Minimizer called with the pose, score function, fold forest,
            move map, and verbosity. Defaults to Cartesian minimization.
        verbose: Print timing information for each step.

    Returns:
        Best-scoring relaxed poses across all repeats.
    """
    if min_fn is None:
        min_fn = _default_cart_min_fn
    if schedule is None:
        schedule = DEFAULT_RELAX_SCHEDULE

    constraint_weight_start = sfxn.get_weight(ScoreType.constraint)
    use_constraints = constraint_weight_start != 0
    if not use_constraints and ramp_constraints:
        print(
            "Warning: ramp_constraints is True but sfxn's 'constraint' weight is 0; no constraints will be used."
        )
    if ramp_constraints is None:
        ramp_constraints = True

    steps = _normalize_schedule(schedule, use_constraints, ramp_constraints)

    if not steps:
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
        torch_device = pose_stack.device
        from tmol.pack.rotamer.dunbrack import (
            create_dunbrack_sampler_from_database,
        )
        import tmol.database

        default_database = tmol.database.ParameterDatabase.get_default()
        dun_sampler = create_dunbrack_sampler_from_database(
            default_database, torch_device
        )

        def default_op(task: PackerTask) -> None:
            task.restrict_to_repacking()
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
    pose_stack: PoseStack,
    sfxn: ScoreFunction,
    fold_forest: FoldForest,
    move_map: MoveMap | CartesianMoveMap,
    packer_pallete: PackerPalette,
    fa_rep_pack_weight: float,
    fa_rep_min_weight: float,
    cst_weight: float,
    task_operations: Sequence[PackerTaskOperation],
    min_fn: RelaxMinimizer,
    verbose: bool,
) -> PoseStack:
    """Execute one weighted packing and minimization stage.

    Args:
        pose_stack: Current poses.
        sfxn: Score function whose repulsive and constraint weights are updated.
        fold_forest: Connectivity passed to the minimizer.
        move_map: Movable degrees of freedom passed to the minimizer.
        packer_pallete: Residue types available during packing.
        fa_rep_pack_weight: Repulsive weight for packing.
        fa_rep_min_weight: Repulsive weight for minimization.
        cst_weight: Constraint weight for both operations.
        task_operations: In-place task configuration callbacks.
        min_fn: Minimization callable.
        verbose: Print synchronized stage timings.

    Returns:
        The minimized pose stack.
    """

    if verbose:
        synchronize_device(pose_stack.device)
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
    if verbose:
        synchronize_device(pose_stack.device)
    end_time1 = time.perf_counter()
    packed_pose_stack = pack_rotamers(pose_stack, sfxn, task, verbose)

    sfxn.set_weight(ScoreType.fa_ljrep, fa_rep_min_weight)
    if verbose:
        print(
            f"minimizing with fa_rep of {fa_rep_min_weight: .2f} and constraint weight of {cst_weight: .2f}"
        )
    if verbose:
        synchronize_device(pose_stack.device)
    end_time2 = time.perf_counter()
    minimized_pose_stack = min_fn(
        packed_pose_stack,
        sfxn,
        fold_forest=fold_forest,
        move_map=move_map,
        verbose=verbose,
    )
    if verbose:
        synchronize_device(pose_stack.device)
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
    best_pose_score: Tensor[torch.float32][:],
    candidate_pose_stack: PoseStack,
    verbose: bool = False,
) -> tuple[PoseStack, Tensor[torch.float32][:]]:
    """Keep the lower-scoring conformation independently for each pose.

    Args:
        sfxn: Score function used for comparison.
        best_pose_stack: Best poses from previous repeats.
        best_pose_score: Best scores shaped ``[n_poses]``.
        candidate_pose_stack: Newly minimized poses.
        verbose: Print accepted scores.

    Returns:
        Updated best poses and scores shaped ``[n_poses]``.
    """
    wpsm = sfxn.render_whole_pose_scoring_module(candidate_pose_stack)
    candidate_score = wpsm(candidate_pose_stack.coords)
    better_mask = candidate_score < best_pose_score

    def select_better(tensor_name: str) -> torch.Tensor:
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
    return best_pose_stack, best_pose_score
