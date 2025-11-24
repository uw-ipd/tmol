import attr
import torch
import time

from tmol.pose.pose_stack import PoseStack
from tmol.score.score_function import ScoreFunction
from tmol.kinematics.move_map import MoveMap
from tmol.kinematics.fold_forest import FoldForest
from tmol.pack.pack_rotamers import pack_rotamers
from tmol.pack.packer_task import PackerPalette, PackerTask
from tmol.pack.rotamer.fixed_aa_chi_sampler import FixedAAChiSampler
from tmol.score.score_types import ScoreType
from tmol.optimization.kin_min import run_kin_min


def fast_relax(
    pose_stack: PoseStack,
    sfxn: ScoreFunction,
    packer_pallete: PackerPalette,
    move_map: MoveMap,
    fold_forest: FoldForest,
    task_operations=None,
    verbose=False,
):
    # Jack Maguire's tuned MonomerRelax2019.txt
    # repeat %%nrepeats%%
    # coord_cst_weight 1.0
    # scale:fa_rep 0.040
    # repack
    # scale:fa_rep 0.051
    # min 0.01
    # coord_cst_weight 0.5
    # scale:fa_rep 0.265
    # repack
    # scale:fa_rep 0.280
    # min 0.01
    # coord_cst_weight 0.0
    # scale:fa_rep 0.559
    # repack
    # scale:fa_rep 0.581
    # min 0.01
    # coord_cst_weight 0.0
    # scale:fa_rep 1
    # repack
    # min 0.00001
    # accept_to_best
    # endrepeat

    if task_operations is None:
        # Create a default task operation that
        # 1. builds rotamers using the Dunbrack rotamer library
        # 2. restricts to repacking
        # 3. includes the current rotamer

        torch_device = pose_stack.device
        from tmol.score.dunbrack.params import DunbrackParamResolver
        from tmol.pack.rotamer.dunbrack.dunbrack_chi_sampler import DunbrackChiSampler
        import tmol.database

        default_database = tmol.database.ParameterDatabase.get_default()
        param_resolver = DunbrackParamResolver.from_database(
            default_database.scoring.dun, torch_device
        )
        dun_sampler = DunbrackChiSampler.from_database(param_resolver)

        def default_op(task):
            task.restrict_to_repacking()
            task.set_include_current()

            fixed_sampler = FixedAAChiSampler()
            task.add_conformer_sampler(dun_sampler)
            task.add_conformer_sampler(fixed_sampler)

        task_operations = [default_op]

    fa_rep_start = float(sfxn._weights[ScoreType.fa_ljrep.value])

    wpsm = sfxn.render_whole_pose_scoring_module(pose_stack)
    best_score = wpsm(pose_stack.coords)
    best_ps = pose_stack.clone()

    rpms_args = [
        pose_stack,
        sfxn,
        fold_forest,
        move_map,
        packer_pallete,
        0,
        0,
        task_operations,
        verbose,
    ]
    ps = pose_stack
    for _ in range(5):
        rpms_args[0] = ps
        rpms_args[5] = 0.040 * fa_rep_start
        rpms_args[6] = 0.051 * fa_rep_start
        ps = relax_pack_min_step(*rpms_args)

        rpms_args[0] = ps
        rpms_args[5] = 0.265 * fa_rep_start
        rpms_args[6] = 0.280 * fa_rep_start
        ps = relax_pack_min_step(*rpms_args)

        rpms_args[0] = ps
        rpms_args[5] = 0.559 * fa_rep_start
        rpms_args[6] = 0.581 * fa_rep_start
        ps = relax_pack_min_step(*rpms_args)

        rpms_args[0] = ps
        rpms_args[5] = fa_rep_start
        rpms_args[6] = fa_rep_start
        ps = relax_pack_min_step(*rpms_args)

        best_ps, best_score = accept_best(sfxn, best_ps, best_score, ps, verbose)
        ps = best_ps.clone()
    return ps


def relax_pack_min_step(
    pose_stack,
    sfxn,
    fold_forest,
    move_map,
    packer_pallete,
    fa_rep_pack_weight,
    fa_rep_min_weight,
    task_operations,
    verbose,
):

    if verbose and torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    task = PackerTask(pose_stack, packer_pallete)
    for op in task_operations:
        op(task)

    sfxn.set_weight(ScoreType.fa_ljrep, fa_rep_pack_weight)
    if verbose:
        print(f"packing with fa_rep of {fa_rep_pack_weight: .2f}")
    if verbose and torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time1 = time.perf_counter()
    packed_pose_stack = pack_rotamers(pose_stack, sfxn, task, verbose)

    sfxn.set_weight(ScoreType.fa_ljrep, fa_rep_min_weight)
    if verbose:
        print(f"minimizing with fa_rep of {fa_rep_min_weight: .2f}")
    if verbose and torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time2 = time.perf_counter()
    minimized_pose_stack = run_kin_min(
        packed_pose_stack, sfxn, fold_forest, move_map, verbose
    )
    if verbose and torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time3 = time.perf_counter()

    if verbose:
        print(
            f"pack-min {end_time3 - start_time: .2f} task-init {end_time1 - start_time: .2f}"
            + f" packing {end_time2 - end_time1: .2f} min {end_time3-end_time2: .2f}"
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
