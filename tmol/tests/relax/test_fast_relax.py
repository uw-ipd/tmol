import torch
import numpy
import pytest

from tmol.relax.fast_relax import _default_cart_min_fn, fast_relax
import time

from tmol.pose.pose_stack_builder import PoseStackBuilder
from tmol.score.score_function import ScoreFunction
from tmol.score.score_types import ScoreType

from tmol.pack.packer_task import PackerPalette
from tmol.pack.rotamer.fixed_aa_chi_sampler import FixedAAChiSampler
from tmol.kinematics.move_map import CartesianMoveMap, MoveMap
from tmol.kinematics.fold_forest import EdgeType, FoldForest

from tmol.io import pose_stack_from_pdb


def get_relax_sfxn(default_database, torch_device):
    sfxn = ScoreFunction(param_db=default_database, device=torch_device)
    sfxn.set_weight(ScoreType.fa_ljatr, 1.0)
    sfxn.set_weight(ScoreType.fa_ljrep, 0.55)
    sfxn.set_weight(ScoreType.fa_lk, 1.0)
    sfxn.set_weight(ScoreType.fa_elec, 1.0)
    sfxn.set_weight(ScoreType.hbond, 1.0)
    sfxn.set_weight(ScoreType.hbond, 1.0)
    sfxn.set_weight(ScoreType.lk_ball_iso, -0.38)
    sfxn.set_weight(ScoreType.lk_ball, 0.92)
    sfxn.set_weight(ScoreType.lk_bridge, -0.33)
    sfxn.set_weight(ScoreType.lk_bridge_uncpl, -0.33)
    sfxn.set_weight(ScoreType.dunbrack_rot, 0.76)
    sfxn.set_weight(ScoreType.dunbrack_rotdev, 0.69)
    sfxn.set_weight(ScoreType.dunbrack_semirot, 0.78)
    sfxn.set_weight(ScoreType.cart_lengths, 0.5)
    sfxn.set_weight(ScoreType.cart_angles, 0.5)
    sfxn.set_weight(ScoreType.cart_torsions, 0.5)
    sfxn.set_weight(ScoreType.cart_impropers, 0.5)
    sfxn.set_weight(ScoreType.cart_hxltorsions, 0.5)
    sfxn.set_weight(ScoreType.omega, 0.48)
    sfxn.set_weight(ScoreType.rama, 0.50)
    sfxn.set_weight(ScoreType.ref, 1.0)
    sfxn.set_weight(ScoreType.disulfide, 1.0)

    return sfxn


@pytest.mark.parametrize("n_poses", [1])
def test_fast_relax_ubq(default_database, ubq_pdb, dun_sampler, torch_device, n_poses):
    # if torch_device == torch.device("cpu"):
    #    return

    p = pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=76)

    pose_stack = PoseStackBuilder.from_poses([p] * n_poses, torch_device)
    sfxn = get_relax_sfxn(default_database, torch_device)
    restype_set = pose_stack.packed_block_types.restype_set

    mm = MoveMap.from_pose_stack(pose_stack)
    mm.move_all_jumps = True
    mm.move_all_named_torsions = True

    palette = PackerPalette(restype_set)
    fold_forest = FoldForest.reasonable_fold_forest(pose_stack)

    def task_op(task):
        task.restrict_to_repacking()
        task.set_include_current()

        fixed_sampler = FixedAAChiSampler()
        task.add_conformer_sampler(dun_sampler)
        task.add_conformer_sampler(fixed_sampler)

    start_time = time.perf_counter()

    verbose = True
    new_pose_stack = fast_relax(
        pose_stack,
        sfxn,
        palette,
        mm,
        fold_forest,
        task_operations=[task_op],
        num_repeats=1,
        verbose=verbose,
    )

    if torch_device == torch.device("cuda"):
        torch.cuda.synchronize()
    stop_time = time.perf_counter()

    elapsed_time = stop_time - start_time

    print(f"1ubq {n_poses} FastRelax Execution time: {elapsed_time:.6f} seconds")


@pytest.mark.parametrize("n_poses", [1])
def test_cart_relax_ubq(default_database, ubq_pdb, dun_sampler, torch_device, n_poses):
    """Cartesian fast-relax on ubiquitin using CartesianSfxnNetwork.

    Mirrors test_fast_relax_ubq but swaps the kinematic min_fn for the
    Cartesian one via _default_cart_min_fn + CartesianMoveMap.
    """
    if torch_device == torch.device("cpu"):
        pytest.skip("CUDA only test")

    p = pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=76)

    pose_stack = PoseStackBuilder.from_poses([p] * n_poses, torch_device)
    sfxn = get_relax_sfxn(default_database, torch_device)
    restype_set = pose_stack.packed_block_types.restype_set

    # CartesianMoveMap with coord_mask=None moves all atoms.
    cart_mm = CartesianMoveMap()
    palette = PackerPalette(restype_set)
    fold_forest = FoldForest.reasonable_fold_forest(pose_stack)

    def task_op(task):
        task.restrict_to_repacking()
        task.set_include_current()

        fixed_sampler = FixedAAChiSampler()
        task.add_conformer_sampler(dun_sampler)
        task.add_conformer_sampler(fixed_sampler)

    start_time = time.perf_counter()

    verbose = True
    new_pose_stack = fast_relax(
        pose_stack,
        sfxn,
        palette,
        cart_mm,
        fold_forest,
        task_operations=[task_op],
        min_fn=_default_cart_min_fn,
        num_repeats=1,
        verbose=verbose,
    )

    torch.cuda.synchronize()
    stop_time = time.perf_counter()

    elapsed_time = stop_time - start_time

    print(f"1ubq {n_poses} CartRelax Execution time: {elapsed_time:.6f} seconds")


@pytest.mark.parametrize("n_poses", [1])
def test_fast_relax_pertuz(
    default_database, erbb2_and_pertuzumab_pdb, dun_sampler, torch_device, n_poses
):

    if torch_device == torch.device("cpu"):
        pytest.skip("CUDA only test")

    res_not_connected = torch.zeros(
        (1, 564 - 9 + 214 + 216 + 6, 2), dtype=torch.bool, device=torch_device
    )
    res_not_connected[0, 100, 1] = True
    res_not_connected[0, 101, 0] = True

    p = pose_stack_from_pdb(
        erbb2_and_pertuzumab_pdb, torch_device, res_not_connected=res_not_connected
    )

    pose_stack = PoseStackBuilder.from_poses([p] * n_poses, torch_device)
    sfxn = get_relax_sfxn(default_database, torch_device)
    restype_set = pose_stack.packed_block_types.restype_set

    edges = numpy.array(
        [
            [EdgeType.root_jump, -1, 0, 0],
            [EdgeType.polymer, 0, 100, 0],
            [EdgeType.jump, 0, 101, 0],
            [EdgeType.polymer, 101, 563 - 9, 0],
            [EdgeType.jump, 0, 563 - 9 + 1, 1],
            [EdgeType.jump, 0, 563 - 9 + 214 + 1, 2],
            [EdgeType.polymer, 563 - 9 + 1, 563 - 9 + 214, 0],
            [EdgeType.polymer, 563 - 9 + 214 + 1, 563 - 9 + 214 + 216 + 6, 0],
        ],
        dtype=numpy.int32,
    )
    edges = numpy.tile(edges, (n_poses, 1, 1)).reshape(n_poses, -1, 4)

    fold_forest = FoldForest.from_edges(edges)
    mm = MoveMap.from_pose_stack(pose_stack)
    # keep Jump 0 fixed, as this is the jump connecting the ends of the missing loop and we don't want that moving
    mm.set_move_all_jump_dofs_for_jump(
        torch.arange(n_poses, dtype=torch.int64, device=torch_device), 1
    )
    mm.set_move_all_jump_dofs_for_jump(
        torch.arange(n_poses, dtype=torch.int64, device=torch_device), 2
    )
    mm.move_all_named_torsions = True

    palette = PackerPalette(restype_set)

    def task_op(task):
        task.restrict_to_repacking()
        task.set_include_current()

        fixed_sampler = FixedAAChiSampler()
        task.add_conformer_sampler(dun_sampler)
        task.add_conformer_sampler(fixed_sampler)

    start_time = time.perf_counter()

    verbose = True
    new_pose_stack = fast_relax(
        pose_stack,
        sfxn,
        palette,
        mm,
        fold_forest,
        task_operations=[task_op],
        num_repeats=1,
        verbose=verbose,
    )

    if torch_device == torch.device("cuda"):
        torch.cuda.synchronize()
    stop_time = time.perf_counter()

    elapsed_time = stop_time - start_time

    print(f"1s78 {n_poses} FastRelax Execution time: {elapsed_time:.6f} seconds")


def test_fast_relax_for_different_shapes(
    ubq_pdb, erbb2_and_pertuzumab_pdb, default_database, dun_sampler, torch_device
):
    if torch_device == torch.device("cpu"):
        pytest.skip("CUDA only test")

    res_not_connected = torch.zeros((1, 40, 2), dtype=torch.bool, device=torch_device)
    res_not_connected[0, 0, 0] = True
    res_not_connected[0, 39, 1] = True

    p1 = pose_stack_from_pdb(
        ubq_pdb,
        torch_device,
        residue_start=10,
        residue_end=50,
        res_not_connected=res_not_connected,
    )
    p2 = pose_stack_from_pdb(ubq_pdb, torch_device)

    res_not_connected3 = torch.zeros(
        (1, 564 - 9 + 214 + 216 + 6, 2), dtype=torch.bool, device=torch_device
    )
    res_not_connected3[0, 100, 1] = True
    res_not_connected3[0, 101, 0] = True

    p3 = pose_stack_from_pdb(
        erbb2_and_pertuzumab_pdb, torch_device
    )  # lets pretend residues 100 and 101 are connected.

    pose_stack = PoseStackBuilder.from_poses([p1, p2, p3], torch_device)
    sfxn = get_relax_sfxn(default_database, torch_device)
    restype_set = pose_stack.packed_block_types.restype_set

    fold_forest = FoldForest.reasonable_fold_forest(pose_stack)
    mm = MoveMap.from_pose_stack(pose_stack)
    mm.move_all_jumps = True
    mm.move_all_named_torsions = True

    palette = PackerPalette(restype_set)

    def task_op(task):
        task.restrict_to_repacking()
        task.set_include_current()

        fixed_sampler = FixedAAChiSampler()
        task.add_conformer_sampler(dun_sampler)
        task.add_conformer_sampler(fixed_sampler)

    start_time = time.perf_counter()

    verbose = True
    new_pose_stack = fast_relax(
        pose_stack,
        sfxn,
        palette,
        mm,
        fold_forest,
        task_operations=[task_op],
        num_repeats=1,
        verbose=verbose,
    )

    if torch_device == torch.device("cuda"):
        torch.cuda.synchronize()
    stop_time = time.perf_counter()

    elapsed_time = stop_time - start_time

    print(
        f"Three differently-shaped PDBs relaxed; Execution time: {elapsed_time:.6f} seconds"
    )
