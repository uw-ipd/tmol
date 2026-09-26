import time
import weakref

import numpy
import pytest
import torch

from tmol.relax import fast_relax
import tmol.relax._fast_relax as fast_relax_module
from tmol.relax._fast_relax import _resolve_cuda_graph_mode
from tmol.optimization import CartesianMinimizer

from tmol.pose import (
    PoseStack,
    PoseStackBuilder,
)
from tmol.score import (
    ScoreFunction,
    ScoreType,
)
from tmol.pack import PackerPalette
from tmol.pack.rotamer import (
    FixedAAChiSampler,
    IncludeCurrentSampler,
)
from tmol.kinematics import (
    CartesianMoveMap,
    MoveMap,
    EdgeType,
    FoldForest,
)
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


@pytest.mark.parametrize(
    "pdb_fixture, expected",
    [
        ("ubq_pdb", False),
        ("dna_pdb", True),
        ("rna_pdb", True),
        ("protein_dna_pdb", True),
    ],
)
def test_fast_relax_automatic_graph_mode(request, pdb_fixture, expected, torch_device):
    pose_stack = pose_stack_from_pdb(request.getfixturevalue(pdb_fixture), torch_device)

    assert _resolve_cuda_graph_mode(pose_stack, None) == (
        expected and torch_device.type == "cuda"
    )
    assert _resolve_cuda_graph_mode(pose_stack, True)
    assert not _resolve_cuda_graph_mode(pose_stack, False)


@pytest.mark.parametrize("n_poses", [1])
def test_fast_relax_ubq(default_database, ubq_pdb, dun_sampler, torch_device, n_poses):
    # if torch_device == torch.device("cpu"):
    #     return

    p = pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=76)

    pose_stack = PoseStackBuilder.from_poses([p] * n_poses, torch_device)
    sfxn = get_relax_sfxn(default_database, torch_device)

    mm = MoveMap.from_pose_stack(pose_stack)
    mm.move_all_jumps = True
    mm.move_all_named_torsions = True

    palette = PackerPalette()
    fold_forest = FoldForest.reasonable_fold_forest(pose_stack)

    def task_op(task):
        task.restrict_to_repacking()
        task.or_bump_check(True)

        fixed_sampler = FixedAAChiSampler()
        task.add_conformer_sampler(dun_sampler)
        task.add_conformer_sampler(fixed_sampler)
        task.add_conformer_sampler(IncludeCurrentSampler())

    start_time = time.perf_counter()

    # Now let's run fast_relax
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
    assert new_pose_stack
    assert isinstance(new_pose_stack, PoseStack)

    elapsed_time = stop_time - start_time

    print(f"1ubq {n_poses} FastRelax Execution time: {elapsed_time:.6f} seconds")


@pytest.mark.parametrize("n_poses", [1])
def test_cart_relax_ubq(default_database, ubq_pdb, dun_sampler, torch_device, n_poses):
    """Cartesian fast-relax on ubiquitin using CartesianSfxnNetwork.

    Exercise the default Cartesian minimizer with CUDA graph capture.
    """
    if torch_device == torch.device("cpu"):
        pytest.skip("CUDA only test")

    p = pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=76)

    pose_stack = PoseStackBuilder.from_poses([p] * n_poses, torch_device)
    sfxn = get_relax_sfxn(default_database, torch_device)

    # CartesianMoveMap with coord_mask=None moves all atoms.
    cart_mm = CartesianMoveMap()
    palette = PackerPalette()
    fold_forest = FoldForest.reasonable_fold_forest(pose_stack)

    def task_op(task):
        task.restrict_to_repacking()
        task.or_bump_check(True)

        fixed_sampler = FixedAAChiSampler()
        task.add_conformer_sampler(dun_sampler)
        task.add_conformer_sampler(fixed_sampler)
        task.add_conformer_sampler(IncludeCurrentSampler())

    start_time = time.perf_counter()

    verbose = True
    new_pose_stack = fast_relax(
        pose_stack,
        sfxn,
        palette,
        cart_mm,
        fold_forest,
        task_operations=[task_op],
        cuda_graph=True,
        num_repeats=1,
        verbose=verbose,
    )

    torch.cuda.synchronize()
    stop_time = time.perf_counter()

    assert new_pose_stack
    assert isinstance(new_pose_stack, PoseStack)

    elapsed_time = stop_time - start_time

    print(f"1ubq {n_poses} CartRelax Execution time: {elapsed_time:.6f} seconds")


def test_fast_relax_releases_minimizer_state_between_packing_stages(
    default_database, ubq_pdb, dun_sampler, torch_device, monkeypatch
):
    """Keep repeated pack/min results exact without retaining prior minimization."""
    pose = pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=8)
    palette = PackerPalette()
    move_map = CartesianMoveMap()
    fold_forest = FoldForest.reasonable_fold_forest(pose)
    schedule = [0.2, 1.0]
    optimizer_kwargs = {
        "max_iter": 1,
        "fixed_iterations": True,
        "gradtol": 0.0,
        "atol": 0.0,
        "rtol": 0.0,
    }

    def task_op(task):
        task.restrict_to_repacking()
        task.add_conformer_sampler(dun_sampler)
        task.add_conformer_sampler(FixedAAChiSampler())
        task.add_conformer_sampler(IncludeCurrentSampler())

    class RecordingMinimizer:
        def __init__(self, release_between_stages):
            self.minimizer = CartesianMinimizer(cuda_graph=torch_device.type == "cuda")
            self.release_between_stages = release_between_stages
            self.energies = []
            self.gradients = []
            self.released_allocations = []
            self.network_refs = []
            self.optimizer_refs = []

        def __call__(
            self,
            current,
            score_function,
            *,
            fold_forest,
            move_map,
            verbose,
        ):
            result = self.minimizer(
                current,
                score_function,
                optimizer_kwargs=optimizer_kwargs,
            )
            network = self.minimizer.network
            self.network_refs.append(weakref.ref(network))
            self.optimizer_refs.append(weakref.ref(self.minimizer.optimizer))
            network.zero_grad()
            energy = network()
            energy.sum().backward()
            self.energies.append(energy.detach().clone())
            self.gradients.append(network.masked_coords.grad.detach().clone())
            return result

        def release_retained_state(self):
            if not self.release_between_stages:
                return
            self.minimizer.release_retained_state()
            if torch_device.type == "cuda":
                torch.cuda.synchronize(torch_device)
                self.released_allocations.append(
                    torch.cuda.memory_allocated(torch_device)
                )

    original_pack_rotamers = fast_relax_module.pack_rotamers

    def run(release_between_stages):
        assignments = []
        pack_entry_allocations = []

        def record_pack(current, score_function, task, verbose=False):
            if torch_device.type == "cuda":
                torch.cuda.synchronize(torch_device)
                pack_entry_allocations.append(torch.cuda.memory_allocated(torch_device))
            packed = original_pack_rotamers(
                current, score_function, task, verbose=verbose
            )
            assignments.append(
                (packed.block_type_ind.detach().clone(), packed.coords.detach().clone())
            )
            return packed

        monkeypatch.setattr(fast_relax_module, "pack_rotamers", record_pack)
        score_function = get_relax_sfxn(default_database, torch_device)
        minimizer = RecordingMinimizer(release_between_stages)
        torch.manual_seed(742)
        result = fast_relax(
            pose,
            score_function,
            palette,
            move_map,
            fold_forest,
            task_operations=[task_op],
            num_repeats=1,
            schedule=schedule,
            min_fn=minimizer,
        )
        return result, assignments, minimizer, pack_entry_allocations

    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        retained_result, retained_assignments, retained, _ = run(False)
        retained.minimizer.release_retained_state()
        released_result, released_assignments, released, pack_allocations = run(True)
    finally:
        torch.set_num_threads(original_threads)

    torch.testing.assert_close(
        released_result.coords, retained_result.coords, rtol=0, atol=0
    )
    assert len(released_assignments) == len(retained_assignments) == len(schedule)
    for released_assignment, retained_assignment in zip(
        released_assignments, retained_assignments
    ):
        torch.testing.assert_close(
            released_assignment[0], retained_assignment[0], rtol=0, atol=0
        )
        torch.testing.assert_close(
            released_assignment[1], retained_assignment[1], rtol=0, atol=0
        )
    for released_energy, retained_energy in zip(released.energies, retained.energies):
        torch.testing.assert_close(
            released_energy, retained_energy, rtol=1e-6, atol=1e-6
        )
    for released_gradient, retained_gradient in zip(
        released.gradients, retained.gradients
    ):
        torch.testing.assert_close(
            released_gradient, retained_gradient, rtol=1e-5, atol=1e-5
        )

    assert released.minimizer.network is None
    assert released.minimizer.optimizer is None
    assert all(reference() is None for reference in released.network_refs)
    assert all(reference() is None for reference in released.optimizer_refs)
    if torch_device.type == "cuda":
        assert len(pack_allocations) == len(schedule)
        assert len(released.released_allocations) == len(schedule)
        assert (
            max(pack_allocations) - min(pack_allocations) < 2 * 1024**2
        ), pack_allocations
        assert (
            max(released.released_allocations) - min(released.released_allocations)
            < 2 * 1024**2
        ), released.released_allocations


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

    palette = PackerPalette()

    def task_op(task):
        task.restrict_to_repacking()
        task.or_bump_check(True)

        fixed_sampler = FixedAAChiSampler()
        task.add_conformer_sampler(dun_sampler)
        task.add_conformer_sampler(fixed_sampler)
        task.add_conformer_sampler(IncludeCurrentSampler())

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

    assert new_pose_stack
    assert isinstance(new_pose_stack, PoseStack)

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

    fold_forest = FoldForest.reasonable_fold_forest(pose_stack)
    mm = MoveMap.from_pose_stack(pose_stack)
    mm.move_all_jumps = True
    mm.move_all_named_torsions = True

    palette = PackerPalette()

    def task_op(task):
        task.restrict_to_repacking()
        task.or_bump_check(True)

        fixed_sampler = FixedAAChiSampler()
        task.add_conformer_sampler(dun_sampler)
        task.add_conformer_sampler(fixed_sampler)
        task.add_conformer_sampler(IncludeCurrentSampler())

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
    assert new_pose_stack
    assert isinstance(new_pose_stack, PoseStack)

    stop_time = time.perf_counter()

    elapsed_time = stop_time - start_time

    print(
        f"Three differently-shaped PDBs relaxed; Execution time: {elapsed_time:.6f} seconds"
    )
