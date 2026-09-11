import gc

import pytest
import torch

from tmol import pose_stack_from_pdb
from tmol.pose import PoseStackBuilder
from tmol.score.dunbrack import DunbrackEnergyTerm

from tmol.tests.score.common import EnergyTermTestBase


def test_smoke(default_database, torch_device: torch.device):
    dunbrack_energy = DunbrackEnergyTerm(param_db=default_database, device=torch_device)

    assert dunbrack_energy.device == torch_device


def test_annotate_block_types(
    fresh_default_packed_block_types, default_database, torch_device: torch.device
):
    dunbrack_energy = DunbrackEnergyTerm(param_db=default_database, device=torch_device)

    pbt = fresh_default_packed_block_types
    bt_list = pbt.active_block_types

    for bt in bt_list:
        dunbrack_energy.setup_block_type(bt)
    dunbrack_energy.setup_packed_block_types(pbt)

    assert hasattr(pbt, "dunbrack_packed_block_data")

    first_tensor = pbt.dunbrack_packed_block_data[0]

    assert first_tensor.device == torch_device
    dunbrack_energy.setup_packed_block_types(pbt)
    assert first_tensor is pbt.dunbrack_packed_block_data[0]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("inference", [False, True])
def test_score_only_matches_derivative_scoring(
    ubq_pdb, default_database, torch_device, dtype, inference
):
    poses = [
        pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=n) for n in (10, 15)
    ]
    pose = PoseStackBuilder.from_poses(poses, torch_device)
    scorer = TestDunbrackEnergyTerm.get_whole_pose_scorer(
        pose, default_database, torch_device
    )
    coords = pose.coords.to(dtype).detach().requires_grad_(True)
    expected = scorer(coords)
    (expected_grad,) = torch.autograd.grad(expected.sum(), coords)

    # Score-only calls must tolerate absent derivative scratch, including
    # padded blocks and residues whose backbone dihedrals are unresolved.
    with torch.inference_mode() if inference else torch.no_grad():
        actual = scorer(coords)
    assert not actual.requires_grad
    assert coords.requires_grad
    torch.testing.assert_close(actual, expected)

    # Switching modes must leave the derivative-enabled path usable.
    after = scorer(coords)
    (after_grad,) = torch.autograd.grad(after.sum(), coords)
    torch.testing.assert_close(after, expected)
    torch.testing.assert_close(after_grad, expected_grad)


@pytest.mark.parametrize("inference", [False, True])
def test_score_only_tracked_input_uses_no_derivative_scratch(
    ubq_pdb, default_database, torch_device, inference
):
    if torch_device.type != "cuda":
        pytest.skip("CUDA allocator statistics required")
    pose = pose_stack_from_pdb(ubq_pdb, torch_device)
    scorer = TestDunbrackEnergyTerm.get_whole_pose_scorer(
        pose, default_database, torch_device
    )
    tracked = pose.coords.detach().requires_grad_(True)
    detached = tracked.detach()

    def peak_allocation(coords):
        torch.cuda.synchronize(torch_device)
        before = torch.cuda.memory_allocated(torch_device)
        torch.cuda.reset_peak_memory_stats(torch_device)
        result = scorer(coords)
        torch.cuda.synchronize(torch_device)
        peak = torch.cuda.max_memory_allocated(torch_device) - before
        return result, peak

    # Isolate these allocations from cyclic garbage left by earlier tests.
    gc.collect()
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        with torch.inference_mode() if inference else torch.no_grad():
            # Warm both paths before measuring only the scoring allocations.
            scorer(detached)
            scorer(tracked)
            expected, detached_peak = peak_allocation(detached)
            actual, tracked_peak = peak_allocation(tracked)
    finally:
        if gc_enabled:
            gc.enable()
    torch.testing.assert_close(actual, expected)
    assert tracked.requires_grad
    assert not actual.requires_grad
    assert tracked_peak <= detached_peak


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_native_derivative_output_preserved_in_no_grad(
    ubq_pdb, default_database, torch_device, dtype
):
    pose = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=10)
    scorer = TestDunbrackEnergyTerm.get_whole_pose_scorer(
        pose, default_database, torch_device
    )
    coords = pose.coords.to(dtype).detach().requires_grad_(True)
    flat = coords.flatten(start_dim=0, end_dim=-2)
    tail = scorer._static_tail_for_coords(coords)
    expected = scorer.term_score_poses(flat, *tail)
    with torch.no_grad():
        actual = scorer.term_score_poses(flat, *tail)
    assert torch.count_nonzero(expected[1]) > 0
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor)


class TestDunbrackEnergyTerm(EnergyTermTestBase):
    energy_term_class = DunbrackEnergyTerm

    @classmethod
    def test_whole_pose_scoring_10(cls, ubq_pdb, default_database, torch_device):
        return super().test_whole_pose_scoring_10(
            ubq_pdb, default_database, torch_device, update_baseline=False
        )

    @classmethod
    def test_whole_pose_scoring_jagged(
        cls,
        ubq_pdb,
        default_database,
        torch_device: torch.device,
    ):
        return super().test_whole_pose_scoring_jagged(
            ubq_pdb, default_database, torch_device, update_baseline=False
        )

    @classmethod
    def test_whole_pose_scoring_gradcheck(cls, ubq_pdb, default_database, torch_device):
        resnums = [(0, 4)]
        return super().test_whole_pose_scoring_gradcheck(
            ubq_pdb, default_database, torch_device, resnums=resnums
        )

    @classmethod
    def test_block_scoring_matches_whole_pose_scoring(
        cls, ubq_pdb, default_database, torch_device
    ):
        return super().test_block_scoring_matches_whole_pose_scoring(
            ubq_pdb, default_database, torch_device
        )

    @classmethod
    def test_block_scoring(cls, ubq_pdb, default_database, torch_device):
        resnums = [(0, 4)]
        return super().test_block_scoring(
            ubq_pdb,
            default_database,
            torch_device,
            resnums=resnums,
            update_baseline=False,
        )

    @classmethod
    def test_block_scoring_reweighted_gradcheck(
        cls, ubq_pdb, default_database, torch_device
    ):
        resnums = [(0, 4)]
        return super().test_block_scoring_reweighted_gradcheck(
            ubq_pdb,
            default_database,
            torch_device,
            resnums=resnums,
        )
