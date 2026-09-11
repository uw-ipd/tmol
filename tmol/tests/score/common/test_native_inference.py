"""Direct native term inference preserves values without discarded derivatives."""

import pytest
import torch

from tmol import pose_stack_from_pdb
from tmol.pose import PoseStackBuilder
from tmol.score.disulfide import DisulfideEnergyTerm
from tmol.score.elec import ElecEnergyTerm
from tmol.score.genbonded import GenBondedEnergyTerm
from tmol.score.hbond import HBondEnergyTerm

TERMS = [
    DisulfideEnergyTerm,
    ElecEnergyTerm,
    GenBondedEnergyTerm,
    HBondEnergyTerm,
]


def make_scorer(term_class, ubq_pdb, disulfide_pdb, database, device):
    pose = PoseStackBuilder.from_poses(
        [
            pose_stack_from_pdb(disulfide_pdb, device),
            pose_stack_from_pdb(ubq_pdb, device, residue_end=15),
        ],
        device,
    )
    term = term_class(param_db=database, device=device)
    for block_type in pose.packed_block_types.active_block_types:
        term.setup_block_type(block_type)
    term.setup_packed_block_types(pose.packed_block_types)
    term.setup_poses(pose)
    return pose, term.render_whole_pose_scoring_module(pose)


@pytest.mark.parametrize("term_class", TERMS, ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("inference", [False, True])
def test_direct_inference_preserves_values_and_later_gradients(
    term_class, dtype, inference, ubq_pdb, disulfide_pdb, default_database, torch_device
):
    pose, scorer = make_scorer(
        term_class, ubq_pdb, disulfide_pdb, default_database, torch_device
    )
    coords = pose.coords.to(dtype).detach().requires_grad_(True)
    expected = scorer(coords)
    (expected_grad,) = torch.autograd.grad(expected.sum(), coords)
    with torch.inference_mode() if inference else torch.no_grad():
        actual = scorer(coords)
    assert coords.requires_grad
    assert not actual.requires_grad
    tolerance = {"rtol": 0, "atol": 0} if torch_device.type == "cpu" else {}
    torch.testing.assert_close(actual, expected, **tolerance)
    after = scorer(coords)
    (after_grad,) = torch.autograd.grad(after.sum(), coords)
    torch.testing.assert_close(after, expected, **tolerance)
    torch.testing.assert_close(after_grad, expected_grad, **tolerance)


@pytest.mark.parametrize("term_class", TERMS, ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("inference", [False, True])
def test_tracked_inference_allocates_no_derivative_scratch(
    term_class, inference, ubq_pdb, disulfide_pdb, default_database, torch_device
):
    if torch_device.type != "cuda":
        pytest.skip("CUDA allocator statistics required")
    pose, scorer = make_scorer(
        term_class, ubq_pdb, disulfide_pdb, default_database, torch_device
    )
    tracked = pose.coords.detach().requires_grad_(True)
    detached = tracked.detach()

    def allocation(coords):
        torch.cuda.synchronize(torch_device)
        before = torch.cuda.memory_allocated(torch_device)
        torch.cuda.reset_peak_memory_stats(torch_device)
        result = scorer(coords)
        torch.cuda.synchronize(torch_device)
        return result, torch.cuda.max_memory_allocated(torch_device) - before

    with torch.inference_mode() if inference else torch.no_grad():
        scorer(detached)
        scorer(tracked)
        expected, detached_peak = allocation(detached)
        actual, tracked_peak = allocation(tracked)
    torch.testing.assert_close(actual, expected)
    assert tracked_peak <= detached_peak
    assert tracked.requires_grad
