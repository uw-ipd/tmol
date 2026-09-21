"""Tests for shared rendered-scoring module behavior."""

import torch

from tmol import pose_stack_from_pdb
from tmol.score.common import TermScoringModule, TermWholePoseScoringModule


def test_float64_static_parameters_are_built_lazily() -> None:
    module = TermScoringModule(
        "test",
        [torch.tensor([1.0], dtype=torch.float32)],
        lambda *args: args,
    )
    module.common_parameters = []
    module._build_static_tails(False)

    assert module._static_tail_f64 is None

    tail_f32 = module._static_tail_for_coords(torch.zeros(1))
    assert tail_f32[0].dtype == torch.float32
    assert module._static_tail_f64 is None

    tail_f64 = module._static_tail_for_coords(torch.zeros(1, dtype=torch.float64))
    assert tail_f64[0].dtype == torch.float64
    assert (
        module._static_tail_for_coords(torch.zeros(1, dtype=torch.float64)) is tail_f64
    )


def test_custom_scorer_can_enable_grad_in_no_grad(ubq_pdb, torch_device):
    pose = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=2)

    def custom_score(coords, *args):
        with torch.enable_grad():
            score = coords.square().sum()
            (derivative,) = torch.autograd.grad(score, coords)
        return derivative.sum(), None

    module = TermWholePoseScoringModule("custom", pose, [], custom_score)
    coords = pose.coords.reshape(-1, 3).detach().requires_grad_(True)
    with torch.no_grad():
        actual = module(coords)
    torch.testing.assert_close(actual, 2 * coords.detach().sum())
    assert coords.requires_grad
