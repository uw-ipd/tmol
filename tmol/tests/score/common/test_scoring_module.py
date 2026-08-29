"""Tests for shared rendered-scoring module behavior."""

import torch

from tmol.score.common import TermScoringModule


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
