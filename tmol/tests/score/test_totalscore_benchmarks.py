import pytest
import torch

from tmol.system.score_support import get_full_score_system_for
from tmol.score.modules.coords import coords_for

from tmol.system.packed import PackedResidueSystemStack


@pytest.mark.benchmark(group="total_score_setup")
@pytest.mark.parametrize("system_size", [40, 75, 150, 300, 600])
def test_setup(benchmark, systems_bysize, system_size, torch_device):
    @benchmark
    def setup():
        score_system = get_full_score_system_for(
            systems_bysize[system_size], device=torch_device
        )
        coords = coords_for(systems_bysize[system_size], score_system)
        return score_system.intra_total(coords)

    score = setup
    assert score == score


@pytest.mark.benchmark(group="total_score_onepass")
@pytest.mark.parametrize("system_size", [40, 75, 150, 300, 600])
def test_full(benchmark, systems_bysize, system_size, torch_device):
    score_system = get_full_score_system_for(
        systems_bysize[system_size], device=torch_device
    )
    coords = coords_for(systems_bysize[system_size], score_system)

    @benchmark
    def forward_backward():
        total = score_system.intra_total(coords)
        total.backward()
        return total

    forward_backward


@pytest.mark.benchmark(group="stacked_totalscore_onepass")
@pytest.mark.parametrize("nstacks", [1, 3, 10, 30])
def test_stacked_full(benchmark, ubq_system, nstacks, torch_device):
    stack = PackedResidueSystemStack((ubq_system,) * nstacks)
    score_system = get_full_score_system_for(stack, device=torch_device)
    coords = coords_for(stack, score_system)

    @benchmark
    def forward_backward():
        total = score_system.intra_total(coords)
        tsum = torch.sum(total)
        tsum.backward(retain_graph=True)
        return total

    forward_backward
