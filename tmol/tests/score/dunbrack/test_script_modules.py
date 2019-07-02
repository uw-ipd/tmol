import numpy
import torch

from tmol.system.packed import PackedResidueSystem
from tmol.score.coordinates import CartesianAtomicCoordinateProvider
from tmol.score.device import TorchDevice
from tmol.score.dunbrack.score_graph import DunbrackScoreGraph
from tmol.score.dunbrack.script_modules import DunbrackScoreModule
from tmol.score.score_graph import score_graph


@score_graph
class CartDunbrackGraph(
    CartesianAtomicCoordinateProvider, DunbrackScoreGraph, TorchDevice
):
    pass


def test_dunbrack_score(ubq_system, torch_device, default_database):
    dunbrack_graph = CartDunbrackGraph.build_for(
        ubq_system, device=torch_device, parameter_database=default_database
    )
    op = DunbrackScoreModule(
        dunbrack_graph.dun_param_resolver, dunbrack_graph.dun_resolve_indices
    )

    V = op(dunbrack_graph.coords[0, :])

    dunE_gold = torch.tensor([70.6497, 240.3101, 99.6609])

    torch.testing.assert_allclose(dunE_gold, V.cpu())


def test_dunbrack_scoregraph(ubq_system, torch_device, default_database):
    dunbrack_graph = CartDunbrackGraph.build_for(
        ubq_system, device=torch_device, parameter_database=default_database
    )

    V = dunbrack_graph.intra_score().total

    dunE_gold = torch.tensor([410.6207])

    torch.testing.assert_allclose(dunE_gold, V.cpu())


def test_dunbrack_gradcheck(ubq_res, torch_device):
    test_system = PackedResidueSystem.from_residues(ubq_res[:6])
    real_space = CartDunbrackGraph.build_for(test_system, device=torch_device)

    coord_mask = torch.isnan(real_space.coords).sum(dim=-1) == 0
    start_coords = real_space.coords[coord_mask]

    def total_score(coords):
        state_coords = real_space.coords.detach().clone()
        state_coords[coord_mask] = coords
        real_space.coords = state_coords
        return real_space.intra_score().total

    result = torch.autograd.gradcheck(
        total_score, (start_coords,), eps=2e-3, atol=5e-2, raise_exception=False
    )

    if result:
        return

    result = total_score(start_coords)

    # Extract results from torch/autograd/gradcheck.py
    from torch.autograd.gradcheck import get_numerical_jacobian, get_analytical_jacobian

    (analytical,), reentrant, correct_grad_sizes = get_analytical_jacobian(
        (start_coords,), result
    )
    numerical = get_numerical_jacobian(total_score, start_coords, start_coords, 2e-3)

    a = analytical.reshape(-1, 3)
    n = numerical.reshape(-1, 3)

    print(a)
    print(n)
    print(torch.abs(a - n))
    assert False
