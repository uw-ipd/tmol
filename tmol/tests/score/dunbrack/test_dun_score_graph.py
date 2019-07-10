import numpy
import torch

from tmol.system.packed import PackedResidueSystem
from tmol.score.coordinates import (
    CartesianAtomicCoordinateProvider,
    KinematicAtomicCoordinateProvider,
)
from tmol.score.device import TorchDevice
from tmol.score.dunbrack.score_graph import DunbrackScoreGraph
from tmol.score.score_graph import score_graph
from tmol.utility.cuda.synchronize import synchronize_if_cuda_available


@score_graph
class CartDunbrackGraph(
    CartesianAtomicCoordinateProvider, DunbrackScoreGraph, TorchDevice
):
    pass


@score_graph
class KinematicDunbrackGraph(
    KinematicAtomicCoordinateProvider, DunbrackScoreGraph, TorchDevice
):
    pass


def test_dunbrack_score_graph_smoke(ubq_system, default_database, torch_device):
    CartDunbrackGraph.build_for(
        ubq_system, device=torch_device, parameter_database=default_database
    )


def test_dunbrack_score_setup(ubq_system, default_database, torch_device):
    dunbrack_graph = CartDunbrackGraph.build_for(
        ubq_system, device=torch_device, parameter_database=default_database
    )

    dun_params = dunbrack_graph.dun_resolve_indices

    ndihe_gold = numpy.array(
        [
            5,
            5,
            4,
            4,
            3,
            6,
            3,
            4,
            3,
            6,
            3,
            4,
            3,
            4,
            5,
            3,
            5,
            5,
            3,
            4,
            3,
            4,
            5,
            4,
            3,
            6,
            6,
            4,
            5,
            4,
            6,
            5,
            4,
            5,
            5,
            4,
            5,
            5,
            6,
            4,
            4,
            4,
            6,
            5,
            4,
            5,
            4,
            6,
            3,
            4,
            3,
            4,
            4,
            4,
            4,
            5,
            6,
            5,
            3,
            3,
            4,
            4,
            4,
            3,
            4,
            6,
            4,
            6,
        ],
        dtype=int,
    )
    numpy.testing.assert_array_equal(ndihe_gold, dun_params.ndihe_for_res.cpu().numpy())


def test_dunbrack_score(ubq_system, torch_device, default_database):
    dunbrack_graph = CartDunbrackGraph.build_for(
        ubq_system, device=torch_device, parameter_database=default_database
    )
    intra_graph = dunbrack_graph.intra_score()
    e_dun_tot = intra_graph.dun_score
    synchronize_if_cuda_available()
    e_dun_gold = torch.Tensor([70.6497, 240.3100, 99.6609])
    torch.testing.assert_allclose(e_dun_gold, e_dun_tot.cpu())


def test_cartesian_space_dun_gradcheck(ubq_res, torch_device):
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

    a = analytical
    n = numerical

    print(a.t())
    print(n.t())
    assert False


# Only run the CPU version of this test, since on the GPU
#     f1s = torch.cross(Xs, Xs - dsc_dx)
# creates non-zero f1s even when dsc_dx is zero everywhere
def test_kinematic_space_dun_gradcheck(ubq_res):
    test_system = PackedResidueSystem.from_residues(ubq_res[:6])
    torsion_space = KinematicDunbrackGraph.build_for(test_system)

    start_dofs = torsion_space.dofs.clone()

    def total_score(dofs):
        torsion_space.dofs = dofs
        return torsion_space.intra_score().total

    # x = total_score(start_dofs)

    assert torch.autograd.gradcheck(total_score, (start_dofs,), eps=2e-3, atol=5e-2)
