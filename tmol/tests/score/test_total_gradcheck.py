import torch

from tmol.system.packed import PackedResidueSystem

from tmol.score.total_score_graphs import TotalScoreGraph
from tmol.score.score_graph import score_graph
from tmol.score.coordinates import (
    CartesianAtomicCoordinateProvider,
    KinematicAtomicCoordinateProvider,
)

from tmol.tests.autograd import gradcheck


@score_graph
class RealSpaceScore(CartesianAtomicCoordinateProvider, TotalScoreGraph):
    pass


@score_graph
class DofSpaceScore(KinematicAtomicCoordinateProvider, TotalScoreGraph):
    pass


def test_torsion_space_gradcheck(ubq_res):
    test_system = PackedResidueSystem.from_residues(ubq_res[:6])

    torsion_space = DofSpaceScore.build_for(test_system)

    start_dofs = torsion_space.dofs.requires_grad_()

    def total_score(minimizable_dofs):
        torsion_space.dofs[:, :6] = minimizable_dofs
        return torsion_space.intra_score().total

    # fd this test needs work...
    assert gradcheck(total_score, (start_dofs[:, :6],), eps=1e-2, atol=5e-2, nfail=0)


def test_real_space_gradcheck(ubq_res):
    test_system = PackedResidueSystem.from_residues(ubq_res[:6])
    real_space = RealSpaceScore.build_for(test_system)

    coord_mask = torch.isnan(real_space.coords).sum(dim=-1) == 0
    start_coords = real_space.coords[coord_mask]

    def total_score(coords):
        state_coords = real_space.coords.detach().clone()
        state_coords[coord_mask] = coords

        real_space.coords = state_coords
        return real_space.intra_score().total

    # fd this test needs work...
    assert gradcheck(total_score, (start_coords,), eps=1e-2, atol=5e-2, nfail=0)
