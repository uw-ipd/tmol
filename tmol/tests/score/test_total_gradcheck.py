import torch

from tmol.system.residue.packed import PackedResidueSystem

from tmol.system.residue.score import (
    system_real_space_graph_params,
    system_torsion_space_graph_params,
)

from tmol.score import TotalScoreGraph
from tmol.score.coordinates import (
    RealSpaceScoreGraph,
    DofSpaceScoreGraph,
)


class RealSpaceScore(
        TotalScoreGraph,
        RealSpaceScoreGraph,
):
    pass


class DofSpaceScore(
        TotalScoreGraph,
        DofSpaceScoreGraph,
):
    pass


def test_torsion_space_gradcheck(ubq_res):
    test_system = PackedResidueSystem.from_residues(ubq_res[:6])

    torsion_space = DofSpaceScore(
        **system_torsion_space_graph_params(test_system)
    )

    start_dofs = torsion_space.dofs.clone()

    def total_score(dofs):
        torsion_space.dofs = dofs
        return torsion_space.total_score

    assert torch.autograd.gradcheck(
        total_score,
        (start_dofs, ),
        eps=1e-3,
        rtol=5e-3,
        atol=5e-4,
    )


def test_real_space_gradcheck(ubq_res):
    test_system = PackedResidueSystem.from_residues(ubq_res[:6])
    real_space = RealSpaceScore(**system_real_space_graph_params(test_system))

    coord_mask = torch.isnan(real_space.coords).sum(dim=-1) == 0
    start_coords = real_space.coords[coord_mask]

    def total_score(coords):
        state_coords = real_space.coords.detach().clone()
        state_coords[coord_mask] = coords

        return real_space.total_score

    assert torch.autograd.gradcheck(total_score, (start_coords, ))
