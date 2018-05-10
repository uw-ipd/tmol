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


def test_torsion_space_by_real_space_total_score(ubq_system):

    real_space = RealSpaceScore(**system_real_space_graph_params(ubq_system))

    torsion_space = DofSpaceScore(
        **system_torsion_space_graph_params(ubq_system)
    )

    real_total = real_space.step()
    torsion_total = torsion_space.step()

    assert (real_total == torsion_total).all()


def test_torsion_space_coord_smoke(ubq_system):
    tsys = ubq_system

    torsion_space = DofSpaceScore(**system_torsion_space_graph_params(tsys))

    start_dofs = torch.tensor(torsion_space.dofs, requires_grad=True)
    start_coords = torch.tensor(torsion_space.coords, requires_grad=False)
    cmask = torch.isnan(start_coords).sum(dim=-1) == 0

    def coord_residuals(dofs):
        torsion_space.dofs = dofs
        return (torsion_space.coords[cmask] -
                start_coords[cmask]).norm(dim=-1).sum()

    torch.random.manual_seed(1663)
    pdofs = torch.tensor((torch.rand_like(start_dofs) - .5) * 1e-2,
                         requires_grad=True)

    assert pdofs.requires_grad

    res = coord_residuals(pdofs)
    assert res.requires_grad

    res.backward(retain_graph=True)
    assert pdofs.grad is not None


def test_torsion_space_to_coordinate_gradcheck(ubq_res):
    tsys = PackedResidueSystem.from_residues(ubq_res[:6])

    torsion_space = DofSpaceScore(**system_torsion_space_graph_params(tsys))

    start_dofs = torsion_space.dofs.detach().clone().requires_grad_()
    start_coords = torsion_space.coords.detach().clone()

    cmask = torch.isnan(start_coords).sum(dim=-1) == 0

    def coord_residuals(dofs):
        torsion_space.dofs = dofs
        res = (torsion_space.coords[cmask] - (start_coords[cmask])).sum(dim=-1)

        return res

    assert torch.autograd.gradcheck(
        coord_residuals, (start_dofs, ), eps=5e-3, atol=5e-4, rtol=5e-3
    )
