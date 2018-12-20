import torch

from tmol.system.packed import PackedResidueSystem

from tmol.score import TotalScoreGraph
from tmol.score.coordinates import (
    CartesianAtomicCoordinateProvider,
    KinematicAtomicCoordinateProvider,
)

from tmol.utility.reactive import reactive_attrs

from tmol.tests.autograd import gradcheck


@reactive_attrs
class RealSpaceScore(CartesianAtomicCoordinateProvider, TotalScoreGraph):
    pass


@reactive_attrs
class DofSpaceScore(KinematicAtomicCoordinateProvider, TotalScoreGraph):
    pass


def test_torsion_space_by_real_space_total_score(ubq_system):

    real_space = RealSpaceScore.build_for(ubq_system)
    torsion_space = DofSpaceScore.build_for(ubq_system)

    real_total = real_space.intra_score().total
    torsion_total = torsion_space.intra_score().total

    assert (real_total == torsion_total).all()


def test_torsion_space_coord_smoke(ubq_system):
    torsion_space = DofSpaceScore.build_for(ubq_system)

    start_dofs = torch.tensor(torsion_space.dofs, requires_grad=True)
    start_coords = torch.tensor(torsion_space.coords, requires_grad=False)
    cmask = torch.isnan(start_coords).sum(dim=-1) == 0

    def coord_residuals(dofs):
        torsion_space.dofs = dofs
        return (torsion_space.coords[cmask] - start_coords[cmask]).norm(dim=-1).sum()

    torch.random.manual_seed(1663)
    pdofs = torch.tensor((torch.rand_like(start_dofs) - .5) * 1e-2, requires_grad=True)

    assert pdofs.requires_grad

    res = coord_residuals(pdofs)
    assert res.requires_grad

    res.backward(retain_graph=True)
    assert pdofs.grad is not None


def test_torsion_space_to_cart_space_gradcheck(ubq_res):
    tsys = PackedResidueSystem.from_residues(ubq_res[:6])

    torsion_space = DofSpaceScore.build_for(tsys)

    start_dofs = torsion_space.dofs.detach().clone().requires_grad_()
    start_coords = torsion_space.coords.detach().clone()

    cmask = torch.isnan(start_coords).sum(dim=-1) == 0

    def coords(dofs):
        torsion_space.dofs = dofs
        return torsion_space.coords[cmask]

    assert gradcheck(coords, (start_dofs,), eps=1e-1, atol=1e-6, rtol=2e-3)
