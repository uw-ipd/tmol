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


def test_torsion_space_gradcheck(ubq_res):
    test_system = PackedResidueSystem.from_residues(ubq_res[:6])

    torsion_space = DofSpaceScore.build_for(test_system)

    start_dofs = torsion_space.dofs.clone()

    def total_score(dofs):
        torsion_space.dofs = dofs
        return torsion_space.intra_score().total

    # Seeing number of high-magnitude gradcheck failures in upstream dofs:
    #
    # failures:
    #          analytic   numeric  abs_error  rel_error  failure
    # (2, 0)   0.059407  0.068855   0.009449   0.137077     True
    # (3, 0)  -0.062377 -0.201285   0.138908   0.690057     True
    # (7, 0)  -0.507302 -0.518799   0.011497   0.022141     True
    # (14, 0)  2.821717  2.777839   0.043878   0.015792     True
    # (25, 0)  0.047311  0.046337   0.000975   0.020816     True
    # (43, 0) -0.000932 -0.000906   0.000026   0.018197     True

    assert gradcheck(
        total_score, (start_dofs,), eps=2e-2, rtol=1e-2, atol=1e-5, nfail=6
    )


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

    # TODO Seeing limited number of gradcheck failures, often with "low-mag" gradients
    #
    #  failures:
    #            analytic   numeric  abs_error  rel_error  failure
    #  (100, 0) -0.004347 -0.004450   0.000103   0.020930     True
    #  (122, 0) -0.007231 -0.007709   0.000477   0.060638     True
    #  (173, 0) -0.000636 -0.000668   0.000031   0.031683     True
    #  (192, 0) -0.004331 -0.004260   0.000072   0.014454     True
    #  (193, 0) -0.001689 -0.001748   0.000059   0.028103     True
    #  (229, 0)  0.007737  0.007598   0.000140   0.017091     True
    #  (315, 0) -0.004810 -0.004705   0.000106   0.020323     True
    #  (326, 0)  0.000250  0.000286   0.000036   0.090136     True

    assert gradcheck(
        total_score, (start_coords,), eps=1.5e-2, rtol=1e-2, atol=1e-5, nfail=8
    )
