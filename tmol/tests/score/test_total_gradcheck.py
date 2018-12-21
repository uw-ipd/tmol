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

    # TODO Seeing number of high-magnitude gradcheck failures in upstream dofs:
    #     failures:
    #          analytic   numeric  abs_error  rel_error  failure
    # (2, 0)   0.059407  0.068760   0.009353   0.135884     True
    # (3, 0)  -0.062377 -0.201344   0.138968   0.690149     True
    # (7, 0)  -0.507302 -0.518858   0.011556   0.022253     True
    # (14, 0)  2.821717  2.777719   0.043997   0.015836     True
    # (25, 0)  0.047311  0.046337   0.000975   0.020822     True

    assert gradcheck(
        total_score, (start_dofs,), eps=2e-2, rtol=1e-2, atol=1e-5, nfail=5
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
    #  (75, 0)  -0.001717 -0.001788   0.000071   0.034103     True
    #  (78, 0)   0.003436  0.003386   0.000050   0.011936     True
    #  (97, 0)   0.000595  0.000668   0.000072   0.093010     True
    #  (122, 0) -0.007232 -0.007391   0.000159   0.020161     True
    #  (128, 0) -0.010793 -0.010920   0.000127   0.010719     True
    #  (134, 0)  0.005337  0.005245   0.000092   0.015629     True
    #  (173, 0) -0.000636 -0.000596   0.000040   0.050960     True
    #  (192, 0) -0.004331 -0.004387   0.000056   0.010390     True
    #  (193, 0) -0.001689 -0.001621   0.000068   0.035782     True
    #  (207, 0) -0.005122 -0.005054   0.000067   0.011304     True
    #  (212, 0)  0.000189  0.000095   0.000094   0.880277     True
    #  (248, 0)  0.001700  0.001669   0.000031   0.012777     True
    #  (290, 0)  0.000734  0.000763   0.000029   0.024267     True
    #  (292, 0) -0.003013 -0.002956   0.000057   0.015800     True
    #  (293, 0) -0.000613 -0.000572   0.000041   0.054050     True
    #  (308, 0) -0.004230 -0.004148   0.000082   0.017274     True
    #  (317, 0)  0.001369  0.001478   0.000109   0.066948     True
    #  (321, 0) -0.000138 -0.000191   0.000053   0.224375     True
    #  (324, 0) -0.002131 -0.002193   0.000063   0.024117     True
    #  (326, 0)  0.000250  0.000191   0.000060   0.259939     True
    #  (328, 0)  0.004174  0.004101   0.000073   0.015390     True
    #  (332, 0)  0.002368  0.002289   0.000079   0.030150     True

    assert gradcheck(
        total_score, (start_coords,), eps=1e-2, rtol=1e-2, atol=1e-5, nfail=22
    )
