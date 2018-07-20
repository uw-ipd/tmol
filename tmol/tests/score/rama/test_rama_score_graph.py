import numpy
import torch

from tmol.utility.reactive import reactive_attrs
from tmol.system.packed import PackedResidueSystem
from tmol.score import (
    CartesianAtomicCoordinateProvider,
    KinematicAtomicCoordinateProvider,
)
from tmol.score.total_score import TotalScoreComponentsGraph
from tmol.score.rama.rama_score_graph import RamaScoreGraph
from tmol.score.bonded_atom import BondedAtomScoreGraph


@reactive_attrs(auto_attribs=True)
class TRama(CartesianAtomicCoordinateProvider, RamaScoreGraph,
            TotalScoreComponentsGraph):
    """Cart total."""
    pass


def test_create_torsion_provider(ubq_system):
    trama = TRama.build_for(ubq_system)
    assert trama

    gold_rama_scores = numpy.array([
        -0.7474, -0.3156, -0.7905, -0.3412, -0.3135, -0.2807, 0.3014, 0.1461,
        -1.8280, -0.4346, -0.0333, -0.2466, 0.2117, -0.0184, 0.2571, 0.9067,
        5.7752, 0.6475, -0.8588, -0.7846, -0.7969, 0.3320, -0.4117, 1.4239,
        1.1622, -0.6051, -0.5258, -0.7472, 0.0234, 1.5939, 1.4089, -0.4842,
        2.2174, -1.8470, 3.4690, 5.0221, 0.2429, -0.0765, -0.9289, -0.5931,
        0.3425, -0.3653, -0.8009, 1.2904, 1.6205, 0.2987, -0.4256, -0.6167,
        -1.0429, -0.3423, 2.3625, 0.1690, 0.5093, -0.5014, 0.0158, -0.6732,
        0.8884, -0.0899, -0.7584, -0.7619, 0.7566, 0.5671, 2.0990, -1.1896,
        0.0419, -0.6342, -0.4525, 0.0383, -0.4488, -0.6283, 0.9796, -0.9366,
        1.5038, 2.6801
    ])

    numpy.testing.assert_allclose(
        trama.rama_scores.detach().numpy(), gold_rama_scores, atol=1e-4
    )

    gold_total_score = 16.6269
    assert abs(trama.total_rama - gold_total_score) < 1e-4

    #R3 scores
    R3_scores = [
        -0.265762, -0.276527, -0.282922, -0.163675, -0.148548, 0.00516669,
        0.11188, -0.420469, -0.56565, -0.116964, -0.0699555, -0.00870497,
        0.0483434, 0.0596801, 0.290944, 1.67046, 1.60567, -0.0528071,
        -0.410844, -0.395371, -0.116212, -0.0199245, 0.253053, 0.646534,
        0.139288, -0.28272, -0.318255, -0.18094, 0.404326, 0.750679, 0.231162,
        0.433303, 0.092603, 0.405485, 2.12276, 1.31625, 0.0415896, -0.251359,
        -0.380509, -0.0626546, -0.00569591, -0.291543, 0.122381, 0.727718,
        0.479803, -0.0317097, -0.260576, -0.414911, -0.346305, 0.505052,
        0.632885, 0.169583, 0.00196636, -0.121407, -0.164351, 0.0537982,
        0.199628, -0.212075, -0.380084, -0.00132824, 0.330922, 0.666519,
        0.227355, -0.286907, -0.148068, -0.271684, -0.103557, -0.102637,
        -0.269276, 0.0878474, 0.0107583, 0.141788, 1.04597
    ]

    # weight of 0.5; then assign 0.5 to residue i and 0.5 to residue i+1
    two_body_version = 0.25 * (trama.rama_scores[1:] + trama.rama_scores[:-1])
    numpy.testing.assert_allclose(
        two_body_version.detach().numpy(), numpy.array(R3_scores), atol=1e-4
    )


@reactive_attrs
class DofSpaceRamaScore(
        KinematicAtomicCoordinateProvider,
        BondedAtomScoreGraph,
        RamaScoreGraph,
        TotalScoreComponentsGraph,
):
    pass


def test_torsion_space_rama_gradcheck(ubq_res):
    test_system = PackedResidueSystem.from_residues(ubq_res[:6])

    torsion_space = DofSpaceRamaScore.build_for(test_system)
    print("phi_inds", torsion_space.phi_inds)
    print("res_aas", torsion_space.res_aas)
    print("upper", torsion_space.upper)

    start_dofs = torsion_space.dofs.clone()

    def total_score(dofs):
        torsion_space.dofs = dofs
        return torsion_space.total_score

    assert torch.autograd.gradcheck(
        total_score,
        (start_dofs, ),
        eps=1e-4,
        rtol=5e-3,
        atol=5e-4,
    )


@reactive_attrs
class CartSpaceRamaScore(
        CartesianAtomicCoordinateProvider,
        BondedAtomScoreGraph,
        RamaScoreGraph,
        TotalScoreComponentsGraph,
):
    pass


def test_cartesian_space_rama_gradcheck(ubq_res):
    test_system = PackedResidueSystem.from_residues(ubq_res[:6])

    real_space = CartSpaceRamaScore.build_for(test_system)
    print("phi_inds", real_space.phi_inds)
    print("res_aas", real_space.res_aas)
    print("upper", real_space.upper)

    coord_mask = torch.isnan(real_space.coords).sum(dim=-1) == 0
    start_coords = real_space.coords[coord_mask]

    def total_score(coords):
        state_coords = real_space.coords.detach().clone()
        state_coords[coord_mask] = coords

        return real_space.total_score

    assert torch.autograd.gradcheck(total_score, (start_coords, ))
