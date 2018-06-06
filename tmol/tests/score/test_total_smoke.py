from tmol.system.score import (
    extract_graph_parameters,
)

from tmol.score import TotalScoreGraph
from tmol.score.coordinates import (
    CartesianAtomicCoordinateProvider,
    KinematicAtomicCoordinateProvider,
)

from tmol.utility.reactive import reactive_attrs


@reactive_attrs
class RealSpaceScore(
        CartesianAtomicCoordinateProvider,
        TotalScoreGraph,
):
    pass


@reactive_attrs
class DofSpaceScore(
        KinematicAtomicCoordinateProvider,
        TotalScoreGraph,
):
    pass


def test_torsion_space_smoke(ubq_system, torch_device):
    total_score = DofSpaceScore(
        **extract_graph_parameters(
            DofSpaceScore, ubq_system, device=torch_device
        )
    ).total_score

    assert total_score.device == torch_device


def test_real_space_smoke(ubq_system, torch_device):
    total_score = RealSpaceScore(
        **extract_graph_parameters(
            RealSpaceScore, ubq_system, device=torch_device
        )
    ).total_score

    assert total_score.device == torch_device
