from tmol.system.score import (
    system_cartesian_space_graph_params,
    system_torsion_space_graph_params,
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
        **system_torsion_space_graph_params(ubq_system, device=torch_device)
    ).total_score

    assert total_score.device == torch_device


def test_real_space_smoke(ubq_system, torch_device):
    total_score = RealSpaceScore(
        **system_cartesian_space_graph_params(ubq_system, device=torch_device)
    ).total_score

    assert total_score.device == torch_device
