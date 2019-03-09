from tmol.score.coordinates import CartesianAtomicCoordinateProvider
from tmol.score.omega import OmegaScoreGraph
from tmol.score.score_graph import score_graph
from tmol.system.score_support import rama_graph_inputs


@score_graph
class OmegaGraph(CartesianAtomicCoordinateProvider, OmegaScoreGraph):
    pass


def test_omega_smoke(ubq_system, torch_device):
    omega_graph = OmegaGraph.build_for(ubq_system, device=torch_device)
    assert omega_graph.allomegas.shape[0] == 76
