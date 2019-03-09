from tmol.score.coordinates import CartesianAtomicCoordinateProvider
from tmol.score.rama import RamaScoreGraph
from tmol.score.score_graph import score_graph
from tmol.system.score_support import rama_graph_inputs


@score_graph
class RamaGraph(CartesianAtomicCoordinateProvider, RamaScoreGraph):
    pass


def test_phipsi_identification(default_database, ubq_system):
    tsys = ubq_system
    test_params = rama_graph_inputs(tsys, default_database)
    assert test_params["allphis"].shape[0] == 76
    assert test_params["allpsis"].shape[0] == 76


def test_rama_smoke(ubq_system, torch_device):
    rama_graph = RamaGraph.build_for(ubq_system, device=torch_device)
    assert rama_graph.allphis.shape[0] == 76
    assert rama_graph.allpsis.shape[0] == 76
