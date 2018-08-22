import pytest

from tmol.viewer import SystemViewer

from tmol.score.bonded_atom import BondedAtomScoreGraph
from tmol.score.coordinates import CartesianAtomicCoordinateProvider

from tmol.utility.reactive import reactive_attrs


def test_system_viewer_smoke(ubq_system):
    SystemViewer(ubq_system)
    SystemViewer(ubq_system, style="stick", mode="cdjson")
    with pytest.raises(NotImplementedError):
        SystemViewer(ubq_system, mode="pdb")


def test_residue_viewer_smoke(ubq_res):
    SystemViewer(ubq_res[0], mode="cdjson")

    with pytest.raises(NotImplementedError):
        SystemViewer(ubq_res[1], mode="pdb")


def test_score_graph_viewer_smoke(ubq_system):
    """Viewer can render score graph as cdjson or pdb."""

    @reactive_attrs
    class MinGraph(BondedAtomScoreGraph, CartesianAtomicCoordinateProvider):
        pass

    ubq_graph = MinGraph.build_for(ubq_system)

    SystemViewer(ubq_graph)
    SystemViewer(ubq_graph, mode="pdb")
