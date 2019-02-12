import pytest
import torch

from tmol.viewer import SystemViewer

from tmol.score.bonded_atom import BondedAtomScoreGraph
from tmol.score.coordinates import CartesianAtomicCoordinateProvider

from tmol.score.score_graph import score_graph

from argparse import Namespace


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
    """Viewer can render score graph of depth 1 as cdjson or pdb."""

    @score_graph
    class MinGraph(BondedAtomScoreGraph, CartesianAtomicCoordinateProvider):
        pass

    ubq_graph = MinGraph.build_for(ubq_system)

    # Can render depth 1 graph
    SystemViewer(ubq_graph)
    SystemViewer(ubq_graph, mode="pdb")

    # Can not render multi-layer
    stacked_bonds = torch.stack([torch.tensor(ubq_graph.bonds)] * 5)
    stacked_bonds[..., 0] = torch.arange(5)[:, None]
    stacked_bonds = stacked_bonds.reshape((-1, 3))

    ubq_stack = MinGraph.build_for(
        Namespace(
            stack_depth=5,
            system_size=ubq_graph.system_size,
            device=ubq_graph.device,
            coords=ubq_graph.coords.expand(5, -1, -1),
            atom_types=ubq_graph.atom_types.repeat(5, 0),
            atom_elements=ubq_graph.atom_elements.repeat(5, 0),
            atom_names=ubq_graph.atom_names.repeat(5, 0),
            res_names=ubq_graph.res_names.repeat(5, 0),
            res_indices=ubq_graph.res_indices.repeat(5, 0),
            bonds=stacked_bonds,
        )
    )

    with pytest.raises(NotImplementedError):
        SystemViewer(ubq_stack)

    with pytest.raises(NotImplementedError):
        SystemViewer(ubq_stack, mode="pdb")
