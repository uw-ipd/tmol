import torch

from tmol.score.coordinates import CartesianAtomicCoordinateProvider
from tmol.score.hbond import HBondScoreGraph
from tmol.score.device import TorchDevice

from tmol.system.residue.score import system_cartesian_space_graph_params
from tmol.utility.reactive import reactive_attrs


@reactive_attrs
class HBGraph(
        CartesianAtomicCoordinateProvider,
        HBondScoreGraph,
        TorchDevice,
):
    pass


def test_hbond_smoke(ubq_system, test_hbond_database, torch_device):
    """`bb_only` covers cases missing specific classes of acceptors."""
    hbond_graph = HBGraph(
        hbond_database=test_hbond_database,
        **system_cartesian_space_graph_params(
            ubq_system, requires_grad=True, device=torch_device
        )
    )

    nan_scores = torch.nonzero(torch.isnan(hbond_graph.hbond_scores))
    assert len(nan_scores) == 0
    assert (hbond_graph.total_hbond != 0).all()
    assert hbond_graph.total_score.device == torch_device

    hbond_graph.total_hbond.backward()
    nan_grads = torch.nonzero(torch.isnan(hbond_graph.coords.grad))
    assert len(nan_grads) == 0
