import pytest
import torch
import numpy

from scipy.spatial.distance import pdist, squareform

from tmol.score.device import TorchDevice

from tmol.score.coordinates import (
    CartesianAtomicCoordinateProvider,
)
from tmol.score.total_score import (
    ScoreComponentAttributes,
    TotalScoreComponentsGraph,
)
from tmol.score.interatomic_distance import (
    InteratomicDistanceGraphBase,
    NaiveInteratomicDistanceGraph,
    BlockedInteratomicDistanceGraph,
)

from tmol.system.residue.score import system_cartesian_space_graph_params

from tmol.utility.reactive import reactive_attrs, reactive_property


@reactive_attrs(auto_attribs=True)
class ThresholdDistanceCount(
        InteratomicDistanceGraphBase,
        TotalScoreComponentsGraph,
):
    threshold_distance: float = 6.0

    @property
    def component_total_score_terms(self):
        return ScoreComponentAttributes(
            name="threshold_count",
            total="total_threshold_count",
        )

    @property
    def component_atom_pair_dist_threshold(self):
        return self.threshold_distance

    @reactive_property
    def total_threshold_count(atom_pair_dist, threshold_distance):
        "number of bonds under threshold distance"
        return ((atom_pair_dist < threshold_distance)
                .type(torch.LongTensor).sum())


@pytest.mark.parametrize(
    "interatomic_distance_component",
    [NaiveInteratomicDistanceGraph, BlockedInteratomicDistanceGraph],
    ids=["naive", "blocked"],
)
def test_interatomic_distance(
        ubq_system,
        interatomic_distance_component,
        torch_device,
):
    test_params = system_cartesian_space_graph_params(
        ubq_system,
        drop_missing_atoms=True,
        device=torch_device,
    )

    @reactive_attrs
    class TestGraph(
            CartesianAtomicCoordinateProvider,
            interatomic_distance_component,
            ThresholdDistanceCount,
            TorchDevice,
    ):
        pass

    dgraph = TestGraph(**test_params)

    scipy_distance = pdist(ubq_system.coords)
    scipy_count = numpy.count_nonzero(
        scipy_distance[~numpy.isnan(scipy_distance)] < 6.0
    )

    numpy.testing.assert_allclose(
        numpy.nan_to_num(
            squareform(scipy_distance)[tuple(dgraph.atom_pair_inds)]
        ),
        numpy.nan_to_num(dgraph.atom_pair_dist.detach()),
        rtol=1e-4
    )

    assert scipy_count == dgraph.total_score
