import pytest
import torch
import numpy

from scipy.spatial.distance import pdist, squareform

from tmol.score.device import TorchDevice

from tmol.score.coordinates import CartesianAtomicCoordinateProvider
from tmol.score.total_score import ScoreComponentAttributes, TotalScoreComponentsGraph
from tmol.score.interatomic_distance import (
    InteratomicDistanceGraphBase,
    NaiveInteratomicDistanceGraph,
    BlockedInteratomicDistanceGraph,
    BlockedDistanceAnalysis,
)

from tmol.utility.reactive import reactive_attrs, reactive_property


def test_blocked_interatomic_distance_nulls(multilayer_test_coords):
    """Test that interatomic distance properly sparsifies null blocks."""
    null_padded = multilayer_test_coords.new_full((4, 24 + 8, 3), numpy.nan)
    null_padded[:, :24, :] = multilayer_test_coords

    for l in range(len(multilayer_test_coords)):
        bda = BlockedDistanceAnalysis.setup(
            block_size=8, coords=multilayer_test_coords[l]
        )
        np_bda = BlockedDistanceAnalysis.setup(block_size=8, coords=null_padded[l])
        assert (bda.block_triu_inds == np_bda.block_triu_inds).all()


def test_blocked_interatomic_distance_layered(multilayer_test_coords):
    threshold_distance = 6.0

    expected_block_interactions = torch.Tensor(
        [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # 0-------1------2
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # 2-------1------0
            [[1, 1, 0], [1, 1, 1], [0, 1, 1]],  # ----0---1---2---
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],  # -------012------
        ]
    ).to(dtype=torch.uint8)

    for l in range(len(multilayer_test_coords)):
        bda = BlockedDistanceAnalysis.setup(
            coords=multilayer_test_coords[l], block_size=8
        )
        ebi = expected_block_interactions[l]
        bi = bda.dense_min_dist < threshold_distance
        assert (bi == ebi).all()


@reactive_attrs(auto_attribs=True)
class ThresholdDistanceCount(InteratomicDistanceGraphBase, TotalScoreComponentsGraph):
    threshold_distance: float = 6.0

    @property
    def component_total_score_terms(self):
        return ScoreComponentAttributes(
            name="threshold_count", total="total_threshold_count"
        )

    @property
    def component_atom_pair_dist_threshold(self):
        return self.threshold_distance

    @reactive_property
    def total_threshold_count(atom_pair_dist, threshold_distance):
        "number of bonds under threshold distance"
        return (atom_pair_dist < threshold_distance).type(torch.LongTensor).sum()


@pytest.mark.benchmark(group="interatomic_distance_calculation")
@pytest.mark.parametrize(
    "interatomic_distance_component",
    [NaiveInteratomicDistanceGraph, BlockedInteratomicDistanceGraph],
    ids=["naive", "blocked"],
)
def test_interatomic_distance_smoke(
    benchmark, ubq_system, interatomic_distance_component, torch_device
):
    @reactive_attrs
    class TestGraph(
        CartesianAtomicCoordinateProvider,
        interatomic_distance_component,
        ThresholdDistanceCount,
        TorchDevice,
    ):
        pass

    dgraph = TestGraph.build_for(
        ubq_system, drop_missing_atoms=True, device=torch_device
    )

    scipy_distance = pdist(ubq_system.coords)
    scipy_count = numpy.count_nonzero(
        scipy_distance[~numpy.isnan(scipy_distance)] < 6.0
    )

    numpy.testing.assert_allclose(
        numpy.nan_to_num(squareform(scipy_distance)[tuple(dgraph.atom_pair_inds)]),
        numpy.nan_to_num(dgraph.atom_pair_dist.detach()),
        rtol=1e-4,
    )

    @benchmark
    def total_score():
        # Reset graph by setting coord values,
        # triggering full recalc.
        dgraph.coords = dgraph.coords

        # Calculate total score, rather than atom pair distances
        # As naive implemenation returns a more precise set of distances
        # to the resulting score function.
        return dgraph.total_score

    assert scipy_count == total_score
