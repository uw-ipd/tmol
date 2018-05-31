import unittest
import torch
import numpy

from scipy.spatial.distance import pdist, squareform

from tmol.system.residue.io import read_pdb

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

from tmol.tests.data.pdb import data as test_pdbs

from tmol.system.residue.score import system_cartesian_space_graph_params

from tmol.utility.reactive import reactive_attrs, reactive_property


@reactive_attrs(auto_attribs=True)
class ThresholdDistanceCount(
        InteratomicDistanceGraphBase,
        TotalScoreComponentsGraph,
):
    threshold_distance: float

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


class TestInteratomicDistance(unittest.TestCase):
    def test_naive_distance_calculation(self):
        test_structure = read_pdb(test_pdbs["1ubq"])
        test_params = system_cartesian_space_graph_params(
            test_structure, drop_missing_atoms=True
        )

        scipy_distance = squareform(pdist(test_structure.coords))

        @reactive_attrs
        class TestGraph(
                CartesianAtomicCoordinateProvider,
                NaiveInteratomicDistanceGraph,
                TorchDevice,
        ):
            pass

        dgraph = TestGraph(**test_params)

        numpy.testing.assert_allclose(
            numpy.nan_to_num(scipy_distance[tuple(dgraph.atom_pair_inds)]),
            numpy.nan_to_num(dgraph.atom_pair_dist.detach()),
            rtol=1e-4
        )

    def test_block_distance_by_naive(self):
        test_structure = read_pdb(test_pdbs["1ubq"])
        test_params = system_cartesian_space_graph_params(
            test_structure, drop_missing_atoms=True
        )
        test_params["threshold_distance"] = 6

        @reactive_attrs
        class NaiveGraph(
                CartesianAtomicCoordinateProvider,
                ThresholdDistanceCount,
                NaiveInteratomicDistanceGraph,
                TorchDevice,
        ):
            pass

        @reactive_attrs
        class BlockedGraph(
                CartesianAtomicCoordinateProvider,
                ThresholdDistanceCount,
                BlockedInteratomicDistanceGraph,
                TorchDevice,
        ):
            pass

        scipy_distance = pdist(test_structure.coords)
        scipy_count = numpy.count_nonzero(
            scipy_distance[~numpy.isnan(scipy_distance)] < 6.0
        )

        self.assertEqual(
            scipy_count,
            NaiveGraph(**test_params).total_score,
        )

        self.assertEqual(
            NaiveGraph(**test_params).total_score,
            BlockedGraph(**test_params).total_score,
        )
