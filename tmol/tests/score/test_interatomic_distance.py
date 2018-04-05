import unittest
import torch
import numpy
import properties

from scipy.spatial.distance import pdist, squareform

import tmol.system.residue
from tmol.system.residue.io import read_pdb

import tmol.score.bonded_atom
from tmol.score.interatomic_distance import (
    InteratomicDistanceGraphBase,
    NaiveInteratomicDistanceGraph,
    BlockedInteratomicDistanceGraph
)
from tmol.properties.reactive import derived_from
from tmol.properties.array import VariableT

from tmol.tests.data.pdb import data as test_pdbs

class ThresholdDistanceCount(InteratomicDistanceGraphBase):
    threshold_distance = properties.Float("scoring count distance", min=0, cast=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.score_components.add("total_threshold_count")
        self.atom_pair_dist_thresholds.add(self.threshold_distance)

    @derived_from("atom_pair_dist", VariableT("number of bonds under threshold distance"))
    def total_threshold_count(self):
        return (self.atom_pair_dist < self.threshold_distance).type(torch.LongTensor).sum()

class TestInteratomicDistance(unittest.TestCase):
    def test_naive_distance_calculation(self):
        test_structure = read_pdb(test_pdbs["1ubq"])
        test_params = tmol.score.system_graph_params(
            test_structure,
            drop_missing_atoms=True
        )

        scipy_distance = squareform(pdist(test_structure.coords))

        dgraph = NaiveInteratomicDistanceGraph(**test_params)

        numpy.testing.assert_allclose(
            numpy.nan_to_num(scipy_distance[tuple(dgraph.atom_pair_inds)]),
            numpy.nan_to_num(dgraph.atom_pair_dist.detach()),
            rtol=1e-4
        )

    def test_block_distance_by_naive(self):
        test_structure = read_pdb(test_pdbs["1ubq"])
        test_params = tmol.score.system_graph_params(
            test_structure,
            drop_missing_atoms=True
        )
        test_params["threshold_distance"] = 6

        class NaiveGraph(ThresholdDistanceCount, NaiveInteratomicDistanceGraph):
            pass
        class BlockedGraph(ThresholdDistanceCount, BlockedInteratomicDistanceGraph):
            pass

        scipy_distance = pdist(test_structure.coords)
        scipy_count = numpy.count_nonzero(scipy_distance < 6.0)

        self.assertEqual(
            scipy_count,
            NaiveGraph(**test_params).total_score,
        )

        self.assertEqual(
            NaiveGraph(**test_params).total_score,
            BlockedGraph(**test_params).total_score,
        )

if __name__ == "__main__":
    unittest.main()
