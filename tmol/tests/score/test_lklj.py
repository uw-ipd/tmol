import unittest
import numpy

import tmol.system.residue
import tmol.score

from tmol.tests.data.pdb import data as test_pdbs
from tmol.system.residue.io import read_pdb
from tmol.system.residue.score import system_real_graph_params


class TestLJLK(unittest.TestCase):
    @unittest.expectedFailure
    def test_numpyros_comparison(self):
        test_pdb = test_pdbs["1ubq"]
        test_structure = read_pdb(test_pdb)

        test_params = system_real_graph_params(
            test_structure,
            drop_missing_atoms=False,
            requires_grad=False,
        )

        expected_scores = {
            'lj_atr': -425.3,
            'lj_rep': 248.8,
            'lk': 255.8,
        }

        numpy.testing.assert_allclose(
            tmol.score.ScoreGraph(**test_params).total_lj.detach(),
            expected_scores["lj_atr"] + expected_scores["lj_rep"],
            rtol=5e-3
        )

        numpy.testing.assert_allclose(
            tmol.score.ScoreGraph(**test_params).total_lk.detach(),
            expected_scores["lk"],
            rtol=5e-3
        )

    def test_baseline_comparison(self):
        test_structure = read_pdb(test_pdbs["1ubq"])

        test_params = system_real_graph_params(
            test_structure,
            drop_missing_atoms=False,
            requires_grad=False,
        )

        test_graph = tmol.score.ScoreGraph(**test_params)

        expected_scores = {
            'total_lj': -176.5,
            'total_lk': 249.3,
        }

        for term, val in expected_scores.items():
            numpy.testing.assert_allclose(
                getattr(test_graph, term).detach(), val, rtol=5e-3
            )
