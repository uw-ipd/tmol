import unittest
import torch
import numpy
import properties


import tmol.system.residue
import tmol.score

from tmol.tests.data.pdb import data as test_pdbs

class TestLJLK(unittest.TestCase):
    def test_rosetta_comparison(self):
        test_pdb = test_pdbs["1ubq"]
        test_structure = tmol.system.residue.read_pdb(test_pdb)

        test_params = tmol.score.system_graph_params(
            test_structure,
            drop_missing_atoms=False
        )

        expected_scores = {
                'lj_atr': -425.3,
                'lj_rep': 248.8,
                'lk': 255.8
        }

        numpy.testing.assert_allclose(
            tmol.score.ScoreGraph(**test_params).total_lj.detach(),
            expected_scores["lj_atr"] + expected_scores["lj_rep"],
            rtol = 5e-3
        )

        self.assertAlmostEqual(
            float(tmol.score.ScoreGraph(**test_params).total_lk),
            expected_scores["lk"]
        )
