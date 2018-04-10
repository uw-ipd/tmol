import os
import yaml
import unittest

from .rosetta import rosetta_database

from tmol.support.scoring.hbond_param_import import RosettaHBParams


class TestHBondParamImport(unittest.TestCase):
    @unittest.skipIf(not rosetta_database, "rosetta database not available")
    def test_hbond_parm_import(self):
        params = RosettaHBParams(
            os.path.join(
                rosetta_database,
                "scoring/score_functions/hbonds/sp2_elec_params"
            )
        )

        yaml.load(params.to_yaml())
