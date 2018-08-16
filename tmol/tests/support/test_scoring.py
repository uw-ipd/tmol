import os
import yaml

from .rosetta import requires_rosetta_database

from tmol.support.scoring.hbond_param_import import RosettaHBParams


@requires_rosetta_database
def test_hbond_param_import(rosetta_database):
    params = RosettaHBParams(
        os.path.join(rosetta_database, "scoring/score_functions/hbonds/sp2_elec_params")
    )

    yaml.load(params.to_yaml())
