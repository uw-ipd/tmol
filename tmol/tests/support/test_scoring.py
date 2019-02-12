import os
import yaml
import shutil
import numpy
import zarr

from .rosetta import requires_rosetta_database

from tmol.support.scoring.hbond_param_import import RosettaHBParams
from tmol.support.scoring.rewrite_rama_binary import parse_lines_as_ndarrays


@requires_rosetta_database
def test_hbond_param_import(rosetta_database):
    params = RosettaHBParams(
        os.path.join(rosetta_database, "scoring/score_functions/hbonds/sp2_elec_params")
    )

    yaml.load(params.to_yaml())


@requires_rosetta_database
def test_rama_table_matches_rosetta3(rosetta_database, default_database):
    rama_path = os.path.join(
        rosetta_database, "scoring/score_functions/rama/fd_beta_nov2016/"
    )
    r3_general = parse_lines_as_ndarrays(open(rama_path + "all.ramaProb").readlines())
    r3_prepro_case = parse_lines_as_ndarrays(
        open(rama_path + "prepro.ramaProb").readlines()
    )

    ramatables = default_database.scoring.rama.rama_tables

    assert len(ramatables) == 40

    table_keys = [i.name for i in ramatables]
    for aa in r3_general:
        gen_idx = table_keys.index(aa)
        numpy.testing.assert_allclose(r3_general[aa], ramatables[gen_idx].prob)
        assert ramatables[gen_idx].bbstep == [10.0, 10.0]
        assert ramatables[gen_idx].bbstart == [-180.0, -180.0]

        prepro_idx = table_keys.index(aa + "_prepro")
        numpy.testing.assert_allclose(r3_prepro_case[aa], ramatables[prepro_idx].prob)
        assert ramatables[prepro_idx].bbstep == [10.0, 10.0]
        assert ramatables[prepro_idx].bbstart == [-180.0, -180.0]
