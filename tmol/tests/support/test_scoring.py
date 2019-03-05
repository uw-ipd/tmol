import os
import yaml
import shutil
import numpy
import zarr

from .rosetta import requires_rosetta_database

from tmol.support.scoring.hbond_param_import import RosettaHBParams
from tmol.support.scoring.rewrite_rama_binary import parse_all_tables


@requires_rosetta_database
def test_hbond_param_import(rosetta_database):
    params = RosettaHBParams(
        os.path.join(rosetta_database, "scoring/score_functions/hbonds/sp2_elec_params")
    )

    yaml.load(params.to_yaml())


@requires_rosetta_database
def test_rama_table_read(rosetta_database, default_database):
    r3_rama_dir = os.path.join(
        rosetta_database, "scoring/score_functions/rama/fd_beta_nov2016/"
    )
    r3_paapp_dir = os.path.join(
        rosetta_database, "scoring/score_functions/P_AA_pp/shapovalov/10deg/kappa131/"
    )
    r3_paa_dir = os.path.join(rosetta_database, "scoring/score_functions/P_AA_pp/")

    r3_general, r3_prepro = parse_all_tables(
        0.5, r3_rama_dir, 0.61, r3_paapp_dir, r3_paa_dir
    )

    ramatables = default_database.scoring.rama.rama_tables

    assert len(ramatables) == 40
    table_keys = [i.name for i in ramatables]
    for aa in r3_general:
        gen_idx = table_keys.index(aa)
        numpy.testing.assert_allclose(r3_general[aa], ramatables[gen_idx].table)
        numpy.testing.assert_allclose(
            ramatables[gen_idx].bbstep, numpy.array([numpy.pi / 18.0, numpy.pi / 18.0])
        )
        numpy.testing.assert_allclose(
            ramatables[gen_idx].bbstart, numpy.array([-numpy.pi, -numpy.pi])
        )

        prepro_idx = table_keys.index(aa + "_prepro")
        numpy.testing.assert_allclose(r3_prepro[aa], ramatables[prepro_idx].table)
        numpy.testing.assert_allclose(
            ramatables[prepro_idx].bbstep,
            numpy.array([numpy.pi / 18.0, numpy.pi / 18.0]),
        )
        numpy.testing.assert_allclose(
            ramatables[prepro_idx].bbstart, numpy.array([-numpy.pi, -numpy.pi])
        )
