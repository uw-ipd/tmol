import os
import yaml
import numpy
import torch

from .rosetta import requires_rosetta_database

from tmol.support.scoring.hbond_param_import import RosettaHBParams
from tmol.support.scoring.rewrite_rama_binary import parse_all_tables
from tmol.support.scoring.rewrite_omega_bbdep_binary import parse_all_tables
from tmol.support.scoring.rewrite_dunbrack_binary import (
    write_binary_version_of_dunbrack_rotamer_library,
)
from tmol.database.scoring.dunbrack_libraries import DunbrackRotamerLibrary


@requires_rosetta_database
def test_hbond_param_import(rosetta_database):
    params = RosettaHBParams(
        os.path.join(rosetta_database, "scoring/score_functions/hbonds/sp2_elec_params")
    )

    yaml.safe_load(params.to_yaml())


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
    table_keys = [i.table_id for i in ramatables]
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


@requires_rosetta_database
def test_bbdep_omega_table_read(rosetta_database):
    r3_bbdepomega_dir = os.path.join(rosetta_database, "scoring/score_functions/omega/")

    tables = parse_all_tables(r3_bbdepomega_dir)


@requires_rosetta_database
def test_dunbrack_table_read(rosetta_database, default_database):
    # clean up if previous unit test execution failed
    if os.path.isfile("test_dunbrack.bin"):
        os.remove("test_dunbrack.bin")

    r3_beta16_dunbrack_dir = os.path.join(rosetta_database, "rotamer/beta_nov2016")
    r3_beta15_dunbrack_dir = os.path.join(rosetta_database, "rotamer/ExtendedOpt1-5")
    write_binary_version_of_dunbrack_rotamer_library(
        r3_beta16_dunbrack_dir, r3_beta15_dunbrack_dir, "test_dunbrack.bin"
    )
    fresh = DunbrackRotamerLibrary.from_zarr_archive(
        os.path.join(os.path.dirname(__file__), "copy_dunbrack.yml"),
        "test_dunbrack.bin",
    )

    default = default_database.scoring.dun

    assert len(default.dun_lookup) == len(fresh.dun_lookup)
    assert len(default.rotameric_libraries) == len(fresh.rotameric_libraries)
    assert len(default.semi_rotameric_libraries) == len(fresh.semi_rotameric_libraries)

    def compare_rotameric_data(default_rotdat, fresh_rotdat):
        torch.testing.assert_allclose(default_rotdat.rotamers, fresh_rotdat.rotamers)
        torch.testing.assert_allclose(
            default_rotdat.rotamer_probabilities, fresh_rotdat.rotamer_probabilities
        )
        torch.testing.assert_allclose(
            default_rotdat.rotamer_means, fresh_rotdat.rotamer_means
        )
        torch.testing.assert_allclose(
            default_rotdat.backbone_dihedral_start, fresh_rotdat.backbone_dihedral_start
        )
        torch.testing.assert_allclose(
            default_rotdat.backbone_dihedral_step, fresh_rotdat.backbone_dihedral_step
        )
        torch.testing.assert_allclose(
            default_rotdat.rotamer_alias, fresh_rotdat.rotamer_alias
        )

    for i in range(len(default.rotameric_libraries)):
        assert (
            default.rotameric_libraries[i].table_name
            == fresh.rotameric_libraries[i].table_name
        )
        compare_rotameric_data(
            default.rotameric_libraries[i].rotameric_data,
            fresh.rotameric_libraries[i].rotameric_data,
        )

    def compare_semirotameric_data(default_srdat, fresh_srdat):
        compare_rotameric_data(default_srdat.rotameric_data, fresh_srdat.rotameric_data)
        torch.testing.assert_allclose(
            default_srdat.non_rot_chi_start, fresh_srdat.non_rot_chi_start
        )
        torch.testing.assert_allclose(
            default_srdat.non_rot_chi_step, fresh_srdat.non_rot_chi_step
        )
        torch.testing.assert_allclose(
            default_srdat.non_rot_chi_period, fresh_srdat.non_rot_chi_period
        )
        torch.testing.assert_allclose(
            default_srdat.rotameric_chi_rotamers, fresh_srdat.rotameric_chi_rotamers
        )
        torch.testing.assert_allclose(
            default_srdat.nonrotameric_chi_probabilities,
            fresh_srdat.nonrotameric_chi_probabilities,
        )
        torch.testing.assert_allclose(
            default_srdat.rotamer_boundaries, fresh_srdat.rotamer_boundaries
        )

    for i in range(len(default.semi_rotameric_libraries)):
        assert (
            default.semi_rotameric_libraries[i].table_name
            == fresh.semi_rotameric_libraries[i].table_name
        )
        compare_semirotameric_data(
            default.semi_rotameric_libraries[i], fresh.semi_rotameric_libraries[i]
        )
    os.remove("test_dunbrack.bin")
