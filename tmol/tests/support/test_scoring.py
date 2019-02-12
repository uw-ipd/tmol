import os
import yaml
import shutil
import zarr

from .rosetta import requires_rosetta_database

from tmol.support.scoring.hbond_param_import import RosettaHBParams
from tmol.support.scoring.rewrite_rama_binary import RamaTableImport


@requires_rosetta_database
def test_hbond_param_import(rosetta_database):
    params = RosettaHBParams(
        os.path.join(rosetta_database, "scoring/score_functions/hbonds/sp2_elec_params")
    )

    yaml.load(params.to_yaml())


@requires_rosetta_database
def test_rama_table_import(rosetta_database):

    out_path = "test_rama_writer.bin"
    if os.path.isdir(out_path):
        shutil.rmtree(out_path)

    rama_path = os.path.join(
        rosetta_database, "scoring/score_functions/rama/fd_beta_nov2016/"
    )
    RamaTableImport.zarr_from_db(rama_path, out_path)

    store2 = zarr.LMDBStore(out_path)
    zgroup = zarr.group(store=store2)
    assert "LAA_LEU_STANDARD" in zgroup
    assert "LAA_THR_STANDARD" in zgroup
    assert "LAA_LEU_PREPRO" in zgroup
    assert "LAA_THR_PREPRO" in zgroup
    assert len(zgroup) == 40

    leu_group = zgroup["LAA_LEU_STANDARD"]
    leu_start = leu_group["bb_start"][:]
    assert len(leu_start.shape) == 1
    assert leu_start.shape[0] == 2
    assert leu_start[0] == -180 and leu_start[1] == -180

    leu_step = leu_group["bb_step"][:]
    assert len(leu_step.shape) == 1
    assert leu_step.shape[0] == 2
    assert leu_step[0] == 10 and leu_step[1] == 10

    leu_probs = leu_group["probabilities"][:]
    assert len(leu_probs.shape) == 2
    assert leu_probs.shape[0] == 36
    assert leu_probs.shape[1] == 36

    store2.close()
    shutil.rmtree(out_path)
