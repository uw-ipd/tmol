import tmol.support.scoring.rewrite_rama_binary
import zarr
import os
import shutil


def test_write_leu_binary():
    with open("tmol/tests/support/leu_thr.ramaProb") as fid:
        lines = fid.readlines()

    if os.path.isdir("test_rama_writer.bin"):
        shutil.rmtree("test_rama_writer.bin")
    store = zarr.LMDBStore("test_rama_writer.bin")
    zgroup = zarr.group(store=store)

    tmol.support.scoring.rewrite_rama_binary.write_lines_to_zarr(lines, False, zgroup)
    tmol.support.scoring.rewrite_rama_binary.write_lines_to_zarr(lines, True, zgroup)

    store.close()

    store2 = zarr.LMDBStore("test_rama_writer.bin")
    zgroup = zarr.group(store=store2)
    assert "LAA_LEU_STANDARD" in zgroup
    assert "LAA_THR_STANDARD" in zgroup
    assert "LAA_LEU_PREPRO" in zgroup
    assert "LAA_THR_PREPRO" in zgroup

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
    shutil.rmtree("test_rama_writer.bin")
