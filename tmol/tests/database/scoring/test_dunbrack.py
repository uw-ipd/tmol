from tmol.database.scoring.dunbrack_libraries import DunbrackRotamerLibrary

import pytest
import os


@pytest.mark.benchmark(group="dun_load", min_rounds=1)
def test_load_dunbrack_from_binary(benchmark):
    dirname = os.path.dirname(__file__)

    @benchmark
    def db():
        return DunbrackRotamerLibrary.from_zarr_archive(
            os.path.join(dirname, "../../../database/default/scoring/dunbrack.yaml"),
            os.path.join(dirname, "../../../database/default/scoring/dunbrack.bin"),
        )

    assert db is not None
