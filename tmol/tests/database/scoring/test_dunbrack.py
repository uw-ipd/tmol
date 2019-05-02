from tmol.database.scoring.dunbrack_libraries import DunbrackRotamerLibrary

import pytest


@pytest.mark.benchmark(group="dun_load", min_rounds=1)
def test_load_dunbrack_from_binary(benchmark):
    @benchmark
    def db():
        return DunbrackRotamerLibrary.from_zarr_archive(
            "tmol/database/default/scoring/dunbrack.yaml",
            "tmol/database/default/scoring/dunbrack.bin",
        )

    assert db != None
