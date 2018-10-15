from tmol.database.scoring.dunbrack_libraries import DunbrackRotamerLibrary
import zarr

# import cProfile
import pytest

# def test_print_zarr_binary_file_contents():
#    store = zarr.ZipStore(
#        "tmol/database/default/scoring/dunbrack.bin"
#    )
#    zgroup = zarr.group(store=store)
#    print("Zarr tree?")
#    print(zgroup.tree())
#    print("Zarr tree!")


@pytest.mark.benchmark(group="dun_load", min_rounds=1)
def test_load_dunbrack_from_binary(benchmark):
    @benchmark
    def db():
        return DunbrackRotamerLibrary.from_zarr_archive(
            "tmol/database/default/scoring/dunbrack.bin"
        )

    # cProfile.run('DunbrackRotamerLibrary.from_zarr_archive("tmol/database/default/scoring/dunbrack.bin")')

    assert db != None


if __name__ == "__main__":
    DunbrackRotamerLibrary.from_zarr_archive(
        "tmol/database/default/scoring/dunbrack.bin"
    )
