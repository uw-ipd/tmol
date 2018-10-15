from tmol.database.scoring.dunbrack_libraries import DunbrackRotamerLibrary
import zarr

# def test_print_zarr_binary_file_contents():
#    store = zarr.ZipStore(
#        "tmol/database/default/scoring/dunbrack.bin"
#    )
#    zgroup = zarr.group(store=store)
#    print("Zarr tree?")
#    print(zgroup.tree())
#    print("Zarr tree!")


def test_load_dunbrack_from_binary():
    lib = DunbrackRotamerLibrary.from_zarr_archive(
        "tmol/database/default/scoring/dunbrack.bin"
    )
