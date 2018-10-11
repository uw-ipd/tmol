from tmol.database.scoring.dunbrack_libraries import DunbrackRotamerLibrary


def test_load_dunbrack_from_json():
    lib = DunbrackRotamerLibrary.from_file(
        "tmol/database/default/scoring/dunbrack_library2.json"
    )
