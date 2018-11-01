import numpy
import torch
import pytest

from tmol.database.scoring.rama import RamaDatabase
from tmol.database import ParameterDatabase


def test_rama_from_json():
    fname = "tmol/database/default/scoring/rama/"
    ramadb = RamaDatabase.from_files(fname)
    assert len(ramadb.tables) == 40


def test_rama_mapper():
    ramadb = RamaDatabase.from_files("tmol/database/default/scoring/rama/")
    mapper = ramadb.mapper
    # assert len(mapper.ndots_to_consider) == 1
    # assert mapper.ndots_to_consider[0] == 3
    # assert mapper.substr_end_for_ndotted_prefix("1.2.3.4.5", 3) == 7
    assert (
        mapper.table_ind_for_res(["aa.alpha.l.alanine"], ["aa.alpha.l.proline"]) == 20
    )
    assert mapper.table_ind_for_res(["aa.alpha.l.alanine"], ["aa.alpha.l.glycine"]) == 0
    assert (
        mapper.table_ind_for_res(
            ["aa.alpha.l.serine.phosphorylated"], ["aa.alpha.l.glycine"]
        )
        == 15
    )


@pytest.mark.benchmark(group="rama_load", min_rounds=1)
@pytest.mark.parametrize("method", ["binary"])
def test_rama_load_benchmark(benchmark, method):
    # import yaml

    path = {
        # "json": "tmol/database/default/scoring/rama.json",
        "binary": "tmol/database/default/scoring/rama/"
    }[method]

    load = {
        # "json": lambda infile: helper_structure_ramadbfromtext(infile),
        "binary": lambda infile: RamaDatabase.from_files(infile)
    }[method]

    @benchmark
    def db():
        return load(path)

    assert len(db.tables) == 40


def test_rama_repr():
    db = ParameterDatabase.get_default()
    rama_repr = repr(db.scoring.rama)
    parts = rama_repr.partition("(")
    assert parts[0] == "RamaDatabase"
    rama_path_parts = parts[2].partition("tmol/database/")
    assert rama_path_parts[2] == "default/scoring/rama/)"
