import numpy
import torch
import pytest

from tmol.database.scoring.rama import RamaDatabase
from tmol.database import ParameterDatabase


def test_rama_construction_smoke():
    fname = "tmol/database/default/scoring/rama/"
    ramadb = RamaDatabase.from_files(fname)
    assert len(ramadb.tables) == 40


def test_rama_mapper():
    ramadb = RamaDatabase.from_files("tmol/database/default/scoring/rama/")
    mapper = ramadb.mapper
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

    path = {"binary": "tmol/database/default/scoring/rama/"}[method]

    load = {"binary": lambda infile: RamaDatabase.from_files(infile)}[method]

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
