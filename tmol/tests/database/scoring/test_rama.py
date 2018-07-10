import yaml
import cattr
import json
import pytest

from tmol.database.scoring.rama import RamaDatabase


def test_rama_from_json():
    ramadb = RamaDatabase.from_file("tmol/database/default/scoring/rama.json")
    assert len(ramadb.tables) == 40


@pytest.mark.benchmark(
    group="rama_load",
    min_rounds=1,
)
def test_rama_load_json(benchmark):
    path = "tmol/database/default/scoring/rama.json"

    @benchmark
    def db():
        with open(path, "r") as infile:
            raw = json.load(infile)
        return cattr.structure(raw, RamaDatabase)

    assert len(db.tables) == 40


@pytest.mark.benchmark(
    group="rama_load",
    min_rounds=1,
)
def test_rama_load_yaml_loader(benchmark):
    path = "tmol/database/default/scoring/rama.yaml"

    @benchmark
    def db():
        with open(path, "r") as infile:
            raw = yaml.load(infile)
        return cattr.structure(raw, RamaDatabase)

    assert len(db.tables) == 40


@pytest.mark.benchmark(
    group="rama_load",
    min_rounds=1,
)
def test_rama_load_yaml_cloader(benchmark):
    path = "tmol/database/default/scoring/rama.yaml"

    @benchmark
    def db():
        with open(path, "r") as infile:
            raw = yaml.load(infile, Loader=yaml.CLoader)
        return cattr.structure(raw, RamaDatabase)

    assert len(db.tables) == 40
