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
@pytest.mark.parametrize("method", ["json", "yaml-loader", "yaml-cloader"])
def test_rama_load_benchmark(benchmark, method):
    path = {
        "json": "tmol/database/default/scoring/rama.json",
        "yaml-loader": "tmol/database/default/scoring/rama.yaml",
        "yaml-cloader": "tmol/database/default/scoring/rama.yaml",
    }[method]

    load = {
        "json": lambda infile: json.load(infile),
        "yaml-loader":
            # defaults to yaml.Loader
            lambda infile: yaml.load(infile),
        "yaml-cloader":
            lambda infile: yaml.load(infile, yaml.CLoader),
    }[method]

    @benchmark
    def db():
        with open(path, "r") as infile:
            raw = load(infile)
        return cattr.structure(raw, RamaDatabase)

    assert len(db.tables) == 40
