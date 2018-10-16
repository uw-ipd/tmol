import numpy
import torch
import pytest

from tmol.database.scoring.rama import RamaDatabase, CompactedRamaDatabase
from tmol.database import ParameterDatabase


def test_rama_from_json():
    fname = "tmol/database/default/scoring/rama/"
    ramadb = RamaDatabase.from_files(fname)
    assert len(ramadb.tables) == 40


def test_rama_mapper():
    ramadb = RamaDatabase.from_files("tmol/database/default/scoring/rama/")
    mapper = ramadb.mapper
    assert len(mapper.ndots_to_consider) == 1
    assert mapper.ndots_to_consider[0] == 3
    assert mapper.substr_end_for_ndotted_prefix("1.2.3.4.5", 3) == 7
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


def test_compacted_rama(torch_device):
    ramadb = RamaDatabase.from_files("tmol/database/default/scoring/rama/")
    compacted = CompactedRamaDatabase.from_ramadb(ramadb, torch_device)
    assert compacted.table.shape == (40, 36, 36)
    phi_vals = (
        torch.arange(36, device=torch_device)
        .reshape(-1, 1)
        .repeat(1, 36)
        .reshape(-1, 1)
    )
    psi_vals = torch.arange(36, device=torch_device).repeat(1, 36).reshape(-1, 1)
    x = torch.cat((phi_vals, psi_vals), dim=1)
    y = torch.full((36 * 36, 1), 0, dtype=torch.long, device=torch_device)
    for i in range(20):
        for j in range(2):
            y[:, 0] = i + 20 * j
            xlong = x.type(torch.long)
            inds = y[:, 0] * 36 * 36 + xlong[:, 0] * 36 + xlong[:, 1]
            original_vals = compacted.table.reshape(-1)[inds]
            interp_vals = compacted.bspline.interpolate(x, y)
            numpy.testing.assert_allclose(
                interp_vals.cpu().detach().numpy(),
                original_vals.cpu().detach().numpy(),
                atol=1e-5,
                rtol=1e-7,
            )


def test_load_compacted_rama_once(torch_device):
    db = ParameterDatabase.get_default()
    crama1 = CompactedRamaDatabase.from_ramadb(db.scoring.rama, torch_device)
    crama2 = CompactedRamaDatabase.from_ramadb(db.scoring.rama, torch_device)
    assert crama1 is crama2


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
