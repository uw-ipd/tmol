import numpy
import torch

from tmol.database.scoring.rama import RamaDatabase, CompactedRamaDatabase
from tmol.database import ParameterDatabase


def test_rama_from_json():
    ramadb = RamaDatabase.from_file("tmol/database/default/scoring/rama.json")
    assert len(ramadb.tables) == 40


def test_rama_mapper():
    ramadb = RamaDatabase.from_file("tmol/database/default/scoring/rama.json")
    mapper = ramadb.mapper
    assert len(mapper.ndots_to_consider) == 1
    assert mapper.ndots_to_consider[0] == 3
    assert mapper.substr_end_for_ndotted_prefix("1.2.3.4.5", 3) == 7
    assert (
        mapper.table_ind_for_res(["aa.alpha.l.alanine"], ["aa.alpha.l.proline"]) == 20
    )
    assert mapper.table_ind_for_res(["aa.alpha.l.alanine"], ["aa.alpha.l.glycine"]) == 0


def dont_test_compacted_rama(torch_device):
    ramadb = RamaDatabase.from_file("tmol/database/default/scoring/rama.json")
    compacted = CompactedRamaDatabase.from_ramadb(ramadb, torch_device)
    assert compacted.table.shape == (20, 2, 36, 36)
    phi_vals = (
        torch.arange(36, device=torch_device)
        .reshape(-1, 1)
        .repeat(1, 36)
        .reshape(-1, 1)
    )
    psi_vals = torch.arange(36, device=torch_device).repeat(1, 36).reshape(-1, 1)
    x = torch.cat((phi_vals, psi_vals), dim=1)
    y = torch.full((36 * 36, 2), 0, dtype=torch.long, device=torch_device)
    for i in range(20):
        y[:, 0] = i
        for j in range(2):
            y[:, 1] = j
            xlong = x.type(torch.long)
            inds = (
                y[:, 0] * 2 * 36 * 36
                + y[:, 1] * 36 * 36
                + xlong[:, 0] * 36
                + xlong[:, 1]
            )
            original_vals = compacted.table.reshape(-1)[inds]
            interp_vals = compacted.bspline.interpolate(x, y)
            numpy.testing.assert_allclose(
                interp_vals.cpu().detach().numpy(),
                original_vals.cpu().detach().numpy(),
                atol=1e-5,
                rtol=1e-7,
            )


def dont_test_load_compacted_rama_once(torch_device):
    db = ParameterDatabase.get_default()
    crama1 = CompactedRamaDatabase.from_ramadb(db.scoring.rama, torch_device)
    crama2 = CompactedRamaDatabase.from_ramadb(db.scoring.rama, torch_device)
    assert crama1 is crama2


# @pytest.mark.skip(reason="Slow benchmark in yaml case, not functionally relevant.")
# @pytest.mark.benchmark(group="rama_load", min_rounds=1)
# @pytest.mark.parametrize("method", ["json", "yaml-loader", "yaml-cloader"])
# def test_rama_load_benchmark(benchmark, method):
#     import yaml
#     import json
#     import cattr
#
#     path = {
#         "json": "tmol/database/default/scoring/rama.json",
#         "yaml-loader": "tmol/database/default/scoring/rama.yaml",
#         "yaml-cloader": "tmol/database/default/scoring/rama.yaml",
#     }[method]
#
#     load = {
#         "json": lambda infile: json.load(infile),
#         "yaml-loader":
#         # defaults to yaml.Loader
#         lambda infile: yaml.load(infile),
#         "yaml-cloader": lambda infile: yaml.load(infile, yaml.CLoader),
#     }[method]
#
#     @benchmark
#     def db():
#         with open(path, "r") as infile:
#             raw = load(infile)
#         return cattr.structure(raw, RamaDatabase)
#
#     assert len(db.tables) == 40


def dont_test_rama_request_absent_aa():
    db = ParameterDatabase.get_default()
    rama_table = db.scoring.rama.find("CYX", False)
    assert rama_table is None


def dont_test_rama_repr():
    db = ParameterDatabase.get_default()
    rama_repr = repr(db.scoring.rama)
    parts = rama_repr.partition("(")
    assert parts[0] == "RamaDatabase"
    rama_path_parts = parts[2].partition("tmol/database/")
    assert rama_path_parts[2] == "default/scoring/rama.json)"
