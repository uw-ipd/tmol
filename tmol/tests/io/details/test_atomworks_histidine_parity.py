"""Compare AtomWorks selection directly with tmol's executable HIS fixtures."""

import biotite.structure as struc
import numpy as np
import pytest

from atomworks.io.utils.histidine_tautomer import prepare_histidine_input
from tmol.tests.io.details import test_his_taut_resolution as fixtures


@pytest.mark.parametrize(
    "fixture_name",
    [
        "test_resolve_his_HD1_provided",
        "test_resolve_his_HE2_provided",
        "test_resolve_his_HD1_provided_as_HN",
        "test_resolve_his_HE2_provided_as_HN",
        "test_resolve_his_ND1_provided_as_NH",
        "test_resolve_his_NE2_provided_as_NH",
    ],
)
def test_atomworks_matches_existing_tmol_histidine_fixture(monkeypatch, fixture_name):
    native = fixtures.resolve_his_tautomerization
    checked = []

    def compare(co, res_types, variants, coords, present):
        result = native(co, res_types, variants, coords, present)
        names = co.restypes_ordered_atom_names["HIS"]
        indices = np.flatnonzero(present[0, 0].numpy()[: len(names)])
        atoms = struc.AtomArray(len(indices))
        atoms.res_name[:] = "HIS"
        atoms.res_id[:] = 1
        atoms.atom_name[:] = np.asarray(names)[indices]
        atoms.coord[:] = coords[0, 0, indices].numpy()
        resolved, targets = prepare_histidine_input(atoms)
        chosen = int(result[1][0, 0])
        assert targets[resolved.atom_name == "ND1"].tolist() == [int(chosen in (1, 2))]
        assert targets[resolved.atom_name == "NE2"].tolist() == [int(chosen in (0, 2))]
        for name in ("ND1", "NE2", "HD1", "HE2"):
            if name in resolved.atom_name:
                np.testing.assert_array_equal(
                    resolved.coord[resolved.atom_name == name][0],
                    result[2][0, 0, names.index(name)].numpy(),
                )
        checked.append(True)
        return result

    monkeypatch.setattr(fixtures, "resolve_his_tautomerization", compare)
    getattr(fixtures, fixture_name)()
    assert checked == [True]
