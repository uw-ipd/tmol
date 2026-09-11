"""Unresolved coordinates are identity metadata, not measured bond lengths."""

import math

import biotite.structure as struc
import numpy as np
import pytest
import torch

from tmol.io import atom_array_from_cif, pose_stack_from_biotite
from tmol.ligand import prepare_ligands, load_params_file
from tmol.ligand._preparation import _bond_lengths_by_site
from tmol.ligand._registry import inject_ligand_preparations
from tmol.database import ParameterDatabase
from tmol.tests.data import data_path


def test_attachment_measurements_distinguish_insertion_codes():
    array = struc.AtomArray(4)
    array.chain_id[:] = "A"
    array.res_id[:] = 1
    array.ins_code[:] = ["", "", "A", "A"]
    array.res_name[:] = "LIG"
    array.atom_name[:] = ["C1", "O1", "C1", "O1"]
    array.coord[:] = [[0, 0, 0], [0, 1, 0], [1.4, 1, 0], [1.4, 2, 0]]
    array.bonds = struc.BondList(4, np.array([[0, 1, 1], [1, 2, 1], [2, 3, 1]]))
    lengths = _bond_lengths_by_site(array)
    assert lengths == {
        ("LIG", "C1"): pytest.approx(1.4),
        ("LIG", "O1"): pytest.approx(1.4),
    }


@pytest.mark.parametrize("invalid", [np.nan, np.inf, 0.0])
def test_unresolved_repeat_does_not_overwrite_measured_attachment(invalid):
    array = struc.AtomArray(4)
    array.res_id[:] = [1, 2, 3, 4]
    array.res_name[:] = ["A", "B", "A", "B"]
    array.atom_name[:] = ["O", "C", "O", "C"]
    array.coord[:] = [[0, 0, 0], [1.4, 0, 0], [0, 0, 0], [invalid, 0, 0]]
    array.bonds = struc.BondList(4, np.array([[0, 1, 1], [2, 3, 1]]))
    for indices in ([0, 1, 2, 3], [2, 3, 0, 1]):
        measured = _bond_lengths_by_site(array[indices])
        assert measured == {
            ("A", "O"): pytest.approx(1.4),
            ("B", "C"): pytest.approx(1.4),
        }


@pytest.mark.parametrize("missing", ["NZ", "C11", "both"])
def test_unresolved_biotin_attachment_has_finite_exported_geometry(
    tmp_path, missing, torch_device
):
    array = atom_array_from_cif(data_path("covalent_fixtures", "lys_biotin_1bdo.cif"))
    first, second = next(
        (int(a), int(b))
        for a, b, _ in array.bonds.as_array()
        if {str(array.atom_name[a]), str(array.atom_name[b])} == {"NZ", "C11"}
    )
    for index in (first, second):
        if missing == "both" or array.atom_name[index] == missing:
            array.coord[index] = np.nan
    source = array.coord.copy()
    path = tmp_path / "unresolved-biotin.tmol"
    database, _ = prepare_ligands(array, seed=20250828, params_output=str(path))
    np.testing.assert_array_equal(array.coord, source)
    for rt in database.chemical.residues:
        for ic in rt.icoors:
            assert all(math.isfinite(x) for x in (ic.d, ic.theta, ic.phi)), (
                rt.name,
                ic,
            )
    restored = inject_ligand_preparations(
        ParameterDatabase.get_default(), load_params_file(path)
    )
    assert database.chemical.residues == restored.chemical.residues
    # Re-injection must also accept the same finite patch metadata.
    inject_ligand_preparations(database, load_params_file(path))
    for db in (database, restored):
        if missing != "NZ":
            # Preparation can preserve an unresolved ligand identity, but
            # pose construction still explicitly requires its heavy atoms.
            with pytest.raises(RuntimeError, match="missing heavy atoms"):
                pose_stack_from_biotite(array, torch_device, param_db=db, no_optH=True)
            continue
        pose = pose_stack_from_biotite(array, torch_device, param_db=db, no_optH=True)
        assert torch.isfinite(pose.coords).all()
        active = [
            pose.packed_block_types.active_block_types[t].name
            for t in pose.block_type_ind[0].tolist()
        ]
        assert any("conj_NZ" in name for name in active)
        assert any("BTN:conj_C11" == name for name in active)
