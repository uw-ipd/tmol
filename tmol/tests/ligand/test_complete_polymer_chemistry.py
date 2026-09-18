"""Preparation accepts complete free termini and partially observed phosphates."""

import biotite.structure.info as info
import numpy as np
import pytest

from tmol.database import ParameterDatabase
from tmol.ligand._polymer_profile import (
    cap_residue,
    complete_backbone_from_reference,
    na_profile,
    profile_for_atom_array,
)
from tmol.ligand._structure_to_smiles import ligand_smiles_from_atom_array


@pytest.mark.parametrize("code", ["BAL", "ASP"])
def test_capping_replaces_terminal_hydroxyl_and_preserves_sidechain(code):
    array = info.residue(code)
    db = ParameterDatabase.get_default()
    profile = profile_for_atom_array(array, {"N", "C"}, db.chemical)
    capped, _ = cap_residue(array, profile)
    assert "OXT" not in capped.atom_name
    if code == "ASP":
        assert {"OD1", "OD2"} <= set(capped.atom_name)
    carbon = int(np.where(capped.atom_name == "C")[0][0])
    bonds = capped.bonds.as_array()
    assert sum(int(o) for i, j, o in bonds if carbon in (i, j)) == 4
    assert ligand_smiles_from_atom_array(capped, res_name=code)
    assert "OXT" in array.atom_name  # Source was not modified.


@pytest.mark.parametrize("missing", ["OP1", "OP2"])
def test_phosphate_completion_preserves_one_double_bond(missing):
    db = ParameterDatabase.get_default()
    profile = na_profile(db.chemical, "dna")
    array = info.residue("8OG")
    array = array[~np.isin(array.atom_name, ["OP3", missing])]
    # Either localized resonance form is valid; the observed oxygen is double.
    bonds = array.bonds.as_array()
    for edge in bonds:
        names = {str(array.atom_name[i]) for i in edge[:2]}
        if "P" in names and names & {"OP1", "OP2"}:
            edge[2] = 2
    from biotite.structure import BondList

    array.bonds = BondList(len(array), bonds)
    completed = complete_backbone_from_reference(array, profile, db)
    phosphorus = int(np.where(completed.atom_name == "P")[0][0])
    orders = [int(o) for i, j, o in completed.bonds.as_array() if phosphorus in (i, j)]
    assert sorted(orders) == [1, 1, 2]
    capped, _ = cap_residue(completed, profile)
    assert ligand_smiles_from_atom_array(capped, res_name="8OG")
