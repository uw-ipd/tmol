"""Regeneration repairs oxyacid groups drawn with every terminal oxygen protonated."""

import pytest

from tmol.ligand import prepare_ligand_from_mol2
from tmol.ligand._detect import nonstandard_residue_info_from_mol2
from tmol.ligand._input_repair import deprotonate_overprotonated_oxyacids
from tmol.ligand._openbabel_compat import _build_charged_3d_mol2_mol
from tmol.tests.data import data_path

FIXTURES = data_path("ligand_test", "overprotonated_oxyacids")


@pytest.mark.parametrize("form", ["all_single", "oxonium"])
@pytest.mark.parametrize(
    "stem, charge",
    [("methyl_phosphate", -2), ("methyl_sulfate", -1), ("acetic_acid", -1)],
)
def test_regenerate_deprotonates_overprotonated_oxyacid(stem, charge, form):
    database, _ = prepare_ligand_from_mol2(
        FIXTURES / f"{stem}_{form}.mol2", res_name="LIG", mode="regenerate", seed=17
    )
    charges = [
        row.charge
        for row in database.scoring.elec.atom_charge_parameters
        if row.res == "LIG"
    ]
    assert sum(charges) == pytest.approx(charge, abs=2e-4)

    residue = next(r for r in database.chemical.residues if r.name == "LIG")
    type_element = {t.name: t.element for t in database.chemical.atom_types}
    element = {a.name: type_element[a.atom_type] for a in residue.atoms}
    hydroxyls = [
        (a, b) for a, b, *_ in residue.bonds if {element[a], element[b]} == {"O", "H"}
    ]
    assert hydroxyls == []


@pytest.mark.parametrize(
    "smiles",
    ["COP(=O)(O)O", "OP(O)O", "CO[PH](=O)O", "CS(=O)(=O)O", "CS(=O)O", "CC(=O)O"],
)
def test_valid_acids_are_not_repaired(smiles, tmp_path):
    path = tmp_path / "acid.mol2"
    path.write_text(_build_charged_3d_mol2_mol(smiles, seed=17).write("mol2"))
    array = nonstandard_residue_info_from_mol2(path, res_name="LIG").atom_array

    repaired, promoted = deprotonate_overprotonated_oxyacids(array)
    assert repaired is array
    assert promoted == frozenset()
