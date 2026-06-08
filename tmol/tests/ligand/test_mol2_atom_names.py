"""Tests for Rosetta-style MOL2 duplicate atom-name disambiguation."""

from pathlib import Path

from rdkit import Chem

from tmol.ligand.detect import nonstandard_residue_info_from_mol2
from tmol.ligand.mol2_names import (
    apply_disambiguated_mol2_names,
    disambiguate_mol2_atom_name,
)

PLI_DIR = Path(__file__).parent.parent / "data" / "protein_ligand_test"


def test_disambiguate_mol2_atom_name_follows_mol2gen():
    assert disambiguate_mol2_atom_name("C2'", 1) == "C2'"
    assert disambiguate_mol2_atom_name("C2'", 2) == "C2'2"
    assert disambiguate_mol2_atom_name("C3'", 3) == "C3'3"


def test_fgfr1_mol2_load_has_unique_disambiguated_names():
    mol2_path = PLI_DIR / "fgfr1.lig.mol2"
    info = nonstandard_residue_info_from_mol2(mol2_path, res_name="LG1")
    assert len(info.atom_names) == len(set(info.atom_names))
    assert "C2'2" in info.atom_names
    assert "C3'2" in info.atom_names
    assert info.atom_names.count("C2'") == 1


def test_apply_disambiguated_mol2_names_is_idempotent_on_props():
    mol = Chem.MolFromMol2File(
        str(PLI_DIR / "fgfr1.lig.mol2"),
        sanitize=False,
        removeHs=False,
        cleanupSubstructures=False,
    )
    assert mol is not None
    first = apply_disambiguated_mol2_names(mol)
    second = apply_disambiguated_mol2_names(mol)
    assert first == second
