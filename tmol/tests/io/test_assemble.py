"""A protein file and a ligand file become one CIF that describes itself."""

import numpy as np
import pytest

from tmol.io import (
    assemble_input,
    atom_array_from_file,
    atom_array_from_mol2,
    cif_from_atom_array,
)
from tmol.tests.data import data_path

LIGAND_RES_NAME = "LG1"


@pytest.fixture
def ligand():
    return atom_array_from_mol2(
        data_path("protein_ligand_test") / "ace.lig.mol2",
        res_name=LIGAND_RES_NAME,
    )


@pytest.fixture
def protein():
    return atom_array_from_file(data_path("pdb") / "bysize_015_res_1lu6.pdb")


def test_a_mol2_ligand_arrives_with_its_bond_orders(ligand):
    """A MOL2 states bond orders the dictionary could not supply."""
    assert ligand.array_length() > 0
    assert ligand.bonds is not None and ligand.bonds.get_bond_count() > 0
    assert set(ligand.res_name) == {LIGAND_RES_NAME}
    # Bond orders are read, not guessed: something beyond a plain single bond.
    orders = {int(row[2]) for row in ligand.bonds.as_array()}
    assert orders - {1}


def test_joining_keeps_both_parts_whole(protein, ligand):
    joined = assemble_input(protein, ligand)

    assert joined.array_length() == protein.array_length() + ligand.array_length()
    assert joined.bonds.get_bond_count() == (
        protein.bonds.get_bond_count() + ligand.bonds.get_bond_count()
    )
    # The ligand does not land on a chain the protein already claims.
    protein_chains = {str(c) for c in np.unique(protein.chain_id)}
    ligand_chains = {
        str(c) for c in np.unique(joined.chain_id[joined.res_name == LIGAND_RES_NAME])
    }
    assert not (protein_chains & ligand_chains)


def test_the_written_cif_carries_the_ligand_chemistry(protein, ligand):
    """The ligand describes itself, because no dictionary describes it."""
    text = cif_from_atom_array(assemble_input(protein, ligand))

    assert "_chem_comp_atom" in text
    assert "_chem_comp_bond" in text
    # The component rows name the ligand, not only the standard residues.
    comp_rows = [line for line in text.splitlines() if LIGAND_RES_NAME in line]
    assert comp_rows


def test_the_written_cif_parses_back_into_the_same_ligand(protein, ligand, tmp_path):
    """The round trip is the whole point: chemistry travels inside the file.

    A ligand code is a placeholder, and placeholders collide -- "LG1" is also a real
    dictionary entry -- so this also pins that the file describes the molecule that was
    read rather than whatever else shares its name.
    """
    path = cif_from_atom_array(
        assemble_input(protein, ligand), path=tmp_path / "complex.cif"
    )
    reparsed = atom_array_from_file(path)

    restored = reparsed[reparsed.res_name == LIGAND_RES_NAME]
    assert restored.array_length() == ligand.array_length()
    assert set(restored.atom_name) == set(ligand.atom_name)

    def bond_name_pairs(array):
        return {
            frozenset((str(array.atom_name[i]), str(array.atom_name[j])))
            for i, j, _ in array.bonds.as_array()
        }

    assert bond_name_pairs(restored) == bond_name_pairs(ligand)
