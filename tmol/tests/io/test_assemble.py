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


@pytest.mark.parametrize("route", ["text", "path"])
def test_the_written_cif_parses_back_into_the_same_ligand(
    protein, ligand, tmp_path, route
):
    """The file carries the molecule read, not the dictionary's own "LG1" entry."""
    path = tmp_path / "complex.cif"
    if route == "text":
        path.write_text(cif_from_atom_array(assemble_input(protein, ligand)))
    else:
        assert cif_from_atom_array(assemble_input(protein, ligand), path=path) == path
    reparsed = atom_array_from_file(path)

    restored = reparsed[reparsed.res_name == LIGAND_RES_NAME]
    assert restored.array_length() == ligand.array_length()
    assert set(restored.atom_name) == set(ligand.atom_name)

    def bonds_by_name(array):
        # Orders travel with their pairs: connectivity kept with dictionary orders is the
        # failure this file exists to prevent.
        return {
            (frozenset((str(array.atom_name[i]), str(array.atom_name[j]))), int(order))
            for i, j, order in array.bonds.as_array()
        }

    assert bonds_by_name(restored) == bonds_by_name(ligand)
