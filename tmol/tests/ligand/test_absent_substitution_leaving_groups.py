"""A declared bond at a carbonyl or phosphoryl site displaces one terminal branch."""

import biotite.structure as struc
import numpy as np
import pytest

from tmol.ligand._input_repair import get_absent_substitution_leaving_groups


def carboxyl_template(*, carbonyl: bool = True) -> struc.AtomArray:
    """C1 bearing a double-bonded O and two single-bonded terminal branches."""
    template = struc.AtomArray(5)
    template.atom_name[:] = ["C1", "O1", "O2", "HO2", "O3"]
    template.element[:] = ["C", "O", "O", "H", "O"]
    template.res_name[:] = "LIG"
    template.coord[:] = 0.0
    double = struc.BondType.DOUBLE if carbonyl else struc.BondType.SINGLE
    template.bonds = struc.BondList(
        5,
        np.array(
            [
                [0, 1, int(double)],
                [0, 2, int(struc.BondType.SINGLE)],
                [2, 3, int(struc.BondType.SINGLE)],
                [0, 4, int(struc.BondType.SINGLE)],
            ]
        ),
    )
    return template


def residue_with_absent(absent: set[str]) -> struc.AtomArray:
    """The same component, with the named atoms unobserved (NaN coordinates)."""
    names = ["C1", "O1", "O2", "HO2", "O3"]
    residue = struc.AtomArray(len(names))
    residue.atom_name[:] = names
    residue.element[:] = ["C", "O", "O", "H", "O"]
    residue.res_name[:] = "LIG"
    residue.coord[:] = 0.0
    for name in absent:
        residue.coord[residue.atom_name == name] = np.nan
    return residue


def test_a_single_unobserved_branch_is_the_displaced_one():
    groups = get_absent_substitution_leaving_groups(
        carboxyl_template(), residue_with_absent({"O3"}), {"C1"}
    )
    assert groups == {"C1": ("O3",)}


def test_the_branch_carries_its_own_hydrogens():
    groups = get_absent_substitution_leaving_groups(
        carboxyl_template(), residue_with_absent({"O2", "HO2"}), {"C1"}
    )
    assert groups == {"C1": ("O2", "HO2")}


def test_two_unobserved_branches_stay_ambiguous():
    """Nothing is removed when the file does not say which branch was displaced."""
    groups = get_absent_substitution_leaving_groups(
        carboxyl_template(), residue_with_absent({"O2", "HO2", "O3"}), {"C1"}
    )
    assert groups == {}


def test_a_resolved_branch_is_never_displaced():
    assert (
        get_absent_substitution_leaving_groups(
            carboxyl_template(), residue_with_absent(set()), {"C1"}
        )
        == {}
    )


def test_a_site_without_a_double_bonded_oxygen_is_not_a_substitution_site():
    """Only carbonyl/phosphoryl centres substitute this way."""
    groups = get_absent_substitution_leaving_groups(
        carboxyl_template(carbonyl=False), residue_with_absent({"O3"}), {"C1"}
    )
    assert groups == {}


@pytest.mark.parametrize("connection_atoms", [set(), {"O1"}])
def test_only_the_named_connection_atoms_are_considered(connection_atoms):
    groups = get_absent_substitution_leaving_groups(
        carboxyl_template(), residue_with_absent({"O3"}), connection_atoms
    )
    assert groups == {}
