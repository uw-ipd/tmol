"""Peptide parameter borrowing follows bonded chemical roles, not file order."""

from types import SimpleNamespace

import biotite.structure as struc
import numpy
import pytest

from tmol.ligand._polymer_profile import (
    noncanonical_junction_substitutions,
)


def _substitutions(array, profile, connections, present=None):
    if present is not None:
        array = array[numpy.isin(array.atom_name, list(present))]
    residue = SimpleNamespace(
        atoms=[
            SimpleNamespace(name=n, atom_type=e)
            for n, e in zip(array.atom_name, array.element)
        ],
        bonds=[
            (array.atom_name[i], array.atom_name[j], struc.BondType(int(order)).name)
            for i, j, order in array.bonds.as_array()
        ],
        connections=[
            SimpleNamespace(
                name="down" if n == profile.mainchain_atoms[0] else "up", atom=n
            )
            for n in connections
        ],
        properties=SimpleNamespace(polymer=profile),
    )
    types = {element: SimpleNamespace(element=element) for element in array.element}
    return noncanonical_junction_substitutions(residue, types)


def _chain(names=("N", "CA", "CG", "CD", "OE2", "OE1", "H")):
    array = struc.AtomArray(7)
    array.atom_name = names
    array.element = ["N", "C", "C", "C", "O", "O", "H"]
    # Deliberately list the leaving hydroxyl before the carbonyl oxygen.
    array.bonds = struc.BondList(
        7,
        numpy.array([[0, 1, 1], [1, 2, 1], [2, 3, 1], [3, 4, 1], [3, 5, 2], [0, 6, 1]]),
    )
    profile = SimpleNamespace(mainchain_atoms=tuple(names[:4]))
    return array, profile


@pytest.mark.parametrize("reverse", [False, True])
def test_carbonyl_oxygen_is_selected_by_bond_order(reverse):
    array, profile = _chain()
    if reverse:
        array = array[::-1]
    assert _substitutions(array, profile, {"CD"}) == [
        {"C": "CD", "CA": "CG", "O": "OE1"}
    ]


@pytest.mark.parametrize("which", ["upper", "lower"])
def test_frame_is_substituted_when_only_the_neighbor_is_renamed(which):
    names = ("N", "CB", "CG", "C", "OX", "O", "H")
    array, profile = _chain(names)
    connection = {"upper": "C", "lower": "N"}[which]
    expected = {
        "upper": {"C": "C", "CA": "CG", "O": "O"},
        "lower": {"N": "N", "CA": "CB", "H": "H"},
    }[which]
    assert _substitutions(array, profile, {connection}) == [expected]


@pytest.mark.parametrize("side", ["upper", "lower"])
def test_phosphate_or_ester_does_not_borrow_peptide_parameters(side):
    array, profile = _chain(("X", "CA", "CG", "Y", "OE2", "OE1", "H"))
    array.element[0 if side == "lower" else 3] = "P"
    assert _substitutions(array, profile, {"X" if side == "lower" else "Y"}) == []


def test_renamed_lower_frame_uses_only_retained_hydrogens():
    array, profile = _chain(("NX", "CA", "CG", "CD", "OE2", "OE1", "HX"))
    assert _substitutions(
        array, profile, {"NX"}, present=set(array.atom_name) - {"HX"}
    ) == [{"N": "NX", "CA": "CA"}]


def test_canonical_frame_requires_no_copied_rows():
    array, profile = _chain(("N", "CA", "CA2", "C", "OX", "O", "H"))
    profile.mainchain_atoms = ("N", "CA", "C")
    assert _substitutions(array, profile, {"N"}) == []


@pytest.mark.parametrize("code", ["5CM", "8OG", "PSU"])
def test_nucleotide_profiles_do_not_acquire_peptide_junction_rows(code):
    import biotite.structure.info as info
    from tmol.ligand._polymer_profile import profile_for_atom_array

    array = info.residue(code)
    connections = {"P", "O3'"}
    profile = profile_for_atom_array(array, connections)
    assert profile is not None
    assert _substitutions(array, profile, connections) == []
