"""Peptide parameter borrowing follows bonded chemical roles, not file order."""

from types import SimpleNamespace

import biotite.structure as struc
import numpy
import pytest

from tmol.database import ParameterDatabase
from tmol.ligand._polymer_profile import (
    noncanonical_junction_substitutions,
    substituted_wildcard_rows,
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


@pytest.mark.parametrize(
    "mapping, length, angles, improper",
    [
        (
            {"C": "CD", "CA": "CG", "O": "OE1"},
            ("CD", "+N"),
            {("CG", "CD", "+N"), ("OE1", "CD", "+N")},
            ("CG", "+N", "CD", "OE1"),
        ),
        (
            {"N": "NX", "CA": "CX", "H": "HX"},
            ("NX", "+C"),
            {("CX", "NX", "+C"), ("HX", "NX", "+C")},
            ("CX", "+C", "NX", "HX"),
        ),
    ],
)
def test_each_renamed_frame_preserves_database_parameters(
    mapping, length, angles, improper
):
    db = ParameterDatabase.get_default().scoring.cartbonded
    rows = substituted_wildcard_rows(db, mapping, None, set(mapping.values()))
    assert length in dict(rows["length"])
    assert angles <= set(dict(rows["angle"]))
    assert improper in dict(rows["improper"])
    inverse = {actual: canonical for canonical, actual in mapping.items()}
    for group, entries in rows.items():
        originals = getattr(db.residue_params["wildcard"], f"{group}_parameters")
        for atoms, source in entries:
            assert source in originals
            assert any(atom.startswith("+") for atom in atoms)
            assert all(atom.startswith("+") or atom in inverse for atom in atoms)


@pytest.mark.parametrize("code", ["5CM", "8OG", "PSU"])
def test_nucleotide_profiles_do_not_acquire_peptide_junction_rows(code):
    import biotite.structure.info as info
    from tmol.ligand._polymer_profile import profile_for_atom_array

    array = info.residue(code)
    connections = {"P", "O3'"}
    profile = profile_for_atom_array(array, connections)
    assert profile is not None
    assert _substitutions(array, profile, connections) == []


def test_prepared_gamma_backbone_carries_carbonyl_junction_parameters():
    from tmol.tests.ligand.test_nonstandard_backbones import _prepare

    preparation = _prepare("FGA")
    parameters = preparation.cartbonded_params
    assert ("CD", "+N") in {(p.atm1, p.atm2) for p in parameters.length_parameters}
    assert {("CG", "CD", "+N"), ("OE1", "CD", "+N")} <= {
        (p.atm1, p.atm2, p.atm3) for p in parameters.angle_parameters
    }


def test_generated_amide_hydrogen_receives_its_connection_angle():
    from tmol.tests.ligand.test_nonstandard_backbones import _residue
    from tmol.ligand import prepare_polymer_residue
    from tmol.ligand._registry import rebuild_canonical_ordering

    array = _residue("FGA")
    array = array[array.element != "H"]
    array.atom_name[array.atom_name == "N"] = "NX"
    db = ParameterDatabase.get_default()
    prep = prepare_polymer_residue(
        array,
        rebuild_canonical_ordering(db),
        db,
        connection_atoms={"NX", "CD"},
        seed=20260909,
    )
    elements = {a.name: a.element for a in db.chemical.atom_types}
    elements.update(prep.atom_type_elements or {})
    hydrogens = {
        a.name for a in prep.residue_type.atoms if elements[a.atom_type] == "H"
    }
    amide_h = {
        b if a == "NX" else a
        for a, b, *_ in prep.residue_type.bonds
        if (a == "NX" and b in hydrogens) or (b == "NX" and a in hydrogens)
    }
    assert len(amide_h) == 1
    h = next(iter(amide_h))
    assert (h, "NX", "+C") in {
        (p.atm1, p.atm2, p.atm3) for p in prep.cartbonded_params.angle_parameters
    }
