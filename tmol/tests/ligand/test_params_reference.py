"""Tests for the shared Rosetta ``.params`` reference parser and charge sidecar."""

from __future__ import annotations

from pathlib import Path

import pytest

from types import SimpleNamespace

from tmol.ligand.params_reference import (
    GeneratedFields,
    ReferenceParams,
    as_legacy_dict,
    compare_charges,
    compare_params_strict,
    compare_semantic,
    parse_reference_params,
    reference_bond_keys,
    reference_charges,
)


def _matching_generated_fields(ref: ReferenceParams) -> GeneratedFields:
    """Build a GeneratedFields that exactly matches a reference."""
    return GeneratedFields(
        atom_types=dict(ref.atom_types),
        bond_keys=frozenset(reference_bond_keys(ref)),
        icoor_topology=dict(ref.icoor_topology),
        nbr_atom=ref.nbr_atom,
        charges=dict(ref.charges),
    )


_GROUND_TRUTH = Path(__file__).parent.parent / "data" / "ligand_ground_truth"
_REF1 = _GROUND_TRUTH / "ref1.params"
_REF2 = _GROUND_TRUTH / "ref2.params"


def _atom_record_count(path: Path) -> int:
    """Count ATOM records in a Rosetta .params file."""
    return sum(1 for line in path.open() if line.split()[:1] == ["ATOM"])


def test_charge_sidecar_length_matches_atom_records() -> None:
    """Charge sidecar length and heavy-atom set match the ATOM records."""
    ref = parse_reference_params(_REF1)
    # ref1 has 31 ATOM records (17 heavy + 14 H).
    assert len(ref.charges) == _atom_record_count(_REF1) == 31
    assert ref.heavy_atom_names() == {
        name for name, _t, _q in ref.atoms if not name.startswith("H")
    }
    assert len(ref.heavy_atom_names()) == 17


def test_parse_captures_name_and_nbr_and_charges_are_floats() -> None:
    """Parsing captures name/NBR_ATOM and yields float charges."""
    ref = parse_reference_params(_REF1)
    assert ref.name == "ref1"
    assert ref.nbr_atom  # a real NBR_ATOM is present
    assert ref.has_hydrogen
    assert all(isinstance(q, float) for q in ref.charges.values())


def test_reference_charges_accepts_path_and_object() -> None:
    """reference_charges accepts both a path and a parsed object."""
    ref = parse_reference_params(_REF2)
    from_path = reference_charges(_REF2)
    from_obj = reference_charges(ref)
    assert from_path == from_obj == ref.charges


def test_all_bond_pairs_are_hydrogen_inclusive() -> None:
    """all_bond_pairs includes bonds that touch hydrogens."""
    ref = parse_reference_params(_REF1)
    pairs = ref.all_bond_pairs()
    # at least one bond must touch a hydrogen (mol2/params keep explicit H).
    assert any(any(n.startswith("H") for n in pair) for pair in pairs)


def test_legacy_dict_shape_is_preserved() -> None:
    """as_legacy_dict preserves the expected key set and contents."""
    ref = parse_reference_params(_REF1)
    legacy = as_legacy_dict(ref)
    assert set(legacy) == {
        "atoms",
        "bond_types",
        "cut_bonds",
        "chis",
        "proton_chis",
        "nbr_atom",
        "icoor_topology",
    }
    assert legacy["atoms"] == list(ref.atoms)
    assert legacy["bond_types"] == set(ref.bond_types)


def test_compare_charges_identical_passes() -> None:
    """Identical charge dicts compare as equivalent."""
    charges = reference_charges(_REF1)
    result = compare_charges(charges, dict(charges), tolerance=0.01)
    assert result.ok
    assert result.mismatches == []
    assert result.missing_in_generated == []
    assert result.extra_in_generated == []


def test_compare_charges_perturbation_fails_beyond_tolerance() -> None:
    """A charge perturbation beyond tolerance is reported as a mismatch."""
    charges = reference_charges(_REF1)
    perturbed = dict(charges)
    victim = next(iter(perturbed))
    perturbed[victim] = perturbed[victim] + 0.2
    result = compare_charges(perturbed, charges, tolerance=0.05)
    assert not result.ok
    assert any(name == victim for name, *_ in result.mismatches)


def test_compare_charges_no_shared_atoms_is_not_ok() -> None:
    """Charge dicts with no shared atom names do not compare as equivalent."""
    result = compare_charges({"X1": 0.1}, {"Y1": 0.1}, tolerance=0.05)
    assert not result.ok
    assert result.mismatches == []
    assert result.missing_in_generated == ["Y1"]
    assert result.extra_in_generated == ["X1"]


def test_compare_charges_missing_key_fails_by_default() -> None:
    """A missing generated charge key fails the default comparison."""
    charges = reference_charges(_REF1)
    generated = dict(charges)
    dropped = generated.popitem()[0]
    result = compare_charges(generated, charges, tolerance=0.01)
    assert not result.ok
    assert dropped in result.missing_in_generated


def test_compare_charges_extra_key_fails_by_default() -> None:
    """An extra generated charge key fails the default comparison."""
    charges = reference_charges(_REF1)
    generated = dict(charges)
    generated["ZZ99"] = 0.0
    result = compare_charges(generated, charges, tolerance=0.01)
    assert not result.ok
    assert "ZZ99" in result.extra_in_generated


def test_compare_charges_subset_mode_allows_missing_keys() -> None:
    """Subset mode tolerates missing generated charge keys."""
    charges = reference_charges(_REF1)
    generated = dict(charges)
    generated.popitem()
    result = compare_charges(
        generated, charges, tolerance=0.01, require_same_keys=False
    )
    assert result.ok


def test_reference_params_is_frozen() -> None:
    """ReferenceParams instances are immutable."""
    ref = parse_reference_params(_REF1)
    assert isinstance(ref, ReferenceParams)
    with pytest.raises(Exception):
        ref.name = "mutated"  # type: ignore[misc]


def test_strict_comparator_passes_on_matching_fields() -> None:
    """The strict comparator passes when all generated fields match."""
    ref = parse_reference_params(_REF1)
    result = compare_params_strict(_matching_generated_fields(ref), ref)
    assert result.ok
    assert all(result.checks.values())


def test_strict_comparator_flags_atom_type_change() -> None:
    """The strict comparator flags a changed atom type."""
    ref = parse_reference_params(_REF1)
    gen = _matching_generated_fields(ref)
    victim = next(iter(gen.atom_types))
    gen.atom_types[victim] = "BOGUS"
    result = compare_params_strict(gen, ref)
    assert not result.ok
    assert result.checks["atom_types"] is False


def test_strict_comparator_flags_removed_bond() -> None:
    """The strict comparator flags a removed bond."""
    ref = parse_reference_params(_REF1)
    gen = _matching_generated_fields(ref)
    pruned = set(gen.bond_keys)
    pruned.pop()
    gen = GeneratedFields(
        atom_types=gen.atom_types,
        bond_keys=frozenset(pruned),
        icoor_topology=gen.icoor_topology,
        nbr_atom=gen.nbr_atom,
        charges=gen.charges,
    )
    result = compare_params_strict(gen, ref)
    assert not result.ok
    assert result.checks["bonds"] is False


def test_strict_comparator_flags_icoor_topology_change() -> None:
    """The strict comparator flags an ICOOR topology change."""
    ref = parse_reference_params(_REF1)
    gen = _matching_generated_fields(ref)
    victim = next(iter(gen.icoor_topology))
    gen.icoor_topology[victim] = ("X", "Y", "Z")
    result = compare_params_strict(gen, ref)
    assert not result.ok
    assert result.checks["icoor_topology"] is False


def test_strict_comparator_flags_nbr_atom_change() -> None:
    """The strict comparator flags a changed NBR_ATOM."""
    ref = parse_reference_params(_REF1)
    gen = _matching_generated_fields(ref)
    gen = GeneratedFields(
        atom_types=gen.atom_types,
        bond_keys=gen.bond_keys,
        icoor_topology=gen.icoor_topology,
        nbr_atom="NOPE",
        charges=gen.charges,
    )
    result = compare_params_strict(gen, ref)
    assert not result.ok
    assert result.checks["nbr_atom"] is False


def test_strict_comparator_flags_charge_perturbation() -> None:
    """The strict comparator flags a charge perturbation beyond tolerance."""
    ref = parse_reference_params(_REF1)
    gen = _matching_generated_fields(ref)
    victim = next(iter(gen.charges))
    gen.charges[victim] = gen.charges[victim] + 0.2
    result = compare_params_strict(gen, ref, charge_tolerance=0.01)
    assert not result.ok
    assert result.checks["charges"] is False


# --- heavy-atom semantic (isomorphism) comparator ----------------------------


def _atom(name: str, atom_type: str) -> SimpleNamespace:
    """Build a minimal atom-like object with a name and atom type."""
    return SimpleNamespace(name=name, atom_type=atom_type)


def _stub_prep(atoms, bonds, charges) -> SimpleNamespace:
    """A minimal LigandPreparation-like object for equivalence comparison."""
    residue_type = SimpleNamespace(atoms=atoms, bonds=bonds)
    cartbonded = SimpleNamespace(
        length_parameters=(), angle_parameters=(), improper_parameters=()
    )
    return SimpleNamespace(
        residue_type=residue_type,
        partial_charges=charges,
        cartbonded_params=cartbonded,
    )


def _linear_c3(names) -> SimpleNamespace:
    """Build a linear three-carbon stub preparation from atom names."""
    a, b, c = names
    atoms = [_atom(a, "CR"), _atom(b, "CR"), _atom(c, "CR")]
    bonds = [(a, b, "SINGLE", False), (b, c, "SINGLE", False)]
    charges = {a: 0.0, b: 0.0, c: 0.0}
    return _stub_prep(atoms, bonds, charges)


def test_semantic_comparator_equates_renamed_copy() -> None:
    """The semantic comparator equates a heavy-atom renamed copy."""
    generated = _linear_c3(["C1", "C2", "C3"])
    # Same topology + types, heavy atoms renamed (still carbon).
    reference = _linear_c3(["C7", "C8", "C9"])
    result = compare_semantic(generated, reference)
    assert result.is_equivalent


def test_semantic_comparator_flags_type_change() -> None:
    """The semantic comparator flags an atom-type change."""
    generated = _linear_c3(["C1", "C2", "C3"])
    reference = _linear_c3(["C7", "C8", "C9"])
    reference.residue_type.atoms[1].atom_type = "Nad"  # break a type
    result = compare_semantic(generated, reference)
    assert not result.is_equivalent


def test_semantic_comparator_handles_pdb_carbon_name_collision() -> None:
    """The semantic comparator reads element from atom type, not the name."""
    # A carbon named "CA" (alpha carbon, from OpenBabel PDB-residue naming) must
    # not be read as the element calcium: the element comes from the atom_type
    # (CR -> carbon), so the heavy-atom graph still maps to a generic-named copy.
    generated = _linear_c3(["CA", "CB", "CG"])
    reference = _linear_c3(["C1", "C2", "C3"])
    result = compare_semantic(generated, reference)
    assert result.is_equivalent


def test_element_from_atom_type_prefixes() -> None:
    """Element inference from atom-type prefixes covers expected cases."""
    from tmol.ligand.equivalence import _element_from_atom_type

    cases = {
        "CS1": "C",
        "CR": "C",
        "CDp": "C",
        "Oat": "O",
        "Nad": "N",
        "Sth": "S",
        "Cl": "Cl",
        "ClR": "Cl",
        "Br": "Br",
        "BrR": "Br",
        "F": "F",
        "I": "I",
    }
    for atom_type, element in cases.items():
        assert _element_from_atom_type(atom_type) == element, atom_type
    assert _element_from_atom_type("") is None
