"""Tests for the shared Rosetta ``.params`` reference parser and charge sidecar."""

from __future__ import annotations

from pathlib import Path

import pytest

from tmol.ligand.params_reference import (
    ReferenceParams,
    as_legacy_dict,
    compare_charges,
    parse_reference_params,
    reference_charges,
)

_GROUND_TRUTH = Path(__file__).parent.parent / "data" / "ligand_ground_truth"
_REF1 = _GROUND_TRUTH / "ref1.params"
_REF2 = _GROUND_TRUTH / "ref2.params"


def _atom_record_count(path: Path) -> int:
    return sum(1 for line in path.open() if line.split()[:1] == ["ATOM"])


def test_charge_sidecar_length_matches_atom_records():
    ref = parse_reference_params(_REF1)
    # ref1 has 31 ATOM records (17 heavy + 14 H).
    assert len(ref.charges) == _atom_record_count(_REF1) == 31
    assert ref.heavy_atom_names() == {
        name for name, _t, _q in ref.atoms if not name.startswith("H")
    }
    assert len(ref.heavy_atom_names()) == 17


def test_parse_captures_name_and_nbr_and_charges_are_floats():
    ref = parse_reference_params(_REF1)
    assert ref.name == "ref1"
    assert ref.nbr_atom  # a real NBR_ATOM is present
    assert ref.has_hydrogen
    assert all(isinstance(q, float) for q in ref.charges.values())


def test_reference_charges_accepts_path_and_object():
    ref = parse_reference_params(_REF2)
    from_path = reference_charges(_REF2)
    from_obj = reference_charges(ref)
    assert from_path == from_obj == ref.charges


def test_all_bond_pairs_are_hydrogen_inclusive():
    ref = parse_reference_params(_REF1)
    pairs = ref.all_bond_pairs()
    # at least one bond must touch a hydrogen (mol2/params keep explicit H).
    assert any(any(n.startswith("H") for n in pair) for pair in pairs)


def test_legacy_dict_shape_is_preserved():
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


def test_compare_charges_identical_passes():
    charges = reference_charges(_REF1)
    ok, mismatches = compare_charges(charges, dict(charges), tolerance=0.01)
    assert ok
    assert mismatches == []


def test_compare_charges_perturbation_fails_beyond_tolerance():
    charges = reference_charges(_REF1)
    perturbed = dict(charges)
    victim = next(iter(perturbed))
    perturbed[victim] = perturbed[victim] + 0.2
    ok, mismatches = compare_charges(perturbed, charges, tolerance=0.05)
    assert not ok
    assert any(name == victim for name, *_ in mismatches)


def test_compare_charges_no_shared_atoms_is_not_ok():
    ok, mismatches = compare_charges({"X1": 0.1}, {"Y1": 0.1}, tolerance=0.05)
    assert not ok
    assert mismatches == []


def test_reference_params_is_frozen():
    ref = parse_reference_params(_REF1)
    assert isinstance(ref, ReferenceParams)
    with pytest.raises(Exception):
        ref.name = "mutated"  # type: ignore[misc]
