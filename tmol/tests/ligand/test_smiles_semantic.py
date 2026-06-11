"""SMILES-start semantic equivalence to the Rosetta .params reference.

The SMILES->mol2 path regenerates coordinates, atom names, and the ICOOR tree,
so equivalence is asserted via heavy-atom graph isomorphism (name-agnostic) plus
charge tolerance and CHI-axis sets, not by exact name matching. The generated
side uses the canonical SMILES->mol2->params entry point (OpenBabel builds a 3D
MMFF94 mol2, read without an atom-array round-trip) via ``prepare_seed_entry``,
so this regression validates the real pipeline. Because MMFF94 charges are
topological (graph-determined), the charges match the reference even for
fused-ring aromatics such as ref2.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from tmol.ligand.params_reference import compare_semantic, parse_reference_params
from tmol.ligand.parity_manifest import load_parity_manifest
from tmol.tests.ligand._parity_helpers import (
    chi_axes_equivalent,
    prepare_seed_entry,
    reference_view_from_params,
)

_SEED = load_parity_manifest()


def _semantic_match(prep, ref, *, charge_tolerance: float = 0.05, skip_charges=False):
    """Production-path semantic equivalence plus CHI-axis equality.

    Returns ``(ok, result)`` where ``ok`` is the combined verdict and ``result``
    is the underlying ``EquivalenceResult``. ``cartbonded_params`` is always
    skipped (geometry-derived, no Rosetta counterpart); charges may be skipped
    to assert structural equivalence independently.
    """
    skip = {"cartbonded_params"}
    if skip_charges:
        skip.add("partial_charges")
    view = reference_view_from_params(ref)
    result = compare_semantic(
        prep, view, charge_tolerance=charge_tolerance, skip_checks=frozenset(skip)
    )
    chi_ok = chi_axes_equivalent(prep, ref, view=view)
    return (result.is_equivalent and chi_ok), result


@pytest.fixture(params=_SEED, ids=lambda e: e.name)
def entry_prep(request):
    entry = request.param
    return entry, prepare_seed_entry(entry)


def test_smiles_prep_structural_equivalence(entry_prep):
    # atom set / types / bonds + CHI axes via the production path, charges aside.
    entry, prep = entry_prep
    ref = parse_reference_params(entry.params)
    ok, result = _semantic_match(prep, ref, skip_charges=True)
    assert ok, result.details
    assert result.details.get("cartbonded_params") == "skipped"


@pytest.mark.parametrize("entry", _SEED, ids=lambda e: e.name)
def test_smiles_prep_charge_equivalence(entry):
    # The SMILES->mol2 path carries OpenBabel's topological MMFF94 charges
    # straight through, so charges match the reference for every seed
    # (including fused-ring ref2, which the old atom-array path mis-charged).
    prep = prepare_seed_entry(entry)
    ref = parse_reference_params(entry.params)
    ok, result = _semantic_match(prep, ref, skip_charges=False)
    assert ok, result.details


def test_changed_heavy_graph_is_detected():
    prep = prepare_seed_entry(_SEED[0])
    ref = parse_reference_params(_SEED[0].params)
    heavy_bonds = [
        b for b in ref.bond_types if not any(n.startswith("H") for n in b[0])
    ]
    assert heavy_bonds
    mutated = replace(ref, bond_types=frozenset(set(ref.bond_types) - {heavy_bonds[0]}))
    ok, _ = _semantic_match(prep, mutated, skip_charges=True)
    assert not ok


def test_changed_chi_axis_is_detected():
    prep = prepare_seed_entry(_SEED[0])
    ref = parse_reference_params(_SEED[0].params)
    assert ref.chis, "ref1 must carry CHI records for this negative"
    chis = list(ref.chis)
    num, _quad, biaryl = chis[0]
    chis[0] = (num, ("ZZ0", "ZZ1", "ZZ2", "ZZ3"), biaryl)
    mutated = replace(ref, chis=tuple(chis))
    ok, _ = _semantic_match(prep, mutated, skip_charges=True)
    assert not ok  # the mutated CHI axis no longer matches the emitted set
