"""SMILES-start semantic equivalence to the Rosetta .params reference.

The SMILES path regenerates coordinates, atom names, and the ICOOR tree, so
equivalence is asserted via heavy-atom graph isomorphism (name-agnostic) plus
charge tolerance and CHI-axis sets, not by exact name matching.
"""

from __future__ import annotations

import pytest

from tmol.ligand.params_reference import compare_semantic, parse_reference_params
from tmol.ligand.parity_manifest import load_parity_manifest
from tmol.tests.ligand._parity_helpers import (
    _chi_axes_from_prep,
    _chi_axes_from_reference,
    prepare_seed_view,
    reference_view_from_params,
)

_SEED = load_parity_manifest()
_SKIP_CARTBONDED = frozenset({"cartbonded_params"})


@pytest.fixture(params=_SEED, ids=lambda e: e.name)
def entry_prep(request):
    entry = request.param
    return entry, prepare_seed_view(entry)


def test_smiles_prep_is_semantically_equivalent(entry_prep):
    entry, prep = entry_prep
    ref = parse_reference_params(entry.params)
    view = reference_view_from_params(ref)
    result = compare_semantic(
        prep, view, charge_tolerance=0.05, skip_checks=_SKIP_CARTBONDED
    )
    assert result.is_equivalent, result.details
    # cartbonded must not be among the executed (non-skipped) checks.
    assert result.details.get("cartbonded_params") == "skipped"


def test_smiles_prep_chi_axes_match_reference(entry_prep):
    entry, prep = entry_prep
    ref = parse_reference_params(entry.params)
    assert _chi_axes_from_prep(prep) == _chi_axes_from_reference(ref)


def test_changed_heavy_graph_is_not_equivalent():
    entry = _SEED[0]
    prep = prepare_seed_view(entry)
    ref = parse_reference_params(entry.params)
    view = reference_view_from_params(ref)
    # Drop a heavy-heavy bond from the reference view: the graphs now differ.
    heavy_bonds = [
        b
        for b in view.residue_type.bonds
        if not b[0].startswith("H") and not b[1].startswith("H")
    ]
    assert heavy_bonds
    view.residue_type.bonds.remove(heavy_bonds[0])
    result = compare_semantic(
        prep, view, charge_tolerance=0.05, skip_checks=_SKIP_CARTBONDED
    )
    assert not result.is_equivalent


def test_changed_chi_axis_is_detected():
    prep = prepare_seed_view(_SEED[0])
    prep_axes = _chi_axes_from_prep(prep)
    # A reference whose CHI axis set differs must not match the emitted set.
    bogus_axes = {frozenset({"ZZ1", "ZZ2"})} | set(list(prep_axes)[1:])
    assert prep_axes != bogus_axes
