"""tmol .tmol <-> Rosetta .params serialization consistency regression."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tmol.ligand.parity_manifest import load_parity_manifest
from tmol.tests.ligand._parity_helpers import (
    prepare_seed_entry,
    proton_chi_by_axis_from_prep,
    proton_chi_by_axis_from_reference,
    roundtrip_overlapping_fields,
    write_both_formats,
)
from tmol.ligand import params_file
from tmol.ligand.params_reference import (
    compare_params_strict,
    generated_fields_from_preparation,
    parse_reference_params,
)

_SEED = load_parity_manifest()


@pytest.fixture(params=_SEED, ids=lambda e: e.name)
def seed_prep(request):
    return prepare_seed_entry(request.param)


def test_params_tmol_overlapping_fields_agree(seed_prep, tmp_path):
    result = roundtrip_overlapping_fields(seed_prep, tmp_path)
    assert result.strict.ok, result.strict.details
    assert result.chi_axes_match, (
        result.params_chi_axes,
        result.tmol_chi_axes,
    )
    assert result.ok


def test_charge_perturbation_breaks_consistency(tmp_path):
    prep = prepare_seed_entry(_SEED[0])
    perturbed = dict(prep.partial_charges)
    victim = next(iter(perturbed))
    perturbed[victim] = perturbed[victim] + 0.2

    params_path, tmol_path = write_both_formats(
        prep, tmp_path, params_charges=perturbed
    )
    params_ref = parse_reference_params(params_path)
    tmol_prep = params_file.load_params_file(tmol_path)[0]
    gen = generated_fields_from_preparation(tmol_prep)
    strict = compare_params_strict(gen, params_ref, charge_tolerance=0.01)

    assert not strict.ok
    assert strict.checks["charges"] is False


def test_proton_chi_sample_corruption_breaks_consistency(tmp_path):
    # ref1 carries PROTON_CHI; corrupting a sample value in the .params must be
    # caught by the proton-chi comparison (it was previously missed).
    prep = prepare_seed_entry(_SEED[0])
    params_path, tmol_path = write_both_formats(prep, tmp_path)

    lines = params_path.read_text().splitlines()
    corrupted = False
    for i, line in enumerate(lines):
        if line.startswith("PROTON_CHI") and "SAMPLES" in line:
            toks = line.split()
            si = toks.index("SAMPLES")
            toks[si + 2] = "999.0"  # corrupt the first sample value
            lines[i] = " ".join(toks)
            corrupted = True
            break
    assert corrupted, "expected a PROTON_CHI record in ref1 params"
    params_path.write_text("\n".join(lines) + "\n")

    params_ref = parse_reference_params(params_path)
    tmol_prep = params_file.load_params_file(tmol_path)[0]
    assert proton_chi_by_axis_from_reference(params_ref) != (
        proton_chi_by_axis_from_prep(tmol_prep)
    )


def test_sample_proton_chi_setting_drives_emission():
    # A manifest entry's sample_proton_chi must actually change preparation:
    # on -> proton-chi samples emitted, off -> none.
    base = _SEED[0]
    prep_on = prepare_seed_entry(replace(base, sample_proton_chi=True))
    prep_off = prepare_seed_entry(replace(base, sample_proton_chi=False))
    assert len(prep_on.residue_type.chi_samples) > 0
    assert len(prep_off.residue_type.chi_samples) == 0
