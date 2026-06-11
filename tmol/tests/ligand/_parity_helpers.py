"""Shared helpers for the ligand-prep parity regression.

Not collected by pytest (underscore-prefixed). Provides a seed-preparation
helper and a serialization round-trip check (write a prep as both a Rosetta
``.params`` and a tmol ``.tmol``, read both back, and compare the fields the
two formats share).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tmol.ligand import params_file, params_io, prepare_single_ligand
from tmol.ligand.detect import nonstandard_residue_info_from_smiles
from tmol.ligand.params_reference import (
    StrictComparison,
    compare_params_strict,
    generated_fields_from_preparation,
    parse_reference_params,
)


def prepare_seed_entry(entry):
    """Prepare a ligand from a parity manifest entry's SMILES.

    Uses the entry's charge mode and proton-chi setting so the regression is
    genuinely driven by the manifest.
    """
    info = nonstandard_residue_info_from_smiles(entry.input_smiles, res_name=entry.name)
    return prepare_single_ligand(
        info,
        charge_mode=entry.charge_mode,
        sample_proton_chi=entry.sample_proton_chi,
    )


def _chi_axes_from_reference(ref) -> set:
    """Return the unordered set of central CHI bond {b, c} pairs from a ref."""
    return {frozenset((quad[1], quad[2])) for _n, quad, _biaryl in ref.chis}


def _chi_axes_from_prep(prep) -> set:
    """Return the unordered set of central CHI bond {b, c} pairs from a prep."""
    return {frozenset((tor.b.atom, tor.c.atom)) for tor in prep.residue_type.torsions}


@dataclass
class RoundtripResult:
    """Outcome of the ``.params`` / ``.tmol`` serialization consistency check."""

    strict: StrictComparison
    params_chi_axes: set
    tmol_chi_axes: set
    chi_axes_match: bool

    @property
    def ok(self) -> bool:
        return self.strict.ok and self.chi_axes_match


def write_both_formats(prep, out_dir, *, params_charges=None):
    """Write ``prep`` as a Rosetta ``.params`` and a tmol ``.tmol``.

    ``params_charges`` overrides the charges written to the ``.params`` file
    only (used to construct charge-mismatch negatives); the ``.tmol`` always
    uses the preparation's own charges.
    """
    out_dir = Path(out_dir)
    rt = prep.residue_type
    params_path = out_dir / "rt.params"
    tmol_path = out_dir / "rt.tmol"
    params_io.write_params_file(rt, params_path, params_charges or prep.partial_charges)
    params_file.write_params_file(
        tmol_path,
        [rt],
        {rt.name: prep.partial_charges},
        {rt.name: prep.cartbonded_params},
    )
    return params_path, tmol_path


def roundtrip_overlapping_fields(
    prep, out_dir, *, charge_tolerance: float = 0.01, params_charges=None
) -> RoundtripResult:
    """Write a prep to both formats, read back, and compare overlapping fields.

    Overlapping fields compared: atoms, atom types, all-atom bonds, ICOOR
    topology, neighbour atom, partial charges (within ``charge_tolerance``),
    and CHI axes. Cartbonded params and numeric ``NBR_RADIUS`` are excluded
    (no Rosetta counterpart / tmol hard-codes the radius).
    """
    params_path, tmol_path = write_both_formats(
        prep, out_dir, params_charges=params_charges
    )
    params_ref = parse_reference_params(params_path)
    tmol_prep = params_file.load_params_file(tmol_path)[0]
    gen = generated_fields_from_preparation(tmol_prep)
    strict = compare_params_strict(gen, params_ref, charge_tolerance=charge_tolerance)
    params_axes = _chi_axes_from_reference(params_ref)
    tmol_axes = _chi_axes_from_prep(tmol_prep)
    return RoundtripResult(
        strict=strict,
        params_chi_axes=params_axes,
        tmol_chi_axes=tmol_axes,
        chi_axes_match=params_axes == tmol_axes,
    )
