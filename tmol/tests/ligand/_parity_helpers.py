"""Shared helpers for the ligand-prep parity regression.

Not collected by pytest (underscore-prefixed). Provides a seed-preparation
helper and a serialization round-trip check (write a prep as both a Rosetta
``.params`` and a tmol ``.tmol``, read both back, and compare the fields the
two formats share).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

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


def prepare_seed_view(entry):
    """Prepare a seed ligand from SMILES via the validated regression path.

    Mirrors the primitives the ground-truth regression uses (protonate ->
    add hydrogens -> MMFF94 charges -> atom typing -> residue build), which
    reproduce the reference charges within tolerance. Returns a
    ``LigandPreparation``-like view (restype + charges + empty cartbonded).

    This intentionally does not route through ``prepare_single_ligand``: that
    path currently diverges from the reference on the symmetric aromatic
    charges of some molecules (a known SMILES-path charge difference, tracked
    as a queued finding), which the validated primitives do not.
    """
    from rdkit import Chem

    from tmol.ligand.atom_typing import assign_tmol_atom_types
    from tmol.ligand.mol3d import compute_mmff94_charges
    from tmol.ligand.rdkit_mol import protonate_ligand_mol
    from tmol.ligand.residue_builder import build_residue_type

    mol = Chem.MolFromSmiles(entry.input_smiles)
    protonated = protonate_ligand_mol(mol, ph=7.4)
    protonated = Chem.AddHs(protonated, addCoords=False)
    charges_by_idx = compute_mmff94_charges(protonated)
    atom_types, state = assign_tmol_atom_types(protonated, return_state=True)
    charges = {
        at.atom_name: charges_by_idx[at.index]
        for at in atom_types
        if at.index in charges_by_idx
    }
    restype = build_residue_type(
        protonated,
        entry.name,
        atom_types,
        typing_state=state,
        sample_proton_chi=entry.sample_proton_chi,
    )
    cart = SimpleNamespace(
        length_parameters=(), angle_parameters=(), improper_parameters=()
    )
    return SimpleNamespace(
        residue_type=restype, partial_charges=charges, cartbonded_params=cart
    )


def _chi_axes_from_reference(ref) -> set:
    """Return the unordered set of central CHI bond {b, c} pairs from a ref."""
    return {frozenset((quad[1], quad[2])) for _n, quad, _biaryl in ref.chis}


def _chi_axes_from_prep(prep) -> set:
    """Return the unordered set of central CHI bond {b, c} pairs from a prep."""
    return {frozenset((tor.b.atom, tor.c.atom)) for tor in prep.residue_type.torsions}


def proton_chi_by_axis_from_reference(ref) -> dict:
    """Map each CHI axis to its ``(sorted samples, expansions)`` from a ref.

    Reads ``PROTON_CHI n SAMPLES k v.. [EXTRA m e..]`` lines and resolves the
    CHI number ``n`` to the central ``{b, c}`` bond of the matching CHI record.
    """
    axis_by_num = {num: frozenset((quad[1], quad[2])) for num, quad, _b in ref.chis}
    out: dict = {}
    for line in ref.proton_chis:
        toks = line.split()
        num = int(toks[1])
        si = toks.index("SAMPLES")
        k = int(toks[si + 1])
        samples = tuple(sorted(float(x) for x in toks[si + 2 : si + 2 + k]))
        expansions: tuple = ()
        if "EXTRA" in toks:
            ei = toks.index("EXTRA")
            m = int(toks[ei + 1])
            expansions = tuple(float(x) for x in toks[ei + 2 : ei + 2 + m])
        out[axis_by_num[num]] = (samples, expansions)
    return out


def proton_chi_by_axis_from_prep(prep) -> dict:
    """Map each CHI axis to its ``(sorted samples, expansions)`` from a prep."""
    axis_by_name = {
        tor.name: frozenset((tor.b.atom, tor.c.atom))
        for tor in prep.residue_type.torsions
    }
    return {
        axis_by_name[cs.chi_dihedral]: (
            tuple(sorted(cs.samples)),
            tuple(cs.expansions),
        )
        for cs in prep.residue_type.chi_samples
    }


def reference_view_from_params(ref):
    """Build a ``LigandPreparation``-like view from a parsed ``.params``.

    Suitable as the *reference* argument to
    :func:`tmol.ligand.equivalence.compare_ligand_preparations`: it exposes
    ``residue_type.atoms`` / ``residue_type.bonds``, ``partial_charges``, and an
    empty ``cartbonded_params`` (cartbonded is skipped for the semantic check).
    """
    from tmol.ligand.params_io import _BOND_TOK_TO_TYPE

    atoms = [SimpleNamespace(name=n, atom_type=t) for n, t, _q in ref.atoms]
    bonds = []
    for pair, order, ring in ref.bond_types:
        a, b = sorted(pair)
        label = _BOND_TOK_TO_TYPE.get(str(order).upper(), "SINGLE")
        bonds.append((a, b, label, ring == "RING"))
    cart = SimpleNamespace(
        length_parameters=(), angle_parameters=(), improper_parameters=()
    )
    return SimpleNamespace(
        residue_type=SimpleNamespace(atoms=atoms, bonds=bonds),
        partial_charges=dict(ref.charges),
        cartbonded_params=cart,
    )


@dataclass
class RoundtripResult:
    """Outcome of the ``.params`` / ``.tmol`` serialization consistency check."""

    strict: StrictComparison
    params_chi_axes: set
    tmol_chi_axes: set
    chi_axes_match: bool
    proton_chi_match: bool

    @property
    def ok(self) -> bool:
        return self.strict.ok and self.chi_axes_match and self.proton_chi_match


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
    CHI axes, and ``PROTON_CHI`` samples/expansions by axis. Cartbonded params
    and numeric ``NBR_RADIUS`` are excluded (no Rosetta counterpart / tmol
    hard-codes the radius).
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
    params_proton = proton_chi_by_axis_from_reference(params_ref)
    tmol_proton = proton_chi_by_axis_from_prep(tmol_prep)
    return RoundtripResult(
        strict=strict,
        params_chi_axes=params_axes,
        tmol_chi_axes=tmol_axes,
        chi_axes_match=params_axes == tmol_axes,
        proton_chi_match=params_proton == tmol_proton,
    )
