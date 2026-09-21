"""Shared helpers for the ligand-prep parity regression.

Not collected by pytest (underscore-prefixed). Provides a seed-preparation
helper and comparisons against Rosetta ``.params`` reference files used as
ground truth for chi and proton-chi equivalence.
"""

from __future__ import annotations

from types import SimpleNamespace

from tmol.tests.ligand._params_reference import BOND_TOKEN_TO_TYPE
from tmol.ligand import (
    prepare_single_ligand,
    nonstandard_residue_info_from_smiles_via_mol2,
    LigandPreparation,
)


def prepare_seed_entry(entry) -> LigandPreparation:
    """Prepare a ligand from a parity manifest entry via the SMILES->mol2 route.

    Feeds the entry's *protonated* SMILES (``expected_prot_smiles``) verbatim
    through the canonical SMILES->mol2->params path: OpenBabel builds a 3D
    MMFF94 mol2, which is read without an atom-array round-trip. Protonation is
    pinned (``protonate=False``) so the regression reproduces the ground-truth
    protonation state deterministically. Uses the entry's charge mode so the
    regression is genuinely driven by the manifest.
    """
    info = nonstandard_residue_info_from_smiles_via_mol2(
        entry.expected_prot_smiles, res_name=entry.name, protonate=False
    )
    return prepare_single_ligand(info)


def _chi_axes_from_reference(ref) -> set:
    """Return the unordered set of central CHI bond {b, c} pairs from a ref."""
    return {frozenset((quad[1], quad[2])) for _n, quad, _biaryl in ref.chis}


def _chi_axes_from_prep(prep) -> set:
    """Return the unordered set of central CHI bond {b, c} pairs from a prep."""
    return {frozenset((tor.b.atom, tor.c.atom)) for tor in prep.residue_type.torsions}


def chi_axes_equivalent(prep, ref, *, view=None) -> bool:
    """Compare prep vs reference CHI axes *name-agnostically*.

    The SMILES->mol2 prep keeps the OpenBabel mol2 atom names, while the Rosetta
    reference renames atoms (``mol2genparams --rename_atoms``), so axis bond
    pairs cannot be matched by raw name. Map the prep's heavy-atom names into
    the reference namespace via the same heavy-atom graph isomorphism used by
    :func:`compare_semantic`, then compare the resulting axis sets. Returns
    ``False`` if no isomorphism exists (e.g. a mutated reference graph).
    """
    from tmol.tests.ligand import _heavy_atom_name_mapping

    if view is None:
        view = reference_view_from_params(ref)
    mapping = _heavy_atom_name_mapping(prep.residue_type, view.residue_type)
    if mapping is None:
        return False
    mapped: set = set()
    for tor in prep.residue_type.torsions:
        b, c = tor.b.atom, tor.c.atom
        if b not in mapping or c not in mapping:
            return False
        mapped.add(frozenset((mapping[b], mapping[c])))
    return mapped == _chi_axes_from_reference(ref)


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


def reference_view_from_params(ref) -> SimpleNamespace:
    """Build a ``LigandPreparation``-like view from a parsed ``.params``.

    Suitable as the *reference* argument to
    :func:`tmol.tests.ligand._equivalence.compare_ligand_preparations`: it exposes
    ``residue_type.atoms`` / ``residue_type.bonds``, ``partial_charges``, and an
    empty ``cartbonded_params`` (cartbonded is skipped for the semantic check).
    """

    atoms = [SimpleNamespace(name=n, atom_type=t) for n, t, _q in ref.atoms]
    bonds = []
    for pair, order, ring in ref.bond_types:
        a, b = sorted(pair)
        label = BOND_TOKEN_TO_TYPE.get(str(order).upper(), "SINGLE")
        bonds.append((a, b, label, ring == "RING"))
    cart = SimpleNamespace(
        length_parameters=(), angle_parameters=(), improper_parameters=()
    )
    return SimpleNamespace(
        residue_type=SimpleNamespace(atoms=atoms, bonds=bonds),
        partial_charges=dict(ref.charges),
        cartbonded_params=cart,
    )
