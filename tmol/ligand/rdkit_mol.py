"""RDKit molecule construction and protonation for ligands."""

import logging
from typing import Optional

import biotite.structure as struc
from biotite.interface.rdkit import to_mol
from rdkit import Chem
from rdkit.Chem import AllChem

from tmol.ligand.detect import NonStandardResidueInfo, _strip_metals
from tmol.ligand.dimorphite_dl import protonate_mol_variants

logger = logging.getLogger(__name__)

_ROSETTA_TOPOLOGY_PROP = "_tmol_use_rosetta_topology"


# Map biotite BondType -> the RDKit bond order we want on the round-tripped
# Mol. biotite's ``to_mol`` collapses AROMATIC_DOUBLE → SINGLE + aromatic
# flag, which loses the double-bond information needed by RDKit's sanitize
# valence model — leading to all-single ring bonds for N-heterocycles. We
# restore the Kekulé bond orders by walking the source atom_array.
_BIOTITE_TO_RDKIT_BOND_ORDER = {
    int(struc.BondType.SINGLE): Chem.BondType.SINGLE,
    int(struc.BondType.DOUBLE): Chem.BondType.DOUBLE,
    int(struc.BondType.TRIPLE): Chem.BondType.TRIPLE,
    int(struc.BondType.QUADRUPLE): Chem.BondType.QUADRUPLE,
    int(struc.BondType.AROMATIC_SINGLE): Chem.BondType.SINGLE,
    int(struc.BondType.AROMATIC_DOUBLE): Chem.BondType.DOUBLE,
    int(struc.BondType.AROMATIC_TRIPLE): Chem.BondType.TRIPLE,
    int(struc.BondType.AROMATIC): Chem.BondType.AROMATIC,
}

_SOURCE_KEKULE_PROP = "_tmol_source_kekule"
_SOURCE_AROMATIC_PROP = "_tmol_source_aromatic"


def clear_source_chemistry_props(mol: Chem.Mol) -> None:
    """Drop source Kekulé/aromatic marker props so RDKit can re-perceive chemistry.

    Used after Dimorphite protonation when CIF/mol2 aromatic stamps no longer
    match the protonated graph (common for charged N-heterocycles).
    """
    for key in (_SOURCE_KEKULE_PROP, _SOURCE_AROMATIC_PROP):
        if mol.HasProp(key):
            mol.ClearProp(key)


def _rebuild_mol_via_smiles_preserving_coords(heavy: Chem.Mol) -> Optional[Chem.Mol]:
    """Rebuild heavy-atom topology from SMILES; copy formal charges and coordinates."""
    try:
        Chem.GetSSSR(heavy)
        smiles = Chem.MolToSmiles(heavy, isomericSmiles=True)
    except Exception:
        return None

    fresh = Chem.MolFromSmiles(smiles)
    if fresh is None:
        return None

    match = fresh.GetSubstructMatch(heavy)
    if len(match) != heavy.GetNumAtoms():
        return None

    conf = Chem.Conformer(fresh.GetNumAtoms())
    has_coords = heavy.GetNumConformers() > 0
    for proto_idx, fresh_idx in enumerate(match):
        if has_coords:
            conf.SetAtomPosition(
                fresh_idx, heavy.GetConformer().GetAtomPosition(proto_idx)
            )
        fresh.GetAtomWithIdx(fresh_idx).SetFormalCharge(
            heavy.GetAtomWithIdx(proto_idx).GetFormalCharge()
        )
    if has_coords:
        fresh.AddConformer(conf, assignId=True)
    else:
        AllChem.Compute2DCoords(fresh)

    Chem.SanitizeMol(fresh)
    clear_source_chemistry_props(fresh)
    return fresh


def prepare_input_protonation_mol_for_mmff94(mol: Chem.Mol) -> Chem.Mol:
    """Sanitize an input-protonation mol for MMFF without rebuilding hydrogens."""
    clear_source_chemistry_props(mol)
    clear_aromatic_perception_flags(mol)
    try:
        Chem.SanitizeMol(mol)
    except (
        Chem.rdchem.KekulizeException,
        Chem.rdchem.AtomKekulizeException,
        Chem.rdchem.AtomValenceException,
    ):
        logger.debug(
            "Sanitize after clearing aromatic flags failed; MMFF may retry",
            exc_info=True,
        )
    return mol


def normalize_protonated_mol_for_mmff94(mol: Chem.Mol) -> Chem.Mol:
    """Make a Dimorphite-protonated mol sanitizable and MMFF-ready."""
    clear_source_chemistry_props(mol)
    clear_aromatic_perception_flags(mol)
    try:
        Chem.SanitizeMol(mol)
        return mol
    except (
        Chem.rdchem.KekulizeException,
        Chem.rdchem.AtomKekulizeException,
        Chem.rdchem.AtomValenceException,
    ):
        logger.debug(
            "Sanitize failed after clearing aromatic flags; trying SMILES rebuild",
            exc_info=True,
        )

    heavy = Chem.RemoveHs(mol, sanitize=False)
    rebuilt = _rebuild_mol_via_smiles_preserving_coords(heavy)
    if rebuilt is None:
        return mol

    return Chem.AddHs(rebuilt, addCoords=True)


def clear_aromatic_perception_flags(mol: Chem.Mol) -> None:
    """Clear per-atom/bond aromatic flags (CIF ``tmol_aromatic`` stamps on the mol)."""
    for bond in mol.GetBonds():
        bond.SetIsAromatic(False)
        if bond.GetBondType() == Chem.BondType.AROMATIC:
            bond.SetBondType(Chem.BondType.SINGLE)
    for atom in mol.GetAtoms():
        atom.SetIsAromatic(False)


def _restore_kekule_bonds(mol: Chem.Mol, atom_array: struc.AtomArray) -> None:
    """Overwrite ``mol`` bond orders from the source biotite bond table.

    Sets the ``_SOURCE_KEKULE_PROP`` molecule property to ``"1"`` when
    the source carried explicit Kekulé bond orders for at least one
    ring bond — that flag drives the conditional Kekulé typing later.
    Mutates ``mol`` in place.
    """
    if atom_array.bonds is None:
        return
    saw_kekule = False
    # Only count biotite's AROMATIC_SINGLE / AROMATIC_DOUBLE — those mark
    # ring bonds whose source carried Kekulé orders. Plain SINGLE /
    # DOUBLE bonds appear for non-ring chain edges in every molecule and
    # would falsely trigger Kekulé typing.
    kekule_orders = {
        int(struc.BondType.AROMATIC_SINGLE),
        int(struc.BondType.AROMATIC_DOUBLE),
        int(struc.BondType.AROMATIC_TRIPLE),
    }
    for a, b, raw_type in atom_array.bonds.as_array():
        rdkit_type = _BIOTITE_TO_RDKIT_BOND_ORDER.get(int(raw_type))
        if rdkit_type is None:
            continue
        bond = mol.GetBondBetweenAtoms(int(a), int(b))
        if bond is None:
            continue
        bond.SetBondType(rdkit_type)
        if int(raw_type) in kekule_orders:
            saw_kekule = True
    if saw_kekule:
        mol.SetProp(_SOURCE_KEKULE_PROP, "1")


_SOURCE_SUBTYPE_PROP = "_tmol_source_subtype"


def _apply_source_subtypes(mol: Chem.Mol, atom_array: struc.AtomArray) -> None:
    """Stamp source subtype tags on atoms before H removal.

    The source AtomArray and pre-RemoveHs RDKit mol share atom indices,
    so this is the most reliable point to transfer subtype hints.
    """
    if not hasattr(atom_array, "tmol_source_subtype"):
        return
    subtypes = atom_array.tmol_source_subtype
    for idx, atom in enumerate(mol.GetAtoms()):
        if idx >= len(subtypes):
            break
        sub = str(subtypes[idx])
        if sub and sub != "?":
            atom.SetProp(_SOURCE_SUBTYPE_PROP, sub)


def _kekulize_non_ring_aromatic_bonds(mol: Chem.Mol) -> None:
    """De-aromatize non-ring aromatic bonds to explicit singles.

    Aromatic semantics are ring-based in RDKit/biotite. Non-ring aromatic
    bonds are treated as delocalization placeholders and must be explicit
    non-aromatic bonds for robust downstream handling.
    """
    changed = False
    for bond in mol.GetBonds():
        if bond.GetIsAromatic() and not bond.IsInRing():
            bond.SetIsAromatic(False)
            bond.SetBondType(Chem.BondType.SINGLE)
            changed = True
    if not changed:
        return
    for atom in mol.GetAtoms():
        if atom.GetIsAromatic():
            if not any(b.GetIsAromatic() for b in atom.GetBonds()):
                atom.SetIsAromatic(False)


def normalize_non_ring_aromatic_bonds(mol: Chem.Mol) -> None:
    """Normalize non-ring aromatic placeholders before RDKit sanitize."""
    _kekulize_non_ring_aromatic_bonds(mol)


def _ligand_arr_indices(
    atom_array: struc.AtomArray, *, keep_hydrogens: bool
) -> list[int]:
    """Map RDKit atom indices to AtomArray indices for annotation transfer."""
    if keep_hydrogens:
        return list(range(len(atom_array)))
    return [i for i, e in enumerate(atom_array.element) if str(e) != "H"]


def _apply_atom_array_annotations(
    mol: Chem.Mol, atom_array: struc.AtomArray, arr_indices: list[int]
) -> None:
    """Apply source CIF/mol2 annotations onto ``mol`` at ``arr_indices``.

    Each ``arr_indices[i]`` is the AtomArray index for RDKit atom index ``i``.
    After ``RemoveHs``, ``arr_indices`` lists only heavy-atom source indices;
    when explicit hydrogens are kept, it is ``range(n_atoms)``.
    """
    if mol.GetNumAtoms() < len(arr_indices):
        return

    saw_subtype = False
    if hasattr(atom_array, "tmol_source_subtype"):
        subtypes = atom_array.tmol_source_subtype
        for mol_idx, arr_idx in enumerate(arr_indices):
            if arr_idx >= len(subtypes):
                continue
            sub = str(subtypes[arr_idx])
            if sub and sub != "?":
                mol.GetAtomWithIdx(mol_idx).SetProp(_SOURCE_SUBTYPE_PROP, sub)
                saw_subtype = True
    if saw_subtype:
        mol.SetProp(_ROSETTA_TOPOLOGY_PROP, "1")

    if not hasattr(atom_array, "tmol_aromatic"):
        return
    flags = atom_array.tmol_aromatic
    for mol_idx, arr_idx in enumerate(arr_indices):
        a = mol.GetAtomWithIdx(mol_idx)
        a.SetIsAromatic(bool(flags[arr_idx]))
    for bond in mol.GetBonds():
        if bond.GetBeginAtom().GetIsAromatic() and bond.GetEndAtom().GetIsAromatic():
            bond.SetIsAromatic(True)
        else:
            bond.SetIsAromatic(False)
    mol.SetProp(_SOURCE_AROMATIC_PROP, "1")


def _apply_atom_aromatic_flags_post_removeh(
    mol: Chem.Mol, atom_array: struc.AtomArray, heavy_arr_indices: list[int]
) -> None:
    """Re-stamp source annotations after ``RemoveHs`` (see :func:`_apply_atom_array_annotations`)."""
    _apply_atom_array_annotations(mol, atom_array, heavy_arr_indices)


def source_subtype(atom: Chem.Atom) -> str:
    """Return the source mol2 atom-type subtype tag (e.g. ``ar``, ``2``,
    ``cat``, ``pl3``, ``3``) when known, else ``""``."""
    if atom.HasProp(_SOURCE_SUBTYPE_PROP):
        return atom.GetProp(_SOURCE_SUBTYPE_PROP).strip()
    return ""


def reapply_ligand_source_annotations(
    mol: Chem.Mol,
    ligand_info: NonStandardResidueInfo,
    *,
    keep_hydrogens: bool | None = None,
) -> None:
    """Re-stamp source bond orders and per-atom hints after RDKit rebuilds.

    Dimorphite protonation and ``AssignBondOrdersFromTemplate`` can return a
    fresh Mol that drops ``_tmol_source_subtype`` props and Kekulé bond orders.
    Re-apply from the original AtomArray before atom typing.
    """
    atom_array = ligand_info.atom_array
    if keep_hydrogens is None:
        keep_hydrogens = ligand_info.skip_protonation
    arr_indices = _ligand_arr_indices(atom_array, keep_hydrogens=keep_hydrogens)
    _restore_kekule_bonds(mol, atom_array)
    normalize_non_ring_aromatic_bonds(mol)
    _apply_atom_array_annotations(mol, atom_array, arr_indices)


def source_carried_kekule(mol: Chem.Mol) -> bool:
    """True iff the source molecule was constructed with Kekulé bond orders.

    Set by :func:`_restore_kekule_bonds` when the input AtomArray carried
    explicit ``SINGLE`` / ``DOUBLE`` (or biotite's ``AROMATIC_SINGLE`` /
    ``AROMATIC_DOUBLE``) ring bonds — typical for mol2 files written
    with ``C.2`` (sp2). SMILES inputs come through with only
    ``AROMATIC`` bonds and leave this flag unset.
    """
    return mol.HasProp(_SOURCE_KEKULE_PROP) and mol.GetProp(_SOURCE_KEKULE_PROP) == "1"


def source_has_aromatic_annotations(mol: Chem.Mol) -> bool:
    """True iff aromatic atom flags were provided by the source input."""
    return (
        mol.HasProp(_SOURCE_AROMATIC_PROP) and mol.GetProp(_SOURCE_AROMATIC_PROP) == "1"
    )


ELEMENT_TO_ATOMIC_NUM = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Na": 11,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "K": 19,
    "Br": 35,
    "I": 53,
}


def _remove_hs_tolerant(mol: Chem.Mol) -> Chem.Mol:
    """Remove explicit hydrogens, retrying without sanitize on failure.

    Kekulization can fail mid-pipeline for ligands with formal-charge
    nitrogens or unusual ring patterns. Falling back to ``sanitize=False``
    preserves the bond orders we already set (e.g. by
    :func:`_restore_kekule_bonds`); running ``sanitize_tolerant`` here
    instead silently rewrites DOUBLE bonds back to SINGLE via the cleanup
    pass.
    """
    try:
        return Chem.RemoveHs(mol)
    except (
        Chem.rdchem.KekulizeException,
        Chem.rdchem.AtomKekulizeException,
        Chem.rdchem.AtomValenceException,
    ):
        return Chem.RemoveHs(mol, sanitize=False)


def ligand_atom_array_to_rdkit_mol(
    ligand_info: NonStandardResidueInfo,
    *,
    keep_hydrogens: bool = False,
) -> Chem.Mol:
    """Build an RDKit Mol directly from a ligand AtomArray.

    Args:
        keep_hydrogens: When True, retain explicit hydrogens from the input
            (used for ``skip_protonation`` — preserve mol2/CIF protonation).
    """
    atom_array = ligand_info.atom_array
    has_bonds = atom_array.bonds is not None and atom_array.bonds.get_bond_count() > 0
    if len(atom_array) == 0:
        raise ValueError(f"{ligand_info.res_name}: empty atom array")
    if not has_bonds:
        raise ValueError(
            f"{ligand_info.res_name}: ligand bond inference is unsupported. "
            "Input must provide explicit bond orders (CIF with "
            "_chem_comp_bond.value_order / aromatic annotations). "
            "PDB/topology-only ligand chemistry is not supported."
        )

    raw_types = [int(t) for _, _, t in atom_array.bonds.as_array()]
    unsupported = sorted(
        set(t for t in raw_types if t not in _BIOTITE_TO_RDKIT_BOND_ORDER)
    )
    if unsupported:
        logger.warning(
            "%s: unsupported bond type codes %s in ligand input; "
            "preserving original to_mol bond typing for those edges.",
            ligand_info.res_name,
            unsupported,
        )

    chemistry_orders = {
        int(struc.BondType.DOUBLE),
        int(struc.BondType.TRIPLE),
        int(struc.BondType.QUADRUPLE),
        int(struc.BondType.AROMATIC),
        int(struc.BondType.AROMATIC_SINGLE),
        int(struc.BondType.AROMATIC_DOUBLE),
        int(struc.BondType.AROMATIC_TRIPLE),
    }
    has_chemistry_order_signal = any(t in chemistry_orders for t in raw_types)
    has_custom_aromatic_flags = hasattr(atom_array, "tmol_aromatic")
    if not has_chemistry_order_signal and not has_custom_aromatic_flags:
        raise ValueError(
            f"{ligand_info.res_name}: ligand has topology-only SINGLE bonds with no "
            "chemistry-level bond-order/aromatic annotations. "
            "PDB ligand chemistry inference is unsupported; provide ligand as CIF "
            "with explicit bond orders."
        )

    try:
        mol = to_mol(atom_array)
    except Exception as exc:
        raise ValueError(
            f"{ligand_info.res_name}: failed to read explicit ligand bond chemistry "
            f"from input ({exc}). Provide a CIF with explicit bond orders."
        ) from exc
    _restore_kekule_bonds(mol, atom_array)
    normalize_non_ring_aromatic_bonds(mol)
    _apply_source_subtypes(mol, atom_array)

    if keep_hydrogens:
        arr_indices = _ligand_arr_indices(atom_array, keep_hydrogens=True)
    else:
        mol = _remove_hs_tolerant(mol)
        arr_indices = _ligand_arr_indices(atom_array, keep_hydrogens=False)

    _apply_atom_array_annotations(mol, atom_array, arr_indices)
    mol = _strip_metals(mol)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"{ligand_info.res_name}: failed to build RDKit Mol")
    return mol


def protonate_ligand_mol(
    mol: Chem.Mol,
    ph: float = 7.4,
    precision: float = 0.1,
) -> Chem.Mol:
    """Protonate an RDKit Mol at a target pH and return first variant.

    Dimorphite-DL sometimes rebuilds the molecule from SMILES, which drops the
    3D conformer and can yield an invalid valence. Such a result is untrusted:
    when the input carried coordinates and the variant lost them, we keep the
    input protonation (already chemically valid and placed in 3D).
    """
    try:
        variants = protonate_mol_variants(
            mol,
            min_ph=ph,
            max_ph=ph,
            pka_precision=precision,
            max_variants=128,
            silent=True,
        )
        if variants:
            variant = variants[0]
            if mol.GetNumConformers() and not variant.GetNumConformers():
                return mol
            return variant
    except Exception:
        logger.warning(
            "Dimorphite-DL direct-Mol protonation failed; keeping input mol",
            exc_info=True,
        )
    return mol
