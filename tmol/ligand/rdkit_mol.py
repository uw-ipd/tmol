"""RDKit molecule construction for ligands.

Builds an RDKit ``Mol`` from a ligand ``AtomArray`` while preserving the
source's explicit bond orders and aromatic/subtype annotations. Protonation
and partial-charge generation are handled upstream by the SMILES -> OpenBabel
mol2 step (:func:`tmol.ligand.detect.nonstandard_residue_info_from_smiles_via_mol2`),
so this module does not protonate or recompute chemistry.
"""

import logging

import biotite.structure as struc
from biotite.interface.rdkit import to_mol
from rdkit import Chem

from tmol.ligand.detect import NonStandardResidueInfo, _strip_metals

logger = logging.getLogger(__name__)


# Map biotite BondType -> the RDKit bond order we want
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


def normalize_cumulated_azide(mol: Chem.Mol) -> Chem.Mol:
    """Strip a spurious H from a charge-separated azide/diazo terminus.

    Convert N=N=N-H (which RDKit does not understand) to =[N+]=[N-]
    Do nothing if this group is not found.
    """
    h_idx = [
        h.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() == 7
        and atom.GetFormalCharge() == -1
        and any(b.GetBondType() == Chem.BondType.DOUBLE for b in atom.GetBonds())
        for h in atom.GetNeighbors()
        if h.GetAtomicNum() == 1
    ]
    if not h_idx:
        return mol
    rw = Chem.RWMol(mol)
    for i in sorted(set(h_idx), reverse=True):
        rw.RemoveAtom(i)
    return rw.GetMol()


def _apply_atom_array_annotations(
    mol: Chem.Mol, atom_array: struc.AtomArray, arr_indices: list[int]
) -> None:
    """Apply source CIF/mol2 annotations onto ``mol`` at ``arr_indices``.

    Each ``arr_indices[i]`` is the AtomArray index for RDKit atom index ``i``.
    After ``RemoveHs``, ``arr_indices`` lists only heavy-atom source indices;
    when explicit hydrogens are kept, it is ``range(n_atoms)``.
    """
    if mol.GetNumAtoms() != len(arr_indices):
        return

    if hasattr(atom_array, "tmol_source_subtype"):
        subtypes = atom_array.tmol_source_subtype
        for mol_idx, arr_idx in enumerate(arr_indices):
            if arr_idx >= len(subtypes):
                continue
            sub = str(subtypes[arr_idx])
            if sub and sub != "?":
                mol.GetAtomWithIdx(mol_idx).SetProp(_SOURCE_SUBTYPE_PROP, sub)

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


def source_subtype(atom: Chem.Atom) -> str:
    """Return the source mol2 atom-type subtype tag (e.g. ``ar``, ``2``,
    ``cat``, ``pl3``, ``3``) when known, else ``""``."""
    if atom.HasProp(_SOURCE_SUBTYPE_PROP):
        return atom.GetProp(_SOURCE_SUBTYPE_PROP)
    return ""


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


def _normalize_nitro(mol: Chem.Mol) -> None:
    """Rewrite pentavalent nitro N(=O)=O to [N+](=O)[O-].

    Some inputs draw nitro with two N=O double bonds (valence-5 neutral N), which
    RDKit rejects. Demote one N=O to a single bond and set the charges that make
    it valid. Runs before formal-charge inference and sanitize.
    """
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 7:
            continue
        dbl_term_o = [
            b
            for b in atom.GetBonds()
            if b.GetBondType() == Chem.BondType.DOUBLE
            and b.GetOtherAtom(atom).GetAtomicNum() == 8
            and b.GetOtherAtom(atom).GetDegree() == 1
        ]
        if len(dbl_term_o) < 2:
            continue
        for bond in dbl_term_o[1:]:
            bond.SetBondType(Chem.BondType.SINGLE)
            bond.GetOtherAtom(atom).SetFormalCharge(-1)
        atom.SetFormalCharge(1)


# Neutral (uncharged) valence per element; charge = observed valence - this.
# Only N and O: C is ambiguous, and S/P have expanded octets (their charge sits
# on the bonded O, handled by the O rule -- e.g. phosphate/sulfonate).
_NEUTRAL_VALENCE = {7: 3, 8: 2}


def _is_counterion_oxygen(atom: Chem.Atom) -> bool:
    """True for a terminal O bonded to a cationic N (nitro / N-oxide oxygen).

    Such an O is the counter-anion of a hard N cation, never an omitted-H
    hydroxyl, so it must be -1 even without explicit H. The N is cationic when
    aromatic (pyridinium N-oxide) or over-valent (nitro/tertiary N-oxide);
    a neutral hydroxylamine R2N-OH (valence-3 N) is excluded.
    """
    if atom.GetAtomicNum() != 8 or atom.GetDegree() != 1:
        return False
    n = atom.GetNeighbors()[0]
    if n.GetAtomicNum() != 7:
        return False
    if n.GetIsAromatic():
        return True
    return int(round(sum(b.GetBondTypeAsDouble() for b in n.GetBonds()))) > 3


def _assign_formal_charges_from_valence(mol: Chem.Mol) -> None:
    """Infer formal charge from the explicit bond orders (Lewis rule).

    This is only run on structures containing explicit H.  For others, we
    let dimorphite infer both protonation state and charge -- except a nitro /
    N-oxide oxygen, which stays -1 even without H (see _is_counterion_oxygen).
    """
    has_explicit_h = any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms())
    for atom in mol.GetAtoms():
        neutral = _NEUTRAL_VALENCE.get(atom.GetAtomicNum())
        if neutral is None or atom.GetFormalCharge() != 0:
            continue
        bonds = atom.GetBonds()
        if any(b.GetBondType() == Chem.BondType.AROMATIC for b in bonds):
            continue
        valence = sum(b.GetBondTypeAsDouble() for b in bonds)
        charge = int(round(valence)) - neutral
        if charge < 0 and not has_explicit_h and not _is_counterion_oxygen(atom):
            continue  # ambiguous: omitted H, not a real anion -- leave neutral
        if charge != 0:
            atom.SetFormalCharge(charge)


def rdkit_mol_from_ligand_atom_array(
    atom_array: struc.AtomArray,
    *,
    res_name: str = "ligand",
    keep_hydrogens: bool = False,
) -> Chem.Mol:
    """Build an RDKit Mol from a ligand AtomArray's explicit bond table.

    The single AtomArray -> RDKit builder for the ligand pipeline: it preserves
    the source's explicit bond orders (restoring Kekulé forms biotite's
    ``to_mol`` collapses) and aromatic/subtype annotations. Bond perception from
    geometry is intentionally unsupported — the input must carry chemistry-level
    bond orders.

    Args:
        atom_array: The ligand sub-array (heavy + optional hydrogen atoms).
        res_name: Residue code, used only for log/error messages.
        keep_hydrogens: When True, retain explicit hydrogens from the input
            (used for ``skip_protonation`` — preserve mol2/CIF protonation).
    """
    has_bonds = atom_array.bonds is not None and atom_array.bonds.get_bond_count() > 0
    if len(atom_array) == 0:
        raise ValueError(f"{res_name}: empty atom array")
    if not has_bonds:
        raise ValueError(
            f"{res_name}: ligand bond inference is unsupported. "
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
            res_name,
            unsupported,
        )

    # BondType.ANY (0) marks bonds whose order was perceived from geometry
    # (PDB/topology); explicit SINGLE (1) is a real, provided bond order, so a
    # fully saturated ligand is not topology-only.
    has_custom_aromatic_flags = hasattr(atom_array, "tmol_aromatic")
    has_topology_only_bonds = any(t == int(struc.BondType.ANY) for t in raw_types)
    if has_topology_only_bonds and not has_custom_aromatic_flags:
        raise ValueError(
            f"{res_name}: ligand has topology-only bonds with no "
            "chemistry-level bond-order/aromatic annotations. "
            "PDB ligand chemistry inference is unsupported; provide ligand as CIF "
            "with explicit bond orders."
        )

    try:
        mol = to_mol(atom_array)
    except Exception as exc:
        raise ValueError(
            f"{res_name}: failed to read explicit ligand bond chemistry "
            f"from input ({exc}). Provide a CIF with explicit bond orders."
        ) from exc
    _restore_kekule_bonds(mol, atom_array)
    mol = normalize_cumulated_azide(mol)
    normalize_non_ring_aromatic_bonds(mol)
    _apply_source_subtypes(mol, atom_array)
    _normalize_nitro(mol)
    _assign_formal_charges_from_valence(mol)

    if keep_hydrogens:
        arr_indices = list(range(len(atom_array)))
    else:
        heavy_arr_indices = [
            i for i, e in enumerate(atom_array.element) if str(e) != "H"
        ]
        mol = _remove_hs_tolerant(mol)
        arr_indices = heavy_arr_indices

    _apply_atom_array_annotations(mol, atom_array, arr_indices)
    mol = _strip_metals(mol)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"{res_name}: failed to build RDKit Mol")
    return mol


def ligand_atom_array_to_rdkit_mol(
    ligand_info: NonStandardResidueInfo,
    *,
    keep_hydrogens: bool = False,
) -> Chem.Mol:
    """Build an RDKit Mol from a detected ligand's AtomArray.

    Thin wrapper over :func:`rdkit_mol_from_ligand_atom_array`.
    """
    return rdkit_mol_from_ligand_atom_array(
        ligand_info.atom_array,
        res_name=ligand_info.res_name,
        keep_hydrogens=keep_hydrogens,
    )
