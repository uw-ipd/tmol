"""RDKit molecule construction and protonation for ligands.

Builds RDKit Mol objects from ligand AtomArrays and protonates them at a
target pH using the vendored dimorphite_dl module (direct Mol path, no
SMILES roundtrip in the main pipeline).
"""

import logging

import biotite.structure as struc
from biotite.interface.rdkit import to_mol
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

from tmol.ligand.atom_typing import sanitize_tolerant
from tmol.ligand.detect import NonStandardResidueInfo, _strip_metals
from tmol.ligand.dimorphite_dl import protonate_mol_variants

logger = logging.getLogger(__name__)


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


def _restore_kekule_bonds(mol: Chem.Mol, atom_array) -> None:
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


def _apply_atom_aromatic_flags_post_removeh(
    mol: Chem.Mol, atom_array, heavy_arr_indices
) -> None:
    """Apply ``atom_array.tmol_aromatic`` (custom CIF flag) to the RDKit
    mol after Hs were removed.

    ``RemoveHs`` runs ``Chem.SanitizeMol`` internally, which re-perceives
    aromaticity from the ring-system topology and overwrites whatever
    flags we previously set. We re-stamp here, mapping the post-RemoveHs
    heavy-atom indices back to the source AtomArray indices via
    ``heavy_arr_indices`` (which was captured before Hs were stripped).
    Sets ``_SOURCE_KEKULE_PROP`` so downstream sanitize_tolerant skips
    its own aromaticity perception, and stamps each atom's source-mol2
    subtype hint (``ar`` / ``2`` / ``cat`` / ``3`` / ``pl3`` / …) onto
    an atom prop so the carbon classifier can pick CR vs CD without
    re-deriving it from RDKit's perception.
    """
    if not hasattr(atom_array, "tmol_aromatic"):
        return
    flags = atom_array.tmol_aromatic
    subtypes = (
        atom_array.tmol_source_subtype
        if hasattr(atom_array, "tmol_source_subtype")
        else None
    )
    if mol.GetNumAtoms() != len(heavy_arr_indices):
        return
    for mol_idx, arr_idx in enumerate(heavy_arr_indices):
        a = mol.GetAtomWithIdx(mol_idx)
        a.SetIsAromatic(bool(flags[arr_idx]))
        if subtypes is not None:
            sub = str(subtypes[arr_idx])
            if sub and sub != "?":
                a.SetProp(_SOURCE_SUBTYPE_PROP, sub)
    for bond in mol.GetBonds():
        if bond.GetBeginAtom().GetIsAromatic() and bond.GetEndAtom().GetIsAromatic():
            bond.SetIsAromatic(True)
        else:
            bond.SetIsAromatic(False)
    mol.SetProp(_SOURCE_KEKULE_PROP, "1")


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
    except (Chem.rdchem.KekulizeException, Chem.rdchem.AtomValenceException):
        return Chem.RemoveHs(mol, sanitize=False)


def ligand_atom_array_to_rdkit_mol(ligand_info: NonStandardResidueInfo) -> Chem.Mol:
    """Build an RDKit Mol directly from a ligand AtomArray."""
    atom_array = ligand_info.atom_array
    has_bonds = atom_array.bonds is not None and atom_array.bonds.get_bond_count() > 0
    if has_bonds:
        mol = to_mol(atom_array)
        _restore_kekule_bonds(mol, atom_array)
    else:
        if len(atom_array) == 0:
            raise ValueError(f"{ligand_info.res_name}: empty atom array")
        rwmol = Chem.RWMol()
        conf = Chem.Conformer(len(atom_array))
        for i, (elem, coord) in enumerate(zip(atom_array.element, atom_array.coord)):
            rwmol.AddAtom(Chem.Atom(elem.strip().capitalize()))
            conf.SetAtomPosition(i, (float(coord[0]), float(coord[1]), float(coord[2])))
        rwmol.AddConformer(conf, assignId=True)
        if rwmol.GetNumAtoms() > 1:
            rdDetermineBonds.DetermineBonds(rwmol)
        mol = rwmol.GetMol()

    # Map heavy-atom indices from the source atom_array to the post-
    # ``RemoveHs`` mol so we can re-stamp the source-supplied aromatic
    # flag after RemoveHs's internal sanitize re-perceived aromaticity.
    if has_bonds:
        heavy_arr_indices = [
            i for i, e in enumerate(atom_array.element) if str(e) != "H"
        ]
    else:
        heavy_arr_indices = None

    mol = _remove_hs_tolerant(mol)
    if has_bonds and heavy_arr_indices is not None:
        _apply_atom_aromatic_flags_post_removeh(mol, atom_array, heavy_arr_indices)
    mol = _strip_metals(mol)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"{ligand_info.res_name}: failed to build RDKit Mol")
    return mol


def protonate_ligand_mol(
    mol: Chem.Mol,
    ph: float = 7.4,
    precision: float = 0.1,
) -> Chem.Mol:
    """Protonate an RDKit Mol at a target pH and return first variant."""
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
            return variants[0]
    except Exception:
        logger.warning(
            "Dimorphite-DL direct-Mol protonation failed; keeping input mol",
            exc_info=True,
        )
    return mol
