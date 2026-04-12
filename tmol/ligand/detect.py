"""Detection of non-standard residues in biotite AtomArrays.

Identifies residues that are not represented in tmol's ChemicalDatabase
and classifies them using Biotite's built-in Chemical Component Dictionary
(CCD) as either true ligands (non-polymer) or modified amino acids /
nucleotides (polymer-linked).
"""

import functools
import logging
from typing import Optional

import attr
import biotite.structure as struc
import biotite.structure.info as struc_info
import biotite.structure.info.ccd as ccd
import numpy as np

from tmol.io.canonical_ordering import CanonicalOrdering

logger = logging.getLogger(__name__)

AA_LIKE_CHEM_TYPES = frozenset(
    {
        "D-PEPTIDE LINKING",
        "D-PEPTIDE NH3 AMINO TERMINUS",
        "L-PEPTIDE LINKING",
        "L-PEPTIDE NH3 AMINO TERMINUS",
        "PEPTIDE LINKING",
        "PEPTIDE-LIKE",
    }
)

NA_LIKE_CHEM_TYPES = frozenset(
    {
        "DNA LINKING",
        "DNA OH 3 PRIME TERMINUS",
        "DNA OH 5 PRIME TERMINUS",
        "L-DNA LINKING",
        "L-RNA LINKING",
        "RNA LINKING",
        "RNA OH 3 PRIME TERMINUS",
        "RNA OH 5 PRIME TERMINUS",
    }
)

SKIP_RESIDUES = frozenset({"HOH", "WAT", "DOD", "VRT"})


@attr.s(auto_attribs=True, frozen=True)
class LigandInfo:
    """Detected non-standard residue requiring ligand preparation.

    Attributes:
        res_name: Three-letter residue code (e.g. "ATP", "NAG").
        ccd_type: CCD chemical component type string, or "UNKNOWN" if the
            residue is not in the CCD.
        is_ligand: True if the residue is a non-polymer (true ligand),
            False if it is a modified amino acid or nucleotide.
        atom_names: Atom names for one representative instance.
        elements: Element symbols for each atom.
        coords: Cartesian coordinates of shape (n_atoms, 3).
        atom_array: The ligand sub-AtomArray (with bonds if available).
        ccd_smiles: Canonical SMILES from the CCD, or None if unavailable.
    """

    res_name: str
    ccd_type: str
    is_ligand: bool
    atom_names: tuple[str, ...]
    elements: tuple[str, ...]
    coords: np.ndarray = attr.ib(eq=False, hash=False)
    atom_array: struc.AtomArray = attr.ib(eq=False, hash=False)
    ccd_smiles: Optional[str] = None


@functools.cache
def _chem_comp_type_dict() -> dict[str, str]:
    """Build a dict mapping CCD component IDs to their chemical type."""
    ccd_data = ccd.get_ccd()
    ids = np.char.upper(ccd_data["chem_comp"]["id"].as_array())
    types = np.char.upper(ccd_data["chem_comp"]["type"].as_array())
    return dict(zip(ids, types))


def get_chem_comp_type(res_name: str) -> Optional[str]:
    """Look up the CCD chemical component type for a residue name.

    Args:
        res_name: Three-letter residue code.

    Returns:
        The CCD type string (e.g. "NON-POLYMER", "L-PEPTIDE LINKING"),
        or None if the code is not found in the CCD.
    """
    return _chem_comp_type_dict().get(res_name.upper())


_METAL_SYMBOLS = frozenset(
    {
        "Fe",
        "Zn",
        "Cu",
        "Mn",
        "Co",
        "Ni",
        "Mg",
        "Ca",
        "Na",
        "K",
        "Cr",
        "Mo",
        "W",
        "V",
        "Pt",
        "Pd",
        "Ru",
        "Rh",
        "Ir",
        "Os",
    }
)


def _strip_metals(mol):
    """Remove metal atoms from an RDKit Mol.

    OpenBabel downstream cannot parse CCD coordination-bond SMILES, and
    metals are dropped during ligand preparation anyway.
    """
    from rdkit.Chem import RWMol

    metals = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() in _METAL_SYMBOLS]
    if metals:
        em = RWMol(mol)
        for idx in sorted(metals, reverse=True):
            em.RemoveAtom(idx)
        return em.GetMol()
    return mol


def _atom_array_to_smiles(atom_array: struc.AtomArray) -> Optional[str]:
    """Convert an AtomArray to a canonical SMILES string via RDKit.

    Uses biotite.interface.rdkit.to_mol() for arrays with bonds.
    Falls back to rdDetermineBonds for arrays without bonds.
    Metal atoms are stripped since OpenBabel cannot handle coordination
    bond SMILES downstream.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdDetermineBonds
        from biotite.interface.rdkit import to_mol
    except ImportError:
        return None

    has_bonds = atom_array.bonds is not None and atom_array.bonds.get_bond_count() > 0

    try:
        if has_bonds:
            mol = to_mol(atom_array)
        else:
            if len(atom_array) == 0:
                return None
            rwmol = Chem.RWMol()
            conf = Chem.Conformer(len(atom_array))
            for i, (elem, coord) in enumerate(
                zip(atom_array.element, atom_array.coord)
            ):
                rwmol.AddAtom(Chem.Atom(elem.strip().capitalize()))
                conf.SetAtomPosition(
                    i, (float(coord[0]), float(coord[1]), float(coord[2]))
                )
            rwmol.AddConformer(conf, assignId=True)
            if rwmol.GetNumAtoms() > 1:
                rdDetermineBonds.DetermineBonds(rwmol)
            mol = rwmol.GetMol()
    except Exception:
        return None

    mol = Chem.RemoveHs(mol)
    mol = _strip_metals(mol)
    smi = Chem.MolToSmiles(mol)
    return smi if smi else None


def _get_ccd_smiles(res_name: str) -> Optional[str]:
    """Look up canonical SMILES for a residue from the CCD.

    Uses biotite.structure.info.residue() to get the full CCD AtomArray
    (with bonds) and converts to SMILES via RDKit.
    Returns None if the component is not in the CCD or conversion fails.
    """
    try:
        ccd_array = struc_info.residue(res_name)
    except KeyError:
        return None
    if ccd_array is None:
        return None
    return _atom_array_to_smiles(ccd_array)


def _classify_residue(res_name: str) -> tuple[str, bool]:
    """Classify a residue as ligand or modified polymer.

    Returns:
        A (ccd_type, is_ligand) tuple.
    """
    ccd_type = get_chem_comp_type(res_name)
    if ccd_type is None:
        return ("UNKNOWN", True)
    if ccd_type in AA_LIKE_CHEM_TYPES or ccd_type in NA_LIKE_CHEM_TYPES:
        return (ccd_type, False)
    return (ccd_type, True)


def detect_nonstandard_residues(
    atom_array: struc.AtomArray,
    canonical_ordering: CanonicalOrdering,
) -> list[LigandInfo]:
    """Detect residues in an AtomArray that are not in tmol's database.

    Groups atoms by residue, checks each unique residue name against the
    canonical ordering, and classifies unknowns via the CCD.

    Args:
        atom_array: Biotite AtomArray from a CIF or PDB file.
        canonical_ordering: The current tmol CanonicalOrdering, which
            defines known residue types.

    Returns:
        A list of LigandInfo objects, one per unique unknown residue name.
        Each contains the atoms, elements, and coordinates of the first
        instance encountered.
    """
    known_names = set(canonical_ordering.restype_io_equiv_classes)
    seen: set[str] = set()
    ligands: list[LigandInfo] = []

    residue_starts = struc.get_residue_starts(atom_array)

    for start in residue_starts:
        res_name = atom_array.res_name[start].strip()

        if res_name in known_names or res_name in SKIP_RESIDUES or res_name in seen:
            continue
        seen.add(res_name)

        mask = atom_array.res_name == atom_array.res_name[start]
        if hasattr(atom_array, "res_id"):
            mask &= atom_array.res_id == atom_array.res_id[start]
        if hasattr(atom_array, "chain_id"):
            mask &= atom_array.chain_id == atom_array.chain_id[start]

        sub = atom_array[mask]
        ccd_type, is_ligand = _classify_residue(res_name)
        ccd_smiles = _get_ccd_smiles(res_name)

        logger.info(
            "Detected non-standard residue %s (CCD type: %s, ligand: %s, "
            "%d atoms, CCD SMILES: %s)",
            res_name,
            ccd_type,
            is_ligand,
            len(sub),
            ccd_smiles,
        )

        ligands.append(
            LigandInfo(
                res_name=res_name,
                ccd_type=ccd_type,
                is_ligand=is_ligand,
                atom_names=tuple(sub.atom_name),
                elements=tuple(sub.element),
                coords=sub.coord.copy(),
                atom_array=sub,
                ccd_smiles=ccd_smiles,
            )
        )

    return ligands
