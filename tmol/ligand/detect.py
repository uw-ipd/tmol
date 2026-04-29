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
from biotite.interface.rdkit import to_mol
from rdkit import Chem
from rdkit.Chem import RWMol, rdDetermineBonds

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
class NonStandardResidueInfo:
    """Detected non-standard residue requiring preparation.

    Any residue not in tmol's standard database is represented here,
    regardless of whether it is a true ligand, modified amino acid,
    or modified nucleotide.

    Attributes:
        res_name: Three-letter residue code (e.g. "ATP", "NAG").
        ccd_type: CCD chemical component type string, or "UNKNOWN" if the
            residue is not in the CCD.  Informational only.
        atom_names: Atom names for one representative instance.
        elements: Element symbols for each atom.
        coords: Cartesian coordinates of shape (n_atoms, 3).
        atom_array: The sub-AtomArray (with bonds if available).
        ccd_smiles: Canonical SMILES from the CCD, or None if unavailable.
    """

    res_name: str
    ccd_type: str
    atom_names: tuple[str, ...]
    elements: tuple[str, ...]
    coords: np.ndarray = attr.ib(eq=False, hash=False)
    atom_array: struc.AtomArray = attr.ib(eq=False, hash=False)
    ccd_smiles: Optional[str] = None
    covalently_linked: bool = False


LigandInfo = NonStandardResidueInfo


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
        logger.debug("Failed to convert AtomArray to SMILES", exc_info=True)
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


def detect_nonstandard_residues(
    atom_array: struc.AtomArray,
    canonical_ordering: CanonicalOrdering,
) -> list[NonStandardResidueInfo]:
    """Detect residues in an AtomArray that are not in tmol's database.

    Any residue whose 3-letter code is not in the canonical ordering
    is returned for preparation, regardless of whether it is a ligand,
    modified amino acid, or modified nucleotide.

    Args:
        atom_array: Biotite AtomArray from a CIF or PDB file.
        canonical_ordering: The current tmol CanonicalOrdering, which
            defines known residue types.

    Returns:
        A list of NonStandardResidueInfo objects, one per unique unknown
        residue name.
    """
    known_names = set(canonical_ordering.restype_io_equiv_classes)
    seen: set[str] = set()
    results: list[NonStandardResidueInfo] = []

    covalently_linked_names = _residue_names_with_cross_residue_bonds(atom_array)

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
        ccd_type = get_chem_comp_type(res_name) or "UNKNOWN"
        ccd_smiles = _get_ccd_smiles(res_name)

        logger.info(
            "Detected non-standard residue %s (CCD type: %s, %d atoms)",
            res_name,
            ccd_type,
            len(sub),
        )

        results.append(
            NonStandardResidueInfo(
                res_name=res_name,
                ccd_type=ccd_type,
                atom_names=tuple(sub.atom_name),
                elements=tuple(sub.element),
                coords=sub.coord.copy(),
                atom_array=sub,
                ccd_smiles=ccd_smiles,
                covalently_linked=res_name in covalently_linked_names,
            )
        )

    return results


def _residue_names_with_cross_residue_bonds(
    atom_array: struc.AtomArray,
    spatial_cutoff: float = 1.8,
) -> frozenset[str]:
    """Return the set of res_names that have at least one bond to a different residue.

    A "different residue" is identified by (chain_id, res_id, res_name) — so
    distinct instances of the same residue name (e.g. two NAGs in a glycan
    chain) also count as cross-residue bonds. Used to flag ligands that are
    covalently attached to a polymer or to other ligand instances.

    Detection runs two passes:
    1. Explicit bonds in ``atom_array.bonds`` (if present).
    2. Heavy-atom spatial proximity within ``spatial_cutoff`` Å. This
       catches covalent attachments missing from the bond table — e.g.
       CCD NON-POLYMER caps like ACE in a polypeptide, or files lacking
       ``_struct_conn`` records.
    """
    chain_ids = atom_array.chain_id if hasattr(atom_array, "chain_id") else None
    res_ids = atom_array.res_id
    res_names = atom_array.res_name

    linked: set[str] = set()

    def _record(a: int, b: int) -> None:
        same_chain = chain_ids is None or chain_ids[a] == chain_ids[b]
        if same_chain and res_ids[a] == res_ids[b] and res_names[a] == res_names[b]:
            return
        linked.add(res_names[a].strip())
        linked.add(res_names[b].strip())

    if atom_array.bonds is not None and atom_array.bonds.get_bond_count() > 0:
        for a, b, _ in atom_array.bonds.as_array():
            _record(int(a), int(b))

    if len(atom_array) > 1:
        heavy_mask = np.char.strip(atom_array.element.astype(str)) != "H"
        if heavy_mask.any():
            from scipy.spatial import cKDTree

            heavy_indices = np.nonzero(heavy_mask)[0]
            tree = cKDTree(atom_array.coord[heavy_mask])
            for i, j in tree.query_pairs(spatial_cutoff, output_type="ndarray"):
                _record(int(heavy_indices[i]), int(heavy_indices[j]))

    return frozenset(linked)
