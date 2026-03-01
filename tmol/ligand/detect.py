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
    """

    res_name: str
    ccd_type: str
    is_ligand: bool
    atom_names: tuple[str, ...]
    elements: tuple[str, ...]
    coords: np.ndarray


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

        logger.info(
            "Detected non-standard residue %s (CCD type: %s, ligand: %s, %d atoms)",
            res_name,
            ccd_type,
            is_ligand,
            len(sub),
        )

        ligands.append(
            LigandInfo(
                res_name=res_name,
                ccd_type=ccd_type,
                is_ligand=is_ligand,
                atom_names=tuple(sub.atom_name),
                elements=tuple(sub.element),
                coords=sub.coord.copy(),
            )
        )

    return ligands
