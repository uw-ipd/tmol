"""Molecular charge computation for ligands.

MMFF94 partial charges via RDKit, with a Gasteiger fallback for molecules
RDKit's MMFF94 implementation cannot parameterize (e.g. porphyrin-like
kekulization edge cases).
"""

import logging

from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)


def compute_mmff94_charges(mol: Chem.Mol) -> dict[int, float]:
    """Compute MMFF94 partial charges via RDKit.

    Falls back to Gasteiger charges when MMFF94 parameterization fails
    (e.g. unusual element combinations or kekulization issues like
    porphyrin rings in HEM).
    """
    try:
        props = AllChem.MMFFGetMoleculeProperties(mol)
    except Exception:
        props = None
    if props is None:
        logger.warning("MMFF94 parameterization failed, falling back to Gasteiger")
        AllChem.ComputeGasteigerCharges(mol)
        return {
            i: float(mol.GetAtomWithIdx(i).GetDoubleProp("_GasteigerCharge"))
            for i in range(mol.GetNumAtoms())
        }
    return {i: props.GetMMFFPartialCharge(i) for i in range(mol.GetNumAtoms())}
