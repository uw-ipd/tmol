"""Molecular charge computation and OpenBabel conversion for ligands.

The main pipeline uses RDKit for MMFF94 partial charges and converts to
OpenBabel only for atom typing and residue building (MolBlock roundtrip).
"""

import logging

from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)


def compute_mmff94_charges(mol) -> dict[int, float]:
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


def rdkit_mol_to_obmol(rdkit_mol):
    """Convert an RDKit Mol to an OpenBabel molecule via MolBlock roundtrip.

    No 3D generation or minimization is performed -- the RDKit mol is
    expected to already have coordinates (from the crystal structure).
    """
    from rdkit import Chem
    from openbabel import pybel

    mol_block = Chem.MolToMolBlock(rdkit_mol, kekulize=False)
    mol = pybel.readstring("mol", mol_block)
    return mol
