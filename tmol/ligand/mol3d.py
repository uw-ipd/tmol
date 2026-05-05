"""Molecular charge computation for ligands.

MMFF94 partial charges via RDKit, with a Gasteiger fallback for molecules
RDKit's MMFF94 implementation cannot parameterize (e.g. porphyrin-like
kekulization edge cases).
"""

import logging
from typing import Mapping, Optional

from rdkit import Chem
from rdkit.Chem import AllChem

from tmol.ligand.atom_typing import AtomTypeAssignment

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
        charges = {}
        for i in range(mol.GetNumAtoms()):
            q = float(mol.GetAtomWithIdx(i).GetDoubleProp("_GasteigerCharge"))
            if q != q:  # NaN check
                q = 0.0
            charges[i] = q
        return charges
    return {i: props.GetMMFFPartialCharge(i) for i in range(mol.GetNumAtoms())}


def build_partial_charges(
    mol: Chem.Mol,
    atom_types: list[AtomTypeAssignment],
    input_charges: Optional[Mapping[str, float]] = None,
) -> dict[str, float]:
    """Return ``{atom_name: partial_charge}`` for every atom in ``atom_types``.

    Authoritative caller-supplied charges (e.g. AM1-BCC from mol2) win
    when present; remaining atoms — typically the explicit Hs added by
    ``Chem.AddHs`` — fall back to MMFF94 (with a Gasteiger fallback when
    MMFF94 cannot parameterize).
    """
    by_index = compute_mmff94_charges(mol)
    by_name = dict(input_charges or {})
    charges: dict[str, float] = {}
    for at in atom_types:
        if at.atom_name in by_name:
            charges[at.atom_name] = by_name[at.atom_name]
        elif at.index in by_index:
            charges[at.atom_name] = by_index[at.index]
    return charges
