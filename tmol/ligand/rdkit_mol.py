"""RDKit molecule construction and protonation for ligands.

Builds RDKit Mol objects from ligand AtomArrays and protonates them at a
target pH using the vendored dimorphite_dl module (direct Mol path, no
SMILES roundtrip in the main pipeline).
"""

import logging

from biotite.interface.rdkit import to_mol
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

from tmol.ligand.detect import NonStandardResidueInfo
from tmol.ligand.dimorphite_dl import protonate_mol_variants

logger = logging.getLogger(__name__)


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


def ligand_atom_array_to_rdkit_mol(ligand_info: NonStandardResidueInfo):
    """Build an RDKit Mol directly from a ligand AtomArray."""
    atom_array = ligand_info.atom_array
    has_bonds = atom_array.bonds is not None and atom_array.bonds.get_bond_count() > 0
    if has_bonds:
        mol = to_mol(atom_array)
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

    mol = Chem.RemoveHs(mol)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"{ligand_info.res_name}: failed to build RDKit Mol")
    return mol


def protonate_ligand_mol(
    mol,
    ph: float = 7.4,
    precision: float = 0.1,
):
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
