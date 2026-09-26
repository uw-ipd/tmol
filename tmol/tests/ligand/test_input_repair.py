"""The geometry-based carboxylate repair, and which oxygen it charges."""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from tmol.ligand._input_repair import correct_carboxylate_bond_orders


def _diol_encoded_acetate() -> Chem.Mol:
    """Acetate with both C-O bonds written single, which is what the repair is for."""
    mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)[O-]"))
    AllChem.EmbedMolecule(mol, randomSeed=0)
    AllChem.MMFFOptimizeMolecule(mol)
    rw = Chem.RWMol(Chem.RemoveHs(mol))
    for bond in rw.GetBonds():
        pair = {bond.GetBeginAtom().GetAtomicNum(), bond.GetEndAtom().GetAtomicNum()}
        if pair == {6, 8}:
            bond.SetBondType(Chem.BondType.SINGLE)
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() == 8:
            atom.SetFormalCharge(0)
            atom.SetNoImplicit(True)
            atom.SetNumExplicitHs(0)
    return rw.GetMol()


def test_a_carboxylate_written_as_a_diol_is_corrected():
    """Two single C-O bonds on a planar carbon are a carboxylate missing its double bond."""
    repaired = correct_carboxylate_bond_orders(_diol_encoded_acetate())

    assert min(a.GetFormalCharge() for a in repaired.GetAtoms()) == -1
    assert any(b.GetBondType() == Chem.BondType.DOUBLE for b in repaired.GetBonds())


def test_the_shorter_bond_becomes_the_carbonyl():
    """Which oxygen carries the charge follows the geometry, not the atom order.

    Taking the two terminal oxygens in neighbour order let the order they happen
    to appear in decide which one was written as the carbonyl.
    """
    repaired = correct_carboxylate_bond_orders(_diol_encoded_acetate())
    conf = repaired.GetConformer()

    carbonyl = [
        b for b in repaired.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE
    ][0]
    single = [
        b
        for b in repaired.GetBonds()
        if b.GetBondType() == Chem.BondType.SINGLE
        and {b.GetBeginAtom().GetAtomicNum(), b.GetEndAtom().GetAtomicNum()} == {6, 8}
    ][0]

    def length(bond):
        a = np.asarray(conf.GetAtomPosition(bond.GetBeginAtomIdx()))
        b = np.asarray(conf.GetAtomPosition(bond.GetEndAtomIdx()))
        return float(np.linalg.norm(a - b))

    assert length(carbonyl) <= length(single)
