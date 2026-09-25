"""The geometry-based bond-order repairs, and what they must leave alone."""

from rdkit import Chem
from rdkit.Chem import AllChem

from tmol.ligand._input_repair import correct_carboxylate_bond_orders


def _embedded(smiles: str) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=0)
    AllChem.MMFFOptimizeMolecule(mol)
    return Chem.RemoveHs(mol)


def test_a_neutral_carboxylic_acid_keeps_its_proton():
    """Both its oxygens are terminal and both C-O bonds are short.

    The repair reads that shape as a carboxylate written as a geminal diol, so
    without asking whether a carbonyl is already written it deprotonates an acid
    that was encoded correctly to begin with.
    """
    acid = _embedded("CC(=O)O")

    repaired = correct_carboxylate_bond_orders(acid)

    assert Chem.MolToSmiles(repaired) == Chem.MolToSmiles(acid)
    assert sum(a.GetFormalCharge() for a in repaired.GetAtoms()) == 0


def test_a_carboxylate_written_as_a_diol_is_corrected():
    """The case the repair exists for: two single C-O bonds on a planar carbon."""
    acetate = _embedded("CC(=O)[O-]")
    rw = Chem.RWMol(acetate)
    for bond in rw.GetBonds():
        if {bond.GetBeginAtom().GetAtomicNum(), bond.GetEndAtom().GetAtomicNum()} == {
            6,
            8,
        }:
            bond.SetBondType(Chem.BondType.SINGLE)
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() == 8:
            atom.SetFormalCharge(0)
            atom.SetNoImplicit(True)
            atom.SetNumExplicitHs(0)

    repaired = correct_carboxylate_bond_orders(rw.GetMol())

    charges = sorted(a.GetFormalCharge() for a in repaired.GetAtoms())
    assert charges[0] == -1, Chem.MolToSmiles(repaired)
    assert any(b.GetBondType() == Chem.BondType.DOUBLE for b in repaired.GetBonds())
