"""The geometry-based carboxylate repair, and which oxygen it charges."""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

from tmol.ligand._input_repair import correct_carboxylate_bond_orders


def _diol_encoded_acetate(longer_oxygen: int) -> Chem.Mol:
    """Acetate with both C-O bonds written single and one of them 0.05 A longer."""
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
    conf = rw.GetConformer()
    carbon = np.asarray(conf.GetAtomPosition(1))
    oxygen = np.asarray(conf.GetAtomPosition(longer_oxygen))
    moved = oxygen + 0.05 * (oxygen - carbon) / np.linalg.norm(oxygen - carbon)
    conf.SetAtomPosition(longer_oxygen, Point3D(*moved))
    return rw.GetMol()


@pytest.mark.parametrize("longer_oxygen", [2, 3])
def test_a_carboxylate_written_as_a_diol_takes_its_carbonyl_from_the_geometry(
    longer_oxygen,
):
    """Two single C-O bonds on a planar carbon are a carboxylate missing its double
    bond; the shorter one becomes the carbonyl whichever oxygen comes first."""
    repaired = correct_carboxylate_bond_orders(_diol_encoded_acetate(longer_oxygen))

    (carbonyl,) = [
        b for b in repaired.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE
    ]
    shorter_oxygen = 5 - longer_oxygen
    assert {carbonyl.GetBeginAtomIdx(), carbonyl.GetEndAtomIdx()} == {1, shorter_oxygen}
    assert [a.GetFormalCharge() for a in repaired.GetAtoms()] == [
        -1 if a.GetIdx() == longer_oxygen else 0 for a in repaired.GetAtoms()
    ]


def test_a_nitro_group_written_with_two_double_bonds_is_localized():
    """CCD VYZ writes its nitro group N(=O)=O, past nitrogen's valence; like any
    delocalized group it takes one double bond, an O- and an N+."""
    from atomworks.io.utils.ccd import atom_array_from_ccd_code

    from tmol.ligand._rdkit_mol import rdkit_mol_from_ligand_atom_array

    component = atom_array_from_ccd_code("VYZ")
    mol = rdkit_mol_from_ligand_atom_array(component[component.element != "H"])
    assert Chem.MolToSmiles(mol) == "CC(=O)Nc1ccc([N+](=O)[O-])cc1"
