"""Unresolved coordinates cannot supply evidence for bond-order repairs."""

import numpy as np
import pytest
from rdkit import Chem

from tmol.ligand._structure_to_smiles import _infer_carboxylate_bonds


def motif():
    rw = Chem.RWMol()
    for element in (6, 6, 8, 8):
        rw.AddAtom(Chem.Atom(element))
    for neighbor in (1, 2, 3):
        rw.AddBond(0, neighbor, Chem.BondType.SINGLE)
    conf = Chem.Conformer(4)
    for i, xyz in enumerate(
        [(0, 0, 0), (-1.5, 0, 0), (0.625, 1.08253175, 0), (0.625, -1.08253175, 0)]
    ):
        conf.SetAtomPosition(i, xyz)
    rw.AddConformer(conf)
    return rw


@pytest.mark.parametrize("atom", range(4))
@pytest.mark.parametrize("coordinate", [np.nan, np.inf, -np.inf])
def test_missing_geometry_does_not_change_bonds_or_charges(atom, coordinate):
    rw = motif()
    rw.GetConformer().SetAtomPosition(atom, (coordinate, 0, 0))
    assert _infer_carboxylate_bonds(rw, rw.GetConformer()) == 0
    assert all(b.GetBondType() == Chem.BondType.SINGLE for b in rw.GetBonds())
    assert all(a.GetFormalCharge() == 0 for a in rw.GetAtoms())


@pytest.mark.parametrize("atom", [1, 2, 3])
def test_coincident_geometry_does_not_repair(atom):
    rw = motif()
    rw.GetConformer().SetAtomPosition(atom, (0, 0, 0))
    assert _infer_carboxylate_bonds(rw, rw.GetConformer()) == 0


def test_finite_planar_geometry_repairs_both_oxygen_charges():
    rw = motif()
    rw.GetAtomWithIdx(2).SetFormalCharge(-1)
    assert _infer_carboxylate_bonds(rw, rw.GetConformer()) == 1
    assert rw.GetBondBetweenAtoms(0, 2).GetBondType() == Chem.BondType.DOUBLE
    assert rw.GetBondBetweenAtoms(0, 3).GetBondType() == Chem.BondType.SINGLE
    assert rw.GetAtomWithIdx(2).GetFormalCharge() == 0
    assert rw.GetAtomWithIdx(3).GetFormalCharge() == -1


def test_nonplanar_short_bonds_do_not_repair():
    rw = motif()
    rw.GetConformer().SetAtomPosition(1, (0, 0, 1.5))
    assert _infer_carboxylate_bonds(rw, rw.GetConformer()) == 0


def test_invalid_motif_does_not_disable_other_finite_motifs():
    bad, good = motif(), motif()
    bad.GetConformer().SetAtomPosition(1, (np.nan, 0, 0))
    rw = Chem.RWMol(Chem.CombineMols(bad, good))
    assert _infer_carboxylate_bonds(rw, rw.GetConformer()) == 1
    assert rw.GetBondBetweenAtoms(0, 2).GetBondType() == Chem.BondType.SINGLE
    assert rw.GetBondBetweenAtoms(4, 6).GetBondType() == Chem.BondType.DOUBLE
