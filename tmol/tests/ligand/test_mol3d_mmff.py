"""Tests for the OpenBabel MMFF94 charge computation in tmol.ligand.mol3d.

Charges are computed by handing the RDKit mol to OpenBabel as a Mol block and
running its MMFF94 charge model. OpenBabel re-perceives valence/aromaticity, so
states that RDKit's MMFF gate rejects (protonated amines, hypervalent S, fused
aromatics) are handled without any formal-charge repair. On failure we raise;
there is no silent Gasteiger fallback.
"""

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

import tmol.ligand.mol3d as mol3d


def _mol_with_coords(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    mol = Chem.AddHs(mol)
    AllChem.Compute2DCoords(mol)  # MolToMolBlock needs a conformer
    return mol


def test_compute_mmff94_charges_keys_and_neutral_sum():
    mol = _mol_with_coords("c1ccncc1")  # pyridine, neutral
    charges = mol3d.compute_mmff94_charges(mol)
    assert set(charges) == set(range(mol.GetNumAtoms()))
    assert abs(sum(charges.values())) < 1e-2


def test_compute_mmff94_charges_handles_protonated_amine():
    # Protonated amine: N at valence 4. RDKit's MMFF gate historically rejected
    # this; OpenBabel parameterizes it directly. Net charge of the cation is +1.
    mol = _mol_with_coords("CC[NH3+]")
    charges = mol3d.compute_mmff94_charges(mol)
    assert set(charges) == set(range(mol.GetNumAtoms()))
    assert abs(sum(charges.values()) - 1.0) < 1e-2


def test_compute_mmff94_charges_raises_on_failure(monkeypatch):
    # If OpenBabel cannot provide an MMFF94 charge model, we raise loudly
    # (no Gasteiger fallback).
    class _NoModel:
        @staticmethod
        def FindType(_name):
            return None

    monkeypatch.setattr(mol3d.openbabel, "OBChargeModel", _NoModel)
    with pytest.raises(RuntimeError, match="MMFF94 parameterization failed"):
        mol3d.compute_mmff94_charges(_mol_with_coords("c1ccncc1"))
