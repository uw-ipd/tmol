"""Independent ligand preparations must not share mutable force-field state."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np
import pytest

from tmol.ligand import _conformer_generation as generation


@pytest.mark.parametrize(
    "forcefield,second_operation",
    [
        ("mmff94", "minimize"),
        ("uff", "minimize"),
        ("mmff94", "ligand_charges"),
        ("mmff94", "terminal_charges"),
    ],
)
def test_forcefield_cleanup_preserves_each_threads_molecule(
    monkeypatch, forcefield, second_operation
):
    from openbabel import openbabel as ob, pybel

    def coordinates(molecule):
        return np.array(
            [
                (atom.GetX(), atom.GetY(), atom.GetZ())
                for atom in ob.OBMolAtomIter(molecule)
            ]
        )

    molecules = []
    for smiles, shift in (("CC(=O)C", 0), ("CCC=O", 100)):
        molecule = pybel.readstring("smi", smiles)
        molecule.addh()
        assert ob.OBBuilder().Build(molecule.OBMol)
        for atom in ob.OBMolAtomIter(molecule.OBMol):
            atom.SetVector(atom.GetX() + shift, atom.GetY(), atom.GetZ())
        molecules.append(molecule.OBMol)

    # Exercise both native models, including the UFF fallback transaction.
    find = ob.OBForceField.FindForceField
    monkeypatch.setattr(ob.OBForceField, "FindForceField", lambda _: find(forcefield))

    def operate(molecule, index):
        if index == 0 or second_operation == "minimize":
            generation._forcefield_minimize(molecule, steps=5)
        else:
            from tmol.ligand._openbabel_compat import _model_partial_charges
            from tmol.ligand._terminus_patches import _mmff94_charges
            from rdkit import Chem

            if second_operation == "ligand_charges":
                charges = _model_partial_charges(
                    ob, pybel.Molecule(molecule), "mmff94", "CCC=O"
                )
            else:
                rdkit_mol = Chem.MolFromMolBlock(
                    pybel.Molecule(molecule).write("mol"), removeHs=False
                )
                charges = _mmff94_charges(rdkit_mol, molecule.NumAtoms())
            assert charges is not None and np.isfinite(charges).all()
        return coordinates(molecule)

    expected = [operate(ob.OBMol(molecule), i) for i, molecule in enumerate(molecules)]

    first_ready, second_requested, second_ready = Event(), Event(), Event()
    copies = [ob.OBMol(molecule) for molecule in molecules]
    first = int(copies[0].this)
    setup = ob.OBForceField.Setup

    def interleaved_setup(force_field, molecule, *args):
        result = setup(force_field, molecule, *args)
        if molecule.NumAtoms():
            if int(molecule.this) == first:
                first_ready.set()
                assert second_requested.wait(10)
                # Without serialization, the second Setup replaces the first
                # molecule before its minimization. With the lock, it waits.
                second_ready.wait(0.5)
            else:
                second_ready.set()
        return result

    monkeypatch.setattr(ob.OBForceField, "Setup", interleaved_setup)

    def minimize(index):
        if index:
            assert first_ready.wait(10)
            second_requested.set()
        result = operate(copies[index], index)
        if index:
            second_ready.set()
        return result

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(minimize, i) for i in range(2)]
        actual = [future.result(timeout=15) for future in futures]
    for observed, reference in zip(actual, expected):
        np.testing.assert_allclose(observed, reference, rtol=1e-8, atol=1e-8)
