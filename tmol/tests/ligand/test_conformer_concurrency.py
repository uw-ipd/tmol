"""Independent ligand preparations must not share mutable force-field state."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np
import pytest

from tmol.ligand import _conformer_generation as generation


@pytest.mark.parametrize("forcefield", ("mmff94", "uff"))
def test_forcefield_cleanup_preserves_each_threads_molecule(monkeypatch, forcefield):
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
    expected = []
    for molecule in molecules:
        copy = ob.OBMol(molecule)
        generation._forcefield_minimize(copy, steps=5)
        expected.append(coordinates(copy))

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
        generation._forcefield_minimize(copies[index], steps=5)
        return coordinates(copies[index])

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(minimize, i) for i in range(2)]
        actual = [future.result(timeout=15) for future in futures]
    for observed, reference in zip(actual, expected):
        np.testing.assert_allclose(observed, reference, rtol=1e-8, atol=1e-8)
