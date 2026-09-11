"""Capped group chemistry preserves explicit links and instance identity."""

import biotite.structure as struc
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from tmol.io import atom_array_from_cif
from tmol.ligand import prepare_ligands
from tmol.ligand._conjugate_model import capped_conjugate_models
from tmol.ligand._detect import _dimorphite_protonate_smiles
from tmol.ligand._structure_to_smiles import ligand_smiles_from_atom_array
from tmol.tests.data import data_path
from tmol.tests.pack.test_conjugated_group_packing import FIXTURES


@pytest.fixture(scope="module", params=sorted(FIXTURES))
def conjugate_input(request):
    array = atom_array_from_cif(
        data_path("covalent_fixtures", FIXTURES[request.param] + ".cif")
    )
    db, _ = prepare_ligands(array, seed=20250828)
    return request.param, array, db


@pytest.mark.parametrize("duplicate", [False, True])
def test_capped_group_identity_and_mmff_coverage(conjugate_input, duplicate):
    fixture, original, db = conjugate_input
    array = original.copy()
    if duplicate:
        second = original.copy()
        second.chain_id[:] = "ZZ"
        array = array + second
    models = capped_conjugate_models(array, db.chemical)
    # The source N-glycan fixture also contains an unanchored disaccharide.
    expected_links = {"biotin": 1, "nglycan": 8, "oglycan": 6}[fixture]
    assert sum(len(m.connections) for m in models) == expected_links * (
        2 if duplicate else 1
    )
    source_bonds = {
        (int(a), int(b)): int(order) for a, b, order in array.bonds.as_array()
    }
    source_residues = struc.get_residue_starts(array, add_exclusive_stop=True)
    residues = np.repeat(np.arange(len(source_residues) - 1), np.diff(source_residues))
    seen = set()
    for model in models:
        atoms = model.source_atom_indices
        retained = atoms >= 0
        assert np.isnan(model.atom_array.coord).all()
        assert len(set(atoms[retained])) == int(retained.sum())
        assert not (set(atoms[retained]) & seen)
        seen.update(atoms[retained])
        np.testing.assert_array_equal(
            model.source_residue_indices[retained], residues[atoms[retained]]
        )
        for name in array.get_annotation_categories():
            np.testing.assert_array_equal(
                model.atom_array.get_annotation(name)[retained],
                array.get_annotation(name)[atoms[retained]],
            )
        if duplicate:
            assert np.all(atoms[retained] < len(original)) or np.all(
                atoms[retained] >= len(original)
            )
        for a, b, order in model.atom_array.bonds.as_array():
            if atoms[a] >= 0 and atoms[b] >= 0:
                key = tuple(sorted((int(atoms[a]), int(atoms[b]))))
                assert source_bonds[key] == int(order)
        source_to_local = {
            int(source): i for i, source in enumerate(atoms) if source >= 0
        }
        smi = _dimorphite_protonate_smiles(
            ligand_smiles_from_atom_array(model.atom_array, with_atom_map=True), ph=7.4
        )
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        assert len(Chem.GetMolFrags(mol)) == 1
        props = AllChem.MMFFGetMoleculeProperties(mol)
        assert props is not None
        mapping = {
            at.GetAtomMapNum() - 1: at.GetIdx()
            for at in mol.GetAtoms()
            if at.GetAtomicNum() > 1
        }
        assert set(mapping) == set(range(len(model.atom_array)))
        for a, b, order in model.connections:
            ia, ib = mapping[source_to_local[a]], mapping[source_to_local[b]]
            assert props.GetMMFFBondStretchParams(mol, ia, ib) is not None
            for center, other in ((ia, ib), (ib, ia)):
                for neighbor in mol.GetAtomWithIdx(center).GetNeighbors():
                    if neighbor.GetIdx() != other:
                        assert (
                            props.GetMMFFAngleBendParams(
                                mol, neighbor.GetIdx(), center, other
                            )
                            is not None
                        )
        # The neutral cap model of the biotin amide does not contain the
        # artificial ammonium present in the earlier uncapped diagnostic.
        if fixture == "biotin":
            assert Chem.GetFormalCharge(mol) == 0
    np.testing.assert_array_equal(array.coord[: len(original)], original.coord)


def test_two_lysine_anchors_keep_distinct_chemical_contexts(conjugate_input):
    fixture, original, db = conjugate_input
    if fixture != "biotin":
        pytest.skip("This constructed two-anchor case extends the biotin fixture")
    array = original.copy()
    (first_model,) = capped_conjugate_models(array, db.chemical)
    first_nz = next(
        i
        for link in first_model.connections
        for i in link[:2]
        if array.atom_name[i] == "NZ"
    )
    second_nz = next(
        int(i) for i in np.flatnonzero(array.atom_name == "NZ") if i != first_nz
    )
    carbon = int(
        np.flatnonzero((array.res_name == "BTN") & (array.atom_name == "C10"))[0]
    )
    array.bonds.add_bond(second_nz, carbon, struc.BondType.SINGLE)
    (model,) = capped_conjugate_models(array, db.chemical)
    assert len(model.connections) == 2
    assert len(set(model.source_residue_indices)) == 3
    smi = _dimorphite_protonate_smiles(
        ligand_smiles_from_atom_array(model.atom_array, with_atom_map=True), ph=7.4
    )
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    props = AllChem.MMFFGetMoleculeProperties(mol)
    assert props is not None
    source_to_local = {
        int(source): i
        for i, source in enumerate(model.source_atom_indices)
        if source >= 0
    }
    sites = [
        next(
            a for a in mol.GetAtoms() if a.GetAtomMapNum() == source_to_local[index] + 1
        )
        for index in (first_nz, second_nz)
    ]
    assert [sum(n.GetAtomicNum() == 1 for n in a.GetNeighbors()) for a in sites] == [
        1,
        2,
    ]
    assert props.GetMMFFAtomType(sites[0].GetIdx()) != props.GetMMFFAtomType(
        sites[1].GetIdx()
    )
    assert [a.GetFormalCharge() for a in sites] == [0, 1]


def test_group_models_are_independent_of_coordinates(conjugate_input):
    _, original, db = conjugate_input
    reference = capped_conjugate_models(original, db.chemical)
    changed = original.copy()
    changed.coord[:] = np.random.default_rng(37).normal(size=changed.coord.shape) * 100
    for coordinates_missing in (False, True):
        if coordinates_missing:
            changed.coord[:] = np.nan
        models = capped_conjugate_models(changed, db.chemical)
        assert len(models) == len(reference)
        for actual, expected in zip(models, reference):
            np.testing.assert_array_equal(
                actual.source_atom_indices, expected.source_atom_indices
            )
            np.testing.assert_array_equal(
                actual.source_residue_indices, expected.source_residue_indices
            )
            np.testing.assert_array_equal(
                actual.atom_array.bonds.as_array(), expected.atom_array.bonds.as_array()
            )
            assert actual.connections == expected.connections
            assert ligand_smiles_from_atom_array(
                actual.atom_array, with_atom_map=True
            ) == ligand_smiles_from_atom_array(expected.atom_array, with_atom_map=True)
