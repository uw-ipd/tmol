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


def test_group_topology_is_independent_of_coordinates(conjugate_input):
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


def test_capped_generation_preserves_observed_stereochemistry(conjugate_input):
    from tmol.ligand._connection_params import (
        _parameterized_model,
        _model_identity,
        generate_conjugate_connection_params,
    )
    from tmol.ligand._detect import nonstandard_residue_info_from_smiles_via_mol2
    from tmol.ligand._rdkit_mol import ligand_atom_array_to_rdkit_mol

    _, array, database = conjugate_input
    identities = []
    for reflected in (False, True):
        source = array.copy()
        if reflected:
            source.coord *= -1
        models = capped_conjugate_models(source, database.chemical)
        identities.append([_model_identity(model) for model in models])
        checked = 0
        for model in models:
            assert np.isnan(model.atom_array.coord).all()
            mol, _, mapping = _parameterized_model(model, ph=7.4)
            assert any(
                a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
                for a in mol.GetAtoms()
            )
            assert model.molecule.GetNumConformers() == 0
            smiles = Chem.MolToSmiles(Chem.RemoveHs(mol), ignoreAtomMapNumbers=True)
            info = nonstandard_residue_info_from_smiles_via_mol2(
                smiles, res_name="CONJ", protonate=False, seed=20250828
            )
            generated = ligand_atom_array_to_rdkit_mol(info, keep_hydrogens=True)
            heavy = [a.GetIdx() for a in generated.GetAtoms() if a.GetAtomicNum() > 1]
            generated_indices = dict(zip(info.source_atom_order, heavy, strict=True))
            local_indices = {index: local for local, index in mapping.items()}
            xyz = generated.GetConformer().GetPositions()
            for atom in mol.GetAtoms():
                if atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED:
                    continue
                neighbors = [
                    n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() > 1
                ]
                indices = [local_indices[i] for i in [atom.GetIdx(), *neighbors[:3]]]
                original = model.source_atom_indices[indices]
                if len(indices) != 4 or (original < 0).any():
                    continue
                before = source.coord[original].astype(float)
                if not np.isfinite(before).all():
                    continue
                after = xyz[[generated_indices[i] for i in indices]]
                # Signed local volumes independently compare the observed and
                # regenerated handedness using the same three named neighbours.
                assert (
                    np.linalg.det(before[1:] - before[0])
                    * np.linalg.det(after[1:] - after[0])
                    > 0
                )
                checked += 1
        assert checked > 0
    assert identities[0] != identities[1]
    with pytest.raises(ValueError, match="Incompatible conjugate chemistry"):
        generate_conjugate_connection_params(array + source, database)


@pytest.mark.parametrize(
    "fixture", ["6dmz_mod_d", "gamma_peptide_1gac", "na_rna_psu_1bzt"]
)
def test_declared_input_classes_resolve_prepared_polymer_types(fixture):
    from tmol.ligand._connection_params import generate_conjugate_connection_params

    array = atom_array_from_cif(data_path("ncaa_fixtures", fixture + ".cif"))
    database, ordering = prepare_ligands(array, seed=20250828)
    assert any(
        r.io_equiv_class in array.res_name and r.io_equiv_class != r.name
        for r in database.chemical.residues
        if r.name == r.base_name
    )
    assert set(array.res_name) <= set(ordering.restype_io_equiv_classes)
    # Ordinary polymer and curated disulfide links need no generated attachment
    # records, regardless of internal names such as DARG versus input DAR.
    assert generate_conjugate_connection_params(array, database) == ()
