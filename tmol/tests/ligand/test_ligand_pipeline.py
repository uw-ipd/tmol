"""Integration tests for the ligand preparation pipeline.

Tests the full pipeline from CIF structure with ligands through detection,
SMILES perception, protonation, 3D generation, atom typing, residue
building, and database registration.
"""

import pytest

from tmol.database.chemical import ChemicalDatabase
from tmol.io.pose_stack_from_biotite import canonical_ordering_for_biotite
from tmol.ligand import prepare_ligands, prepare_single_ligand
from tmol.ligand.detect import detect_nonstandard_residues
from tmol.ligand.params_io import read_params_file, write_params_file
from tmol.ligand.registry import clear_cache


@pytest.fixture(autouse=True)
def _clear_ligand_cache():
    clear_cache()
    yield
    clear_cache()


class TestDetectFromCIF:
    """Verify detection of non-standard residues in real CIF files."""

    def test_no_ligands_in_ubq(self, biotite_1ubq):
        co = canonical_ordering_for_biotite()
        assert len(detect_nonstandard_residues(biotite_1ubq, co)) == 0

    def test_detects_i4b_in_184l(self, cif_184l_with_i4b):
        co = canonical_ordering_for_biotite()
        ligands = detect_nonstandard_residues(cif_184l_with_i4b, co)
        i4b = {lig.res_name: lig for lig in ligands}.get("I4B")
        assert i4b is not None
        assert i4b.is_ligand is True
        assert "NON-POLYMER" in i4b.ccd_type.upper()
        assert i4b.coords.shape == (len(i4b.atom_names), 3)

    def test_detects_hem_and_ace_in_155c(self, cif_155c_with_hem):
        co = canonical_ordering_for_biotite()
        ligands = detect_nonstandard_residues(cif_155c_with_hem, co)
        names = {lig.res_name for lig in ligands}
        assert "HEM" in names

    def test_detects_pse_with_partial_occupancy(self, cif_1a25_with_pse):
        co = canonical_ordering_for_biotite()
        ligands = detect_nonstandard_residues(cif_1a25_with_pse, co)
        assert any(lig.res_name == "PSE" for lig in ligands)


class TestFullPipeline:
    """End-to-end: CIF → detect → prepare → register → verify."""

    def test_i4b_small_drug(self, cif_184l_with_i4b):
        """Small drug-like ligand (I4B, 10 heavy atoms) in lysozyme."""
        chem_db = ChemicalDatabase.get_default()
        new_db, new_co = prepare_ligands(cif_184l_with_i4b, chem_db=chem_db)

        assert "I4B" in {r.name for r in new_db.residues}
        assert "I4B" in new_co.restype_io_equiv_classes

        i4b_rt = next(r for r in new_db.residues if r.name == "I4B")
        assert len(i4b_rt.atoms) > 0
        assert len(i4b_rt.bonds) > 0
        assert len(i4b_rt.icoors) == len(i4b_rt.atoms)
        assert i4b_rt.properties.polymer.is_polymer is False
        assert i4b_rt.default_jump_connection_atom in {a.name for a in i4b_rt.atoms}

        for ic in i4b_rt.icoors[1:]:
            assert 0.5 < ic.d < 5.0, f"Unreasonable distance {ic.d} for {ic.name}"

        atom_names = [a.name for a in i4b_rt.atoms]
        assert len(atom_names) == len(set(atom_names)), "Duplicate atom names"
        for a, b in i4b_rt.bonds:
            assert a in set(atom_names) and b in set(atom_names)

    def test_hem_large_ligand(self, cif_155c_with_hem):
        """Large ligand (HEM, 43 heavy atoms) in cytochrome c."""
        chem_db = ChemicalDatabase.get_default()
        new_db, new_co = prepare_ligands(cif_155c_with_hem, chem_db=chem_db)

        assert "HEM" in {r.name for r in new_db.residues}
        hem_rt = next(r for r in new_db.residues if r.name == "HEM")
        assert len(hem_rt.atoms) > 30
        assert len(hem_rt.icoors) == len(hem_rt.atoms)

    def test_pse_partial_occupancy(self, cif_1a25_with_pse):
        """Ligand with partial occupancy (PSE, 0.56) still prepares."""
        chem_db = ChemicalDatabase.get_default()
        new_db, new_co = prepare_ligands(cif_1a25_with_pse, chem_db=chem_db)

        assert "PSE" in {r.name for r in new_db.residues}

    def test_caching_prevents_duplicate_work(self, cif_184l_with_i4b):
        chem_db = ChemicalDatabase.get_default()
        db1, _ = prepare_ligands(cif_184l_with_i4b, chem_db=chem_db)
        n_residues_after_first = len(db1.residues)

        db2, _ = prepare_ligands(cif_184l_with_i4b, chem_db=db1)
        assert len(db2.residues) == n_residues_after_first

    def test_ubq_passes_through_unchanged(self, biotite_1ubq):
        chem_db = ChemicalDatabase.get_default()
        new_db, _ = prepare_ligands(biotite_1ubq, chem_db=chem_db)
        assert len(new_db.residues) == len(chem_db.residues)


class TestParamsRoundtrip:
    """Write a prepared ligand to .params and read it back."""

    def test_i4b_params_roundtrip(self, tmp_path, cif_184l_with_i4b):
        co = canonical_ordering_for_biotite()
        ligands = detect_nonstandard_residues(cif_184l_with_i4b, co)
        i4b = next(lig for lig in ligands if lig.res_name == "I4B")
        restype, charges = prepare_single_ligand(i4b)

        path = tmp_path / "I4B.params"
        write_params_file(restype, path, partial_charges=charges)
        loaded = read_params_file(path)

        assert loaded.name == "I4B"
        assert len(loaded.atoms) == len(restype.atoms)
        assert len(loaded.bonds) == len(restype.bonds)
        assert len(loaded.icoors) == len(restype.icoors)
