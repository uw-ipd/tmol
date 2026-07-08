"""Targeted unit tests for ligand helper functions and error contracts.

These complement the end-to-end pipeline tests by exercising the smaller,
branch-heavy helpers directly: mol2-text parsers, SMILES/charge utilities, the
authoritative-charge mapper, the Rosetta ``.params`` reader, and the ``.tmol``
params loader's validation paths. The focus is on the documented fallback and
"fail loudly" behavior that the e2e happy paths never reach.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

import biotite.structure as struc

DATA = Path(__file__).parent.parent / "data"
GROUND_TRUTH = DATA / "ligand_test" / "ligand_ground_truth"


# --------------------------------------------------------------------------- #
# detect.py helpers
# --------------------------------------------------------------------------- #
class TestDetectHelpers:
    def test_strip_metals_removes_metal_atoms(self) -> None:
        from tmol.ligand.detect import _strip_metals

        mol = Chem.MolFromSmiles("[Fe]")
        assert mol.GetNumAtoms() == 1
        stripped = _strip_metals(mol)
        assert stripped.GetNumAtoms() == 0

    def test_strip_metals_no_op_without_metals(self) -> None:
        from tmol.ligand.detect import _strip_metals

        mol = Chem.MolFromSmiles("CCO")
        assert _strip_metals(mol).GetNumAtoms() == mol.GetNumAtoms()

    def test_rdkit_bond_to_biotite_type_orders(self) -> None:
        from tmol.ligand.detect import _rdkit_bond_to_biotite_type

        benzene = Chem.MolFromSmiles("c1ccccc1")
        aromatic_bond = benzene.GetBondWithIdx(0)
        assert _rdkit_bond_to_biotite_type(aromatic_bond) == int(
            struc.BondType.AROMATIC
        )

        single = Chem.MolFromSmiles("CC").GetBondWithIdx(0)
        assert _rdkit_bond_to_biotite_type(single) == int(struc.BondType.SINGLE)

        double = Chem.MolFromSmiles("C=C").GetBondWithIdx(0)
        assert _rdkit_bond_to_biotite_type(double) == int(struc.BondType.DOUBLE)

        triple = Chem.MolFromSmiles("C#C").GetBondWithIdx(0)
        assert _rdkit_bond_to_biotite_type(triple) == int(struc.BondType.TRIPLE)

    def test_rdkit_bond_to_biotite_type_quadruple_and_fallback(self) -> None:
        from tmol.ligand.detect import _rdkit_bond_to_biotite_type

        rw = Chem.RWMol()
        rw.AddAtom(Chem.Atom(6))
        rw.AddAtom(Chem.Atom(6))
        rw.AddBond(0, 1, Chem.BondType.QUADRUPLE)
        quad = rw.GetBondBetweenAtoms(0, 1)
        assert _rdkit_bond_to_biotite_type(quad) == int(struc.BondType.QUADRUPLE)

        rw2 = Chem.RWMol()
        rw2.AddAtom(Chem.Atom(7))
        rw2.AddAtom(Chem.Atom(6))
        rw2.AddBond(0, 1, Chem.BondType.DATIVE)
        dative = rw2.GetBondBetweenAtoms(0, 1)
        assert _rdkit_bond_to_biotite_type(dative) == int(struc.BondType.SINGLE)

    def test_infer_res_name_from_mol2(self) -> None:
        from tmol.ligand.detect import _infer_res_name_from_mol2

        mol = Chem.MolFromSmiles("CC")
        # No Tripos substructure name -> fallback used.
        assert _infer_res_name_from_mol2(mol, "FALL") == "FALL"
        mol.GetAtomWithIdx(0).SetProp("_TriposSubstName", "LIG")
        assert _infer_res_name_from_mol2(mol, "FALL") == "LIG"

    def test_source_subtype_from_mol2_atom_type(self) -> None:
        from tmol.ligand.detect import _source_subtype_from_mol2_atom_type

        assert _source_subtype_from_mol2_atom_type("C.ar") == "ar"
        assert _source_subtype_from_mol2_atom_type("C") == "?"
        assert _source_subtype_from_mol2_atom_type("") == "?"

    def test_mol2_charge_model_from_text(self) -> None:
        from tmol.ligand.detect import _mol2_charge_model_from_text

        good = (
            "@<TRIPOS>MOLECULE\n"
            "LIG\n"
            " 3 2 1 0 0\n"
            "SMALL\n"
            "GASTEIGER\n"
            "@<TRIPOS>ATOM\n"
        )
        assert _mol2_charge_model_from_text(good) == "GASTEIGER"

        # Section ends before the 4th line -> empty.
        truncated = "@<TRIPOS>MOLECULE\nLIG\n 3 2\n@<TRIPOS>ATOM\n"
        assert _mol2_charge_model_from_text(truncated) == ""

        # No molecule block at all -> empty.
        assert _mol2_charge_model_from_text("nothing here\n") == ""

    def test_mol2_single_bond_ids(self) -> None:
        from tmol.ligand.detect import _mol2_single_bond_ids

        text = (
            "@<TRIPOS>BOND\n"
            "1 1 2 1\n"  # single -> included
            "2 2 3 2\n"  # double -> excluded
            "3 3 4 ar\n"  # aromatic -> excluded
            "bad line\n"  # too few tokens -> skipped
            "5 x y 1\n"  # non-integer atom ids -> skipped
        )
        bonds = _mol2_single_bond_ids(text)
        assert frozenset((1, 2)) in bonds
        assert frozenset((2, 3)) not in bonds
        assert len(bonds) == 1

    def test_charge_model_is_authoritative(self) -> None:
        from tmol.ligand.detect import _charge_model_is_authoritative

        assert _charge_model_is_authoritative("MMFF94") is True
        assert _charge_model_is_authoritative("GASTEIGER") is False
        assert _charge_model_is_authoritative("") is False

    def test_normalize_radical_oxygens(self) -> None:
        from tmol.ligand.detect import _normalize_radical_oxygens

        # Bare radical oxygen on a carboxyl carbon -> becomes [O-].
        fixed = _normalize_radical_oxygens("CC(=O)[O]")
        mol = Chem.MolFromSmiles(fixed)
        assert mol is not None
        assert any(a.GetFormalCharge() == -1 for a in mol.GetAtoms())

        # Nothing to change -> returned unchanged.
        assert _normalize_radical_oxygens("CCO") == "CCO"
        # Unparseable input -> returned unchanged.
        assert _normalize_radical_oxygens("not a smiles!!!") == "not a smiles!!!"

    def test_dimorphite_protonate_smiles(self) -> None:
        from tmol.ligand.detect import _dimorphite_protonate_smiles

        # Carboxylic acid deprotonates near physiological pH.
        out = _dimorphite_protonate_smiles("CC(=O)O", ph=7.4)
        assert Chem.MolFromSmiles(out) is not None
        # Unparseable input is returned unchanged.
        assert _dimorphite_protonate_smiles("xxx!!!") == "xxx!!!"


# --------------------------------------------------------------------------- #
# structure_to_smiles.py helpers
# --------------------------------------------------------------------------- #
class TestStructureToSmiles:
    def _array(self):
        import biotite.structure.io.pdbx as pdbx

        fixture = DATA / "ligand_cif_fixtures" / "vww.bonds_present.cif"
        cif = pdbx.CIFFile.read(str(fixture))
        arr = pdbx.get_structure(cif, model=1, include_bonds=True)
        if isinstance(arr, struc.AtomArrayStack):
            arr = arr[0]
        return arr

    def test_system_charge_explicit_overrides_annotation(self) -> None:
        from tmol.ligand.structure_to_smiles import _system_charge

        arr = self._array()
        assert _system_charge(arr, 3) == 3

    def test_mol_to_smiles_returns_none_on_failure(self, monkeypatch) -> None:
        import tmol.ligand.structure_to_smiles as mod

        def _boom(*a, **k):
            raise RuntimeError("boom")

        monkeypatch.setattr(mod.Chem, "MolToSmiles", _boom)
        assert mod._mol_to_smiles(Chem.MolFromSmiles("CCO")) is None

    def test_smiles_from_atom_array_raises_when_no_candidates(
        self, monkeypatch
    ) -> None:
        import tmol.ligand.structure_to_smiles as mod

        monkeypatch.setattr(
            mod, "ligand_smiles_candidates_from_atom_array", lambda *a, **k: []
        )
        with pytest.raises(ValueError, match="Could not derive a SMILES"):
            mod.ligand_smiles_from_atom_array(self._array(), res_name="LIG")


# --------------------------------------------------------------------------- #
# mol3d.py
# --------------------------------------------------------------------------- #
class TestAuthoritativeCharges:
    def test_maps_by_index(self) -> None:
        from tmol.ligand.mol3d import authoritative_charges_by_index

        mol = Chem.MolFromSmiles("CCO")
        names = ["C1", "C2", "O1"]
        charges = {"C1": 0.1, "C2": -0.1, "O1": -0.3}
        by_index = authoritative_charges_by_index(names, charges, mol)
        assert by_index == {0: 0.1, 1: -0.1, 2: -0.3}

    def test_raises_without_charges(self) -> None:
        from tmol.ligand.mol3d import authoritative_charges_by_index

        mol = Chem.MolFromSmiles("CCO")
        with pytest.raises(ValueError, match="no authoritative partial charges"):
            authoritative_charges_by_index(["C1", "C2", "O1"], None, mol)

    def test_raises_on_count_mismatch(self) -> None:
        from tmol.ligand.mol3d import authoritative_charges_by_index

        mol = Chem.MolFromSmiles("CCO")
        with pytest.raises(ValueError, match="atom-count mismatch"):
            authoritative_charges_by_index(["C1", "C2"], {"C1": 0.0}, mol)

    def test_raises_on_missing_atom(self) -> None:
        from tmol.ligand.mol3d import authoritative_charges_by_index

        mol = Chem.MolFromSmiles("CCO")
        with pytest.raises(ValueError, match="missing for atoms"):
            authoritative_charges_by_index(
                ["C1", "C2", "O1"], {"C1": 0.1, "C2": -0.1}, mol, ligand_name="LIG"
            )


# --------------------------------------------------------------------------- #
# equivalence.py element-name helper
# --------------------------------------------------------------------------- #
class TestEquivalenceElementFromName:
    def test_infer_element_from_name(self) -> None:
        from tmol.ligand.equivalence import _infer_element_from_name

        assert _infer_element_from_name("CB") == "C"
        assert _infer_element_from_name("CA") == "Ca"
        assert _infer_element_from_name("CL") == "Cl"
        assert _infer_element_from_name("C") == "C"
        assert _infer_element_from_name("") == "?"
        assert _infer_element_from_name("1") == "?"


# --------------------------------------------------------------------------- #
# rdkit_mol.py error contracts
# --------------------------------------------------------------------------- #
class TestLigandAtomArrayToRdkitMol:
    def _info(self, arr):
        from tmol.ligand.detect import NonStandardResidueInfo

        return NonStandardResidueInfo(
            res_name="LG1",
            ccd_type="UNKNOWN",
            atom_names=tuple(str(n) for n in arr.atom_name),
            elements=tuple(str(e) for e in arr.element),
            coords=arr.coord.copy(),
            atom_array=arr,
        )

    def _carbon_array(self, n: int):
        arr = struc.AtomArray(n)
        arr.coord = np.zeros((n, 3), dtype=np.float32)
        arr.atom_name = np.array([f"C{i}" for i in range(n)], dtype="U4")
        arr.element = np.array(["C"] * n, dtype="U4")
        return arr

    def test_empty_array_raises(self) -> None:
        from tmol.ligand.rdkit_mol import ligand_atom_array_to_rdkit_mol

        arr = struc.AtomArray(0)
        arr.coord = np.zeros((0, 3), dtype=np.float32)
        arr.atom_name = np.array([], dtype="U4")
        arr.element = np.array([], dtype="U4")
        with pytest.raises(ValueError, match="empty atom array"):
            ligand_atom_array_to_rdkit_mol(self._info(arr))

    def test_no_bonds_raises(self) -> None:
        from tmol.ligand.rdkit_mol import ligand_atom_array_to_rdkit_mol

        arr = self._carbon_array(2)
        with pytest.raises(ValueError, match="bond inference is unsupported"):
            ligand_atom_array_to_rdkit_mol(self._info(arr))

    def test_topology_only_single_bonds_raises(self) -> None:
        from tmol.ligand.rdkit_mol import ligand_atom_array_to_rdkit_mol

        arr = self._carbon_array(2)
        arr.bonds = struc.BondList(
            2, np.array([[0, 1, int(struc.BondType.SINGLE)]], dtype=np.uint32)
        )
        with pytest.raises(ValueError, match="topology-only SINGLE bonds"):
            ligand_atom_array_to_rdkit_mol(self._info(arr))


# --------------------------------------------------------------------------- #
# preparation.py helper contracts
# --------------------------------------------------------------------------- #
class TestPreparationHelpers:
    def test_prepare_single_ligand_requires_protonation_and_charges(self) -> None:
        from tmol.ligand.preparation import (
            _ligand_info_from_cif,
            prepare_single_ligand,
        )

        cif = DATA / "ligand_cif_fixtures" / "vww.bonds_present.cif"
        info = _ligand_info_from_cif(str(cif), None)
        # Raw CIF info has no explicit-H / authoritative charges -> rejected.
        with pytest.raises(ValueError, match="requires a ligand that already"):
            prepare_single_ligand(info)

    def test_residue_covers_cif_heavy_atoms_empty_is_true(self) -> None:
        from tmol.ligand.preparation import _residue_covers_cif_heavy_atoms

        # Empty CIF heavy-atom set short-circuits to True without inspecting prep.
        assert _residue_covers_cif_heavy_atoms(object(), set()) is True

    @staticmethod
    def _info(atom_names, elements):
        from tmol.ligand.detect import NonStandardResidueInfo

        return NonStandardResidueInfo(
            res_name="LIG",
            ccd_type="NON-POLYMER",
            atom_names=tuple(atom_names),
            elements=tuple(elements),
            coords=np.zeros((len(atom_names), 3), dtype=float),
            atom_array=None,
        )

    @staticmethod
    def _at(atom_name, element, index):
        from tmol.ligand.atom_typing import AtomTypeAssignment

        return AtomTypeAssignment(
            atom_name=atom_name, atom_type="X", element=element, index=index
        )

    def test_rename_by_index_bails_on_name_element_length_mismatch(self) -> None:
        from tmol.ligand.preparation import _rename_atoms_to_cif_by_index

        info = self._info(["C1", "C2"], ["C"])  # mismatched lengths
        assert _rename_atoms_to_cif_by_index([self._at("C1", "C", 0)], info) is None

    def test_rename_by_index_bails_on_out_of_range_index(self) -> None:
        from tmol.ligand.preparation import _rename_atoms_to_cif_by_index

        info = self._info(["C1"], ["C"])
        # Heavy atom whose index exceeds the CIF atom list -> bail to graph path.
        ats = [self._at("C1", "C", 0), self._at("C2", "C", 5)]
        assert _rename_atoms_to_cif_by_index(ats, info) is None

    def test_rename_by_index_bails_on_duplicate_names(self) -> None:
        from tmol.ligand.preparation import _rename_atoms_to_cif_by_index

        info = self._info(["C1", "C1"], ["C", "C"])  # duplicate target names
        ats = [self._at("C1", "C", 0), self._at("Cx", "C", 1)]
        assert _rename_atoms_to_cif_by_index(ats, info) is None

    def test_rename_by_index_bails_on_element_mismatch(self) -> None:
        from tmol.ligand.preparation import _rename_atoms_to_cif_by_index

        info = self._info(["N1"], ["N"])
        assert _rename_atoms_to_cif_by_index([self._at("C1", "C", 0)], info) is None


# --------------------------------------------------------------------------- #
# params_io.py reader + format guard
# --------------------------------------------------------------------------- #
class TestParamsIo:
    def test_read_rosetta_params_file(self) -> None:
        from tmol.ligand.params_io import read_params_file

        params = sorted((GROUND_TRUTH / "params").glob("*.params"))
        assert params, "expected ground-truth .params fixtures"
        rt = read_params_file(params[0])
        assert len(rt.atoms) > 0
        assert len(rt.bonds) > 0

    def test_write_params_file_rejects_unknown_format(self) -> None:
        from tmol.ligand.params_io import write_params_file

        with pytest.raises(ValueError, match="unknown params format"):
            write_params_file(object(), "/tmp/ignored.out", format="bogus")
