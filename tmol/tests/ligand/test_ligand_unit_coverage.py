"""Targeted unit tests for ligand helper functions and error contracts.

These complement the end-to-end pipeline tests by exercising the smaller,
branch-heavy helpers directly: mol2-text parsers, SMILES/charge utilities, the
authoritative-charge mapper, the Rosetta ``.params`` reader, and the ``.tmol``
params loader's validation paths. The focus is on the documented fallback and
"fail loudly" behavior that the e2e happy paths never reach.
"""

from __future__ import annotations

from tmol.tests.data import data_path

import numpy as np
import pytest
from rdkit import Chem

import biotite.structure as struc

DATA = data_path()
GROUND_TRUTH = DATA / "ligand_test" / "ligand_ground_truth"
NEUTRALIZED_AMMONIUM_MOL2 = (
    "@<TRIPOS>MOLECULE\nammonium\n5 4 0 0 0\nSMALL\n{charge_model}\n"
    "@<TRIPOS>ATOM\n"
    "1 N 0 0 0 N.4 1 NH4 -0.4\n"
    "2 H1 1 0 0 H 1 NH4 0.1\n"
    "3 H2 -1 0 0 H 1 NH4 0.1\n"
    "4 H3 0 1 0 H 1 NH4 0.1\n"
    "5 H4 0 0 1 H 1 NH4 0.1\n"
    "@<TRIPOS>UNITY_ATOM_ATTR\n1 1\ncharge 1\n"
    "@<TRIPOS>BOND\n1 1 2 1\n2 1 3 1\n3 1 4 1\n4 1 5 1\n"
)


# --------------------------------------------------------------------------- #
# detect.py helpers
# --------------------------------------------------------------------------- #


def delocalized_center(neighbors, center, substituents):
    """Build one acyclic center whose neighbour bonds are Tripos ``ar``.

    ``neighbors`` is ``(atomic_number, n_hydrogens, n_carbon_substituents)`` per
    delocalized neighbour, ordered nearest-first: among the neighbours that may
    take it, the double bond goes to the nearest.
    """
    mol = Chem.RWMol()
    center_index = mol.AddAtom(Chem.Atom(center))
    delocalized_bonds = set()
    coordinates = {center_index: (0.0, 0.0, 0.0)}
    for rank, (atomic_number, n_hydrogens, n_substituents) in enumerate(neighbors):
        index = mol.AddAtom(Chem.Atom(atomic_number))
        coordinates[index] = (1.25 + 0.2 * rank, 0.0, 0.0)
        mol.AddBond(center_index, index, Chem.BondType.AROMATIC)
        delocalized_bonds.add(frozenset((center_index, index)))
        for _ in range(n_hydrogens):
            mol.AddBond(index, mol.AddAtom(Chem.Atom(1)), Chem.BondType.SINGLE)
        for _ in range(n_substituents):
            mol.AddBond(index, mol.AddAtom(Chem.Atom(6)), Chem.BondType.SINGLE)
    for _ in range(substituents):
        mol.AddBond(center_index, mol.AddAtom(Chem.Atom(6)), Chem.BondType.SINGLE)
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for index in range(mol.GetNumAtoms()):
        conformer.SetAtomPosition(index, coordinates.get(index, (0.0, 0.0, 5.0)))
    mol.AddConformer(conformer)
    return mol.GetMol(), delocalized_bonds


class TestDetectHelpers:
    def test_strip_metals_removes_metal_atoms(self) -> None:
        from tmol.ligand import _strip_metals

        mol = Chem.MolFromSmiles("[Fe]")
        assert mol.GetNumAtoms() == 1
        stripped = _strip_metals(mol)
        assert stripped.GetNumAtoms() == 0

    def test_strip_metals_no_op_without_metals(self) -> None:
        from tmol.ligand import _strip_metals

        mol = Chem.MolFromSmiles("CCO")
        assert _strip_metals(mol).GetNumAtoms() == mol.GetNumAtoms()

    def test_rdkit_bond_to_biotite_type_orders(self) -> None:
        from tmol.ligand import _rdkit_bond_to_biotite_type

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
        from tmol.ligand import _rdkit_bond_to_biotite_type

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
        assert _rdkit_bond_to_biotite_type(dative) == int(struc.BondType.ANY)

    def test_infer_res_name_from_mol2(self) -> None:
        from tmol.ligand import _infer_res_name_from_mol2

        mol = Chem.MolFromSmiles("CC")
        # No Tripos substructure name -> fallback used.
        assert _infer_res_name_from_mol2(mol, "FALL") == "FALL"
        mol.GetAtomWithIdx(0).SetProp("_TriposSubstName", "LIG")
        assert _infer_res_name_from_mol2(mol, "FALL") == "LIG"

    def test_source_subtype_from_mol2_atom_type(self) -> None:
        from tmol.ligand import _source_subtype_from_mol2_atom_type

        assert _source_subtype_from_mol2_atom_type("C.ar") == "ar"
        assert _source_subtype_from_mol2_atom_type("C") == "?"
        assert _source_subtype_from_mol2_atom_type("") == "?"

    def test_mol2_charge_model_from_text(self) -> None:
        from tmol.ligand import _mol2_charge_model_from_text

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
        from tmol.ligand import _mol2_single_bond_ids

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

    @pytest.mark.parametrize(
        "model, expected",
        [
            ("MMFF94", True),
            ("MMFF94_CHARGES", True),
            ("AM1-BCC", True),
            ("USER_CHARGES", True),
            ("NO_CHARGES", False),
            ("DEL_RE", False),
            ("GASTEIGER", False),
            ("GAST_HUCK", False),
            ("HUCKEL", False),
            ("PULLMAN", False),
            ("MULLIKEN_CHARGES", False),
            ("GAUSS80_CHARGES", False),
            ("AMPAC_CHARGES", False),
        ],
    )
    def test_charge_model_is_authoritative(self, model, expected) -> None:
        from tmol.ligand import _charge_model_is_authoritative

        assert _charge_model_is_authoritative(model) is expected

    @pytest.mark.parametrize(
        "center_atomic_number, n_substituents",
        [(6, 0), (15, 1), (16, 1)],
        ids=["carbonate", "phosphate", "sulfonate"],
    )
    def test_infer_oxyacid_bonds_is_generic(
        self, center_atomic_number, n_substituents
    ) -> None:
        from tmol.ligand._detect import _infer_oxyacid_bonds

        mol = Chem.RWMol()
        center = mol.AddAtom(Chem.Atom(center_atomic_number))
        delocalized_bonds = set()
        declared_charges = {}
        oxygen_indices = []
        for charge in (0, -1, -1):
            oxygen = Chem.Atom(8)
            oxygen.SetFormalCharge(charge)
            oxygen.SetProp("_TriposAtomType", "O.co2")
            oxygen_index = mol.AddAtom(oxygen)
            mol.AddBond(center, oxygen_index, Chem.BondType.SINGLE)
            oxygen_indices.append(oxygen_index)
            delocalized_bonds.add(frozenset((center, oxygen_index)))
            if charge:
                declared_charges[oxygen_index] = charge
        for _ in range(n_substituents):
            substituent = mol.AddAtom(Chem.Atom(6))
            mol.AddBond(center, substituent, Chem.BondType.SINGLE)

        _infer_oxyacid_bonds(mol, declared_charges, delocalized_bonds)

        orders = [
            mol.GetBondBetweenAtoms(center, oxygen).GetBondTypeAsDouble()
            for oxygen in oxygen_indices
        ]
        assert sorted(orders) == [1.0, 1.0, 2.0]

    @pytest.mark.parametrize(
        "neighbors, center, substituents, expected",
        [
            (((7, 1, 0), (7, 2, 0)), 6, 1, "CC(=N)N"),
            (((7, 0, 0), (7, 0, 0)), 6, 1, "CC(=N)N"),
            (((7, 1, 0), (7, 2, 0), (7, 2, 0)), 6, 0, "N=C(N)N"),
            (((7, 1, 1), (7, 2, 0), (7, 2, 0)), 6, 0, "C[NH+]=C(N)N"),
            (((7, 2, 0), (7, 1, 1), (7, 2, 0)), 6, 0, "CNC(N)=[NH2+]"),
            (((8, 0, 0), (8, 1, 0)), 6, 1, "CC(=O)O"),
            (((8, 1, 0), (8, 0, 0)), 6, 1, "CC(=O)O"),
            (((8, 0, 0), (8, 0, 0)), 6, 1, "CC(=O)[O-]"),
            (((8, 0, 0), (8, 0, 0), (8, 0, 1)), 15, 1, "COP(C)(=O)[O-]"),
            (((8, 0, 0), (8, 0, 0)), 7, 1, "C[N+](=O)[O-]"),
        ],
        ids=[
            "amidine",
            "amidine_without_hydrogens",
            "guanidine",
            "guanidinium_substituted_nitrogen_nearest",
            "guanidinium_substituted_nitrogen_second",
            "carboxylic_acid",
            "carboxylic_acid_hydroxyl_nearest",
            "carboxylate",
            "phosphonate_ester",
            "nitro",
        ],
    )
    def test_localization_places_the_double_bond_without_reprotonating(
        self, neighbors, center, substituents, expected
    ) -> None:
        """The double bond goes where it does not force a neighbour positive.

        A hydroxyl oxygen and a phosphodiester's bridging oxygen would both have
        to become cations to take it, so they keep their single bond. A
        guanidinium has no such option -- every nitrogen is the cation in some
        Kekule form -- and refusing it there left the center a bond short: a
        carbanion whose nitrogens come back with an extra hydrogen each.
        """
        from tmol.ligand._detect import _infer_oxyacid_bonds

        molecule, delocalized_bonds = delocalized_center(
            neighbors, center, substituents
        )

        _infer_oxyacid_bonds(molecule, {}, delocalized_bonds)

        Chem.SanitizeMol(molecule)
        assert Chem.MolToSmiles(Chem.RemoveHs(molecule)) == expected

    @pytest.mark.parametrize(
        "neighbors, center, substituents, declared, expected, net",
        [
            (((7, 1, 1), (7, 2, 0), (7, 2, 0)), 6, 0, {4: 1}, "CNC(N)=[NH2+]", 1),
            (((7, 2, 0), (7, 2, 0)), 6, 1, {4: 1}, "CC(N)=[NH2+]", 1),
            (((8, 0, 0), (8, 0, 0)), 6, 1, {2: -1}, "CC(=O)[O-]", -1),
            (((8, 0, 0), (8, 0, 0), (8, 0, 0)), 16, 1, {3: -1}, "CS(=O)(=O)[O-]", -1),
        ],
        ids=["guanidinium", "amidinium", "carboxylate", "mesylate"],
    )
    def test_a_declared_charge_and_an_inferred_one_agree(
        self, neighbors, center, substituents, declared, expected, net
    ) -> None:
        """What the file says settles where the double bond goes, not what it adds.

        A guanidinium's +1 nitrogen is the one holding the double bond; a
        carboxylate's -1 oxygen is the one that is not. Reading the declaration
        as an extra charge instead of as a placement gave the guanidinium a
        second cation -- a net the caller could not catch, since it compares
        against the declared charges plus the ones localization invented.
        """
        from tmol.ligand._detect import _infer_oxyacid_bonds

        molecule, delocalized_bonds = delocalized_center(
            neighbors, center, substituents
        )
        for index, charge in declared.items():
            molecule.GetAtomWithIdx(index).SetFormalCharge(charge)

        _infer_oxyacid_bonds(molecule, dict(declared), delocalized_bonds)

        Chem.SanitizeMol(molecule)
        assert Chem.MolToSmiles(Chem.RemoveHs(molecule)) == expected
        assert Chem.GetFormalCharge(molecule) == net

    def test_contradictory_declared_charges_fail_rather_than_invent_a_center(
        self,
    ) -> None:
        """Both carboxylate oxygens cannot be the anion; refuse instead of balancing.

        Handing the double bond only to neighbours the file left uncharged meant
        there were none here, so no double bond was written at all and the
        carbon absorbed the difference as a charge -- which localization then
        recorded, moving the expected net with it so nothing downstream noticed.
        Leaving it unsanitizable routes the molecule to the fallback reader.
        """
        from tmol.ligand._detect import _infer_oxyacid_bonds

        molecule, delocalized_bonds = delocalized_center(((8, 0, 0), (8, 0, 0)), 6, 1)
        declared = {1: -1, 2: -1}
        for index, charge in declared.items():
            molecule.GetAtomWithIdx(index).SetFormalCharge(charge)

        _infer_oxyacid_bonds(molecule, dict(declared), delocalized_bonds)

        assert not any(
            atom.GetSymbol() == "C" and atom.GetFormalCharge()
            for atom in molecule.GetAtoms()
        )
        with pytest.raises(Chem.AtomValenceException):
            Chem.SanitizeMol(molecule)

    def test_authoritative_neutralized_charges_need_not_match_formal_charge(
        self,
    ) -> None:
        from tmol.ligand import nonstandard_residue_info_from_mol2_block

        mol2 = NEUTRALIZED_AMMONIUM_MOL2.format(charge_model="USER_CHARGES")
        info = nonstandard_residue_info_from_mol2_block(mol2)

        assert info.atom_array.charge.sum() == 1
        assert sum(info.partial_charges.values()) == pytest.approx(0)
        assert info.skip_protonation

    @pytest.mark.parametrize(
        "charge_model",
        [
            "NO_CHARGES",
            "DEL_RE",
            "GASTEIGER",
            "GAST_HUCK",
            "HUCKEL",
            "PULLMAN",
            "MULLIKEN_CHARGES",
            "GAUSS80_CHARGES",
            "AMPAC_CHARGES",
        ],
    )
    def test_non_authoritative_charge_models_regenerate_in_auto(
        self, monkeypatch, tmp_path, charge_model
    ) -> None:
        import tmol.ligand._preparation as preparation

        path = tmp_path / "ammonium.mol2"
        path.write_text(NEUTRALIZED_AMMONIUM_MOL2.format(charge_model=charge_model))
        with pytest.raises(ValueError, match="authoritative partial charges"):
            preparation._prepare_mol2(path, mode="keep")
        monkeypatch.setattr(
            preparation,
            "_prepare_ligand_via_smiles",
            lambda _info, **_kwargs: "regenerated",
        )

        assert preparation._prepare_mol2(path, mode="auto") == "regenerated"

    @pytest.mark.parametrize(
        "charges, n_atoms, succeeds",
        [
            ([0.17, -0.03], 2, True),
            ([np.nan, 0.0], 2, False),
            ([0.0], 2, False),
        ],
        ids=["nonintegral-net", "nonfinite", "wrong-shape"],
    )
    def test_openbabel_charge_validation(self, charges, n_atoms, succeeds) -> None:
        from tmol.ligand._openbabel_compat import _compute_charges_with_fallback

        class Atom:
            def __init__(self, charge):
                self.charge = charge

            def GetPartialCharge(self):
                return self.charge

        class Molecule:
            def __init__(self):
                self.atoms = [Atom(charge) for charge in charges]
                self.provenance = []

            def DeleteData(self, _name):
                pass

            def NumAtoms(self):
                return n_atoms

            def GetTotalCharge(self):
                return 0

            def CloneData(self, value):
                self.provenance.append(value)

        class ChargeModel:
            def ComputeCharges(self, _mol):
                return True

        class ChargeModels:
            @staticmethod
            def FindType(_name):
                return ChargeModel()

        class PairData:
            def SetAttribute(self, value):
                self.attribute = value

            def SetValue(self, value):
                self.value = value

        class OpenBabel:
            OBChargeModel = ChargeModels
            OBPairData = PairData

            @staticmethod
            def OBMolAtomIter(mol):
                return iter(mol.atoms)

        pymol = type("PyMol", (), {"OBMol": Molecule()})()
        if succeeds:
            assert (
                _compute_charges_with_fallback(OpenBabel, pymol, "mmff94", "[NH4+]")
                == "mmff94"
            )
            assert len(pymol.OBMol.provenance) == 1
        else:
            with pytest.raises(ValueError, match="could not compute"):
                _compute_charges_with_fallback(OpenBabel, pymol, "mmff94", "[NH4+]")

    @pytest.mark.parametrize(
        "mode, prepared_input, expected",
        [
            ("keep", True, "kept"),
            ("auto", True, "kept"),
            ("auto", False, "regenerated"),
            ("regenerate", True, "regenerated"),
            ("regenerate", False, "regenerated"),
        ],
    )
    def test_mol2_preparation_mode_dispatch(
        self, monkeypatch, mode, prepared_input, expected
    ) -> None:
        from types import SimpleNamespace

        import tmol.ligand._detect as detect
        import tmol.ligand._preparation as preparation

        info = SimpleNamespace(skip_protonation=prepared_input)
        monkeypatch.setattr(
            detect, "nonstandard_residue_info_from_mol2", lambda *_a, **_k: info
        )
        monkeypatch.setattr(preparation, "prepare_single_ligand", lambda _info: "kept")
        monkeypatch.setattr(
            preparation,
            "_prepare_ligand_via_smiles",
            lambda _info, **_k: "regenerated",
        )

        assert preparation._prepare_mol2("lig.mol2", mode=mode) == expected

    def test_normalize_radical_oxygens(self) -> None:
        from tmol.ligand import _normalize_radical_oxygens

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
        from tmol.ligand import _dimorphite_protonate_smiles

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
        from tmol.io import atom_array_from_cif

        fixture = DATA / "ligand_cif_fixtures" / "vww.bonds_present.cif"
        # a single-ligand file supplying a whole molecule under a code of its own
        return atom_array_from_cif(fixture, use_ccd=False)

    def test_mol_to_smiles_returns_none_on_failure(self, monkeypatch) -> None:
        import tmol.ligand._structure_to_smiles as mod

        def _boom(*a, **k):
            raise RuntimeError("boom")

        monkeypatch.setattr(mod.Chem, "MolToSmiles", _boom)
        assert mod._mol_to_smiles(Chem.MolFromSmiles("CCO")) is None

    def test_smiles_from_atom_array_raises_when_no_smiles(self, monkeypatch) -> None:
        import tmol.ligand._structure_to_smiles as mod

        # Bonds are present, but SMILES generation yields nothing.
        monkeypatch.setattr(mod, "_mol_to_smiles", lambda *a, **k: None)
        with pytest.raises(ValueError, match="Could not derive a SMILES"):
            mod.ligand_smiles_from_atom_array(self._array(), res_name="LIG")


# --------------------------------------------------------------------------- #
# mol3d.py
# --------------------------------------------------------------------------- #
class TestAuthoritativeCharges:
    def test_maps_by_index(self) -> None:
        from tmol.ligand import authoritative_charges_by_index

        mol = Chem.MolFromSmiles("CCO")
        names = ["C1", "C2", "O1"]
        charges = {"C1": 0.1, "C2": -0.1, "O1": -0.3}
        by_index = authoritative_charges_by_index(names, charges, mol)
        assert by_index == {0: 0.1, 1: -0.1, 2: -0.3}

    def test_raises_without_charges(self) -> None:
        from tmol.ligand import authoritative_charges_by_index

        mol = Chem.MolFromSmiles("CCO")
        with pytest.raises(ValueError, match="no authoritative partial charges"):
            authoritative_charges_by_index(["C1", "C2", "O1"], None, mol)

    def test_raises_on_count_mismatch(self) -> None:
        from tmol.ligand import authoritative_charges_by_index

        mol = Chem.MolFromSmiles("CCO")
        with pytest.raises(ValueError, match="atom-count mismatch"):
            authoritative_charges_by_index(["C1", "C2"], {"C1": 0.0}, mol)

    def test_raises_on_missing_atom(self) -> None:
        from tmol.ligand import authoritative_charges_by_index

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
        from tmol.tests.ligand import _infer_element_from_name

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
        from tmol.ligand import NonStandardResidueInfo

        return NonStandardResidueInfo(
            res_name="LG1",
            component_type="UNKNOWN",
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
        from tmol.ligand import ligand_atom_array_to_rdkit_mol

        arr = struc.AtomArray(0)
        arr.coord = np.zeros((0, 3), dtype=np.float32)
        arr.atom_name = np.array([], dtype="U4")
        arr.element = np.array([], dtype="U4")
        with pytest.raises(ValueError, match="empty atom array"):
            ligand_atom_array_to_rdkit_mol(self._info(arr))

    def test_no_bonds_raises(self) -> None:
        from tmol.ligand import ligand_atom_array_to_rdkit_mol

        arr = self._carbon_array(2)
        with pytest.raises(ValueError, match="bond inference is unsupported"):
            ligand_atom_array_to_rdkit_mol(self._info(arr))

    def test_topology_only_any_bonds_raises(self) -> None:
        from tmol.ligand import ligand_atom_array_to_rdkit_mol

        arr = self._carbon_array(2)
        # BondType.ANY marks bonds whose order was perceived from geometry
        #   (e.g. PDB input) ... ensure this fails
        arr.bonds = struc.BondList(
            2, np.array([[0, 1, int(struc.BondType.ANY)]], dtype=np.uint32)
        )
        with pytest.raises(ValueError, match="topology-only bonds"):
            ligand_atom_array_to_rdkit_mol(self._info(arr))


# --------------------------------------------------------------------------- #
# preparation.py helper contracts
# --------------------------------------------------------------------------- #
class TestPreparationHelpers:
    def test_prepare_single_ligand_requires_protonation_and_charges(self) -> None:
        from tmol.ligand import (
            _ligand_info_from_cif,
            prepare_single_ligand,
        )

        cif = DATA / "ligand_cif_fixtures" / "vww.bonds_present.cif"
        info = _ligand_info_from_cif(str(cif), None)
        # Raw CIF info has no explicit-H / authoritative charges -> rejected.
        with pytest.raises(ValueError, match="requires a ligand that already"):
            prepare_single_ligand(info)

    def test_residue_covers_cif_heavy_atoms_empty_is_true(self) -> None:
        from tmol.ligand import _residue_covers_cif_heavy_atoms

        # Empty CIF heavy-atom set short-circuits to True without inspecting prep.
        assert _residue_covers_cif_heavy_atoms(object(), set()) is True
