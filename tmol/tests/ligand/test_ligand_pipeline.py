"""Integration tests for the ligand preparation pipeline.

Tests the full pipeline from CIF structure with ligands through detection,
SMILES perception, protonation, 3D generation, atom typing, residue
building, and database registration. Includes ground truth regression tests
against reference pipeline outputs.
"""

from pathlib import Path

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

    @pytest.fixture
    def canonical_ordering(self):
        return canonical_ordering_for_biotite()

    def test_no_ligands_in_ubq(self, biotite_1ubq, canonical_ordering):
        assert len(detect_nonstandard_residues(biotite_1ubq, canonical_ordering)) == 0

    def test_detects_i4b_in_184l(self, cif_184l_with_i4b, canonical_ordering):
        ligands = detect_nonstandard_residues(cif_184l_with_i4b, canonical_ordering)
        i4b = {lig.res_name: lig for lig in ligands}.get("I4B")
        assert i4b is not None
        assert i4b.is_ligand is True
        assert "NON-POLYMER" in i4b.ccd_type.upper()
        assert i4b.coords.shape == (len(i4b.atom_names), 3)

    def test_detects_hem_in_155c(self, cif_155c_with_hem, canonical_ordering):
        ligands = detect_nonstandard_residues(cif_155c_with_hem, canonical_ordering)
        names = {lig.res_name for lig in ligands}
        assert "HEM" in names

    def test_detects_pse_with_partial_occupancy(
        self, cif_1a25_with_pse, canonical_ordering
    ):
        ligands = detect_nonstandard_residues(cif_1a25_with_pse, canonical_ordering)
        assert any(lig.res_name == "PSE" for lig in ligands)


class TestFullPipeline:
    """End-to-end: CIF -> detect -> prepare -> register -> verify."""

    @pytest.fixture
    def chem_db(self):
        return ChemicalDatabase.get_default()

    def test_i4b_small_drug(self, cif_184l_with_i4b, chem_db):
        """Small drug-like ligand (I4B, 10 heavy atoms) in lysozyme."""
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
        atom_name_set = set(atom_names)
        assert len(atom_names) == len(atom_name_set), "Duplicate atom names"
        for a, b in i4b_rt.bonds:
            assert a in atom_name_set and b in atom_name_set

    def test_hem_large_ligand(self, cif_155c_with_hem, chem_db):
        """Large ligand (HEM, 43 heavy atoms) in cytochrome c."""
        new_db, new_co = prepare_ligands(cif_155c_with_hem, chem_db=chem_db)

        assert "HEM" in {r.name for r in new_db.residues}
        hem_rt = next(r for r in new_db.residues if r.name == "HEM")
        assert len(hem_rt.atoms) > 30
        # Fe loses its element identity (Z=0) during SMILES roundtrip,
        # so the icoor tree has one fewer entry than the atom list.
        assert len(hem_rt.icoors) >= len(hem_rt.atoms) - 1

    def test_pse_partial_occupancy(self, cif_1a25_with_pse, chem_db):
        """Ligand with partial occupancy (PSE, 0.56) still prepares."""
        new_db, new_co = prepare_ligands(cif_1a25_with_pse, chem_db=chem_db)

        assert "PSE" in {r.name for r in new_db.residues}

    def test_caching_prevents_duplicate_work(self, cif_184l_with_i4b, chem_db):
        db1, _ = prepare_ligands(cif_184l_with_i4b, chem_db=chem_db)
        n_residues_after_first = len(db1.residues)

        db2, _ = prepare_ligands(cif_184l_with_i4b, chem_db=db1)
        assert len(db2.residues) == n_residues_after_first

    def test_ubq_passes_through_unchanged(self, biotite_1ubq, chem_db):
        new_db, _ = prepare_ligands(biotite_1ubq, chem_db=chem_db)
        assert len(new_db.residues) == len(chem_db.residues)


class TestPoseStackWithLigand:
    """Build a PoseStack with ligands and verify scoring runs."""

    @staticmethod
    def _score_cif_with_ligand(atom_array, torch_device):
        import torch

        from tmol.io.pose_stack_from_biotite import pose_stack_from_biotite
        from tmol.score import beta2016_score_function

        pose_stack = pose_stack_from_biotite(
            atom_array, torch_device, prepare_ligands=True
        )
        assert pose_stack.coords.shape[0] >= 1
        nonzero_coords = pose_stack.coords[pose_stack.coords != 0]
        assert not torch.any(torch.isnan(nonzero_coords))

        sfxn = beta2016_score_function(torch_device)
        scorer = sfxn.render_whole_pose_scoring_module(pose_stack)
        scores = scorer.unweighted_scores(pose_stack.coords)

        assert not torch.any(torch.isnan(scores))
        assert not torch.any(torch.isinf(scores))

    def test_i4b_posestack_scores(self, cif_184l_with_i4b, torch_device):
        """Small drug-like ligand (I4B, 10 atoms) in lysozyme."""
        self._score_cif_with_ligand(cif_184l_with_i4b, torch_device)

    def test_hem_posestack_scores(self, cif_155c_with_hem, torch_device):
        """Large ligand (HEM, 43 atoms) in cytochrome c."""
        self._score_cif_with_ligand(cif_155c_with_hem, torch_device)

    def test_pse_posestack_scores(self, cif_1a25_with_pse, torch_device):
        """Partial occupancy ligand (PSE, 0.56) scores without errors."""
        self._score_cif_with_ligand(cif_1a25_with_pse, torch_device)


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


def _parse_reference_params(path):
    """Parse a full Rosetta .params file into structured reference data.

    Returns a dict with keys: atoms (list of (name, type, charge)),
    bond_types (set of (a1, a2, order, ring_flag)), cut_bonds (set of
    frozensets), chis (list of (num, a1, a2, a3, a4, biaryl_flag)),
    proton_chis (list of raw line strings), nbr_atom (str),
    icoor_topology (dict of name -> (parent, gp, ggp)).
    """
    atoms = []
    bond_types = set()
    cut_bonds = set()
    chis = []
    proton_chis = []
    nbr_atom = ""
    icoor_topo = {}

    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue

            if parts[0] == "ATOM" and len(parts) >= 5:
                name, atype = parts[1], parts[2]
                charge = float(parts[4])
                atoms.append((name, atype, charge))

            elif parts[0] == "BOND_TYPE" and len(parts) >= 4:
                a1, a2 = parts[1].strip(), parts[2].strip()
                order = parts[3]
                ring = "RING" if len(parts) >= 5 and parts[4] == "RING" else ""
                bond_types.add((frozenset([a1, a2]), order, ring))

            elif parts[0] == "CUT_BOND" and len(parts) >= 3:
                cut_bonds.add(frozenset([parts[1].strip(), parts[2].strip()]))

            elif parts[0] == "CHI" and len(parts) >= 6:
                chi_num = int(parts[1])
                quad = (parts[2], parts[3], parts[4], parts[5])
                biaryl = "#biaryl" in line
                chis.append((chi_num, quad, biaryl))

            elif parts[0] == "PROTON_CHI":
                proton_chis.append(line.strip())

            elif parts[0] == "NBR_ATOM" and len(parts) >= 2:
                nbr_atom = parts[1]

            elif parts[0] == "ICOOR_INTERNAL" and len(parts) >= 8:
                name = parts[1]
                parent, gp, ggp = parts[5], parts[6], parts[7]
                icoor_topo[name] = (parent, gp, ggp)

    return {
        "atoms": atoms,
        "bond_types": bond_types,
        "cut_bonds": cut_bonds,
        "chis": chis,
        "proton_chis": proton_chis,
        "nbr_atom": nbr_atom,
        "icoor_topology": icoor_topo,
    }


def _load_smi_file(path, name):
    """Load a SMILES for a given molecule name from a tab-separated .smi file."""
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2 and parts[1] == name:
                return parts[0]
    raise ValueError(f"Molecule {name!r} not found in {path}")


class TestGroundTruthRegression:
    """Validate pipeline against Rosetta mol2genparams reference outputs.

    The reference .params files were generated by Rosetta's mol2genparams.py
    (from guangfeng/ligand_prep). Our atom typing must produce identical
    atom names, atom types, charges, bond topology, and ICOOR tree structure.
    """

    GROUND_TRUTH_DIR = Path(__file__).parent.parent / "data" / "ligand_ground_truth"
    CHARGE_TOLERANCE = 0.01

    @pytest.fixture(params=["ref1", "ref2"])
    def ref_data(self, request):
        """Load reference data and run our pipeline for comparison."""
        from tmol.ligand.atom_typing import assign_tmol_atom_types
        from tmol.ligand.mol3d import get_partial_charges, smiles_to_obmol
        from tmol.ligand.residue_builder import build_residue_type
        from tmol.ligand.smiles import protonate_ligand_smiles

        name = request.param
        gt = self.GROUND_TRUTH_DIR

        input_smi = _load_smi_file(gt / "designs.smi", name)
        expected_prot_smi = _load_smi_file(gt / "designs.prot.smi", name)
        ref = _parse_reference_params(gt / f"{name}.params")

        prot_smi = protonate_ligand_smiles(input_smi, ph=7.4)
        mol = smiles_to_obmol(prot_smi)
        atom_types = assign_tmol_atom_types(mol.OBMol)
        charges = get_partial_charges(mol)
        restype = build_residue_type(mol.OBMol, name, atom_types)

        return {
            "name": name,
            "input_smiles": input_smi,
            "expected_prot_smiles": expected_prot_smi,
            "actual_prot_smiles": prot_smi,
            "ref": ref,
            "atom_types": atom_types,
            "charges": charges,
            "restype": restype,
        }

    def test_protonation_matches(self, ref_data):
        """dimorphite_dl protonation must produce the expected SMILES."""
        assert ref_data["actual_prot_smiles"] == ref_data["expected_prot_smiles"], (
            f"Protonation mismatch for {ref_data['name']}: "
            f"got {ref_data['actual_prot_smiles']!r}, "
            f"expected {ref_data['expected_prot_smiles']!r}"
        )

    def test_atom_count_matches(self, ref_data):
        """Total atom count must match reference params."""
        ref_count = len(ref_data["ref"]["atoms"])
        actual_count = len(ref_data["restype"].atoms)
        assert actual_count == ref_count, (
            f"Atom count mismatch for {ref_data['name']}: "
            f"got {actual_count}, expected {ref_count}"
        )

    def test_atom_types_match(self, ref_data):
        """Each atom's name and Rosetta type must match the reference."""
        ref_atoms = ref_data["ref"]["atoms"]
        actual_atoms = [(a.name, a.atom_type) for a in ref_data["restype"].atoms]
        ref_name_type = [(name, atype) for name, atype, _ in ref_atoms]

        assert actual_atoms == ref_name_type, (
            f"Atom type mismatch for {ref_data['name']}:\n"
            f"  got:      {actual_atoms}\n"
            f"  expected: {ref_name_type}"
        )

    def test_charges_match(self, ref_data):
        """MMFF94 partial charges must match reference within tolerance."""
        ref_charges = {name: charge for name, _, charge in ref_data["ref"]["atoms"]}
        actual_charges = ref_data["charges"]

        for atom_name, ref_q in ref_charges.items():
            actual_q = actual_charges.get(atom_name)
            assert (
                actual_q is not None
            ), f"Missing charge for {atom_name} in {ref_data['name']}"
            assert abs(actual_q - ref_q) < self.CHARGE_TOLERANCE, (
                f"Charge mismatch for {atom_name} in {ref_data['name']}: "
                f"got {actual_q:.4f}, expected {ref_q:.4f}"
            )

    def test_bond_topology_matches(self, ref_data):
        """Bond pairs must match reference (order-independent)."""
        actual_bonds = set()
        for a, b in ref_data["restype"].bonds:
            actual_bonds.add(frozenset([a, b]))

        ref_bonds = set()
        for pair, _order, _ring in ref_data["ref"]["bond_types"]:
            ref_bonds.add(pair)

        assert actual_bonds == ref_bonds, (
            f"Bond topology mismatch for {ref_data['name']}:\n"
            f"  missing: {ref_bonds - actual_bonds}\n"
            f"  extra:   {actual_bonds - ref_bonds}"
        )

    def test_bond_count_matches(self, ref_data):
        """Bond count must match reference."""
        ref_count = len(ref_data["ref"]["bond_types"])
        actual_count = len(ref_data["restype"].bonds)
        assert actual_count == ref_count, (
            f"Bond count mismatch for {ref_data['name']}: "
            f"got {actual_count}, expected {ref_count}"
        )

    def test_icoor_completeness(self, ref_data):
        """Every atom must appear in the ICOOR tree."""
        actual_icoors = ref_data["restype"].icoors
        actual_atoms = ref_data["restype"].atoms
        icoor_names = {ic.name for ic in actual_icoors}
        atom_names = {a.name for a in actual_atoms}
        assert icoor_names == atom_names, (
            f"ICOOR atom set mismatch for {ref_data['name']}:\n"
            f"  missing from icoor: {atom_names - icoor_names}\n"
            f"  extra in icoor: {icoor_names - atom_names}"
        )
