"""Integration tests for the ligand preparation pipeline.

Tests the full pipeline from CIF structure with ligands through detection,
SMILES perception, protonation, 3D generation, atom typing, residue
building, and database registration. Includes ground truth regression tests
against reference pipeline outputs.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from tmol.io.pose_stack_from_biotite import canonical_ordering_for_biotite
from tmol.ligand import prepare_ligands, prepare_single_ligand
import tmol.ligand as ligand_module
from tmol.ligand.detect import detect_nonstandard_residues
from tmol.ligand.dimorphite_dl import protonate_mol_variants
from tmol.ligand.params_io import read_params_file, write_params_file
from tmol.ligand.registry import get_default_cache
from tmol.ligand.registry import clear_cache
from tmol.ligand.registry import get_cached_charges_for_key, get_cached_ligand_for_key
from tmol.ligand.registry import register_ligand


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
    def param_db(self):
        from tmol.database import ParameterDatabase

        return ParameterDatabase.get_fresh_default()

    def test_i4b_small_drug(self, cif_184l_with_i4b, param_db):
        """Small drug-like ligand (I4B, 10 heavy atoms) in lysozyme."""
        param_db, new_co = prepare_ligands(cif_184l_with_i4b, param_db=param_db)

        assert "I4B" in {r.name for r in param_db.chemical.residues}
        assert "I4B" in new_co.restype_io_equiv_classes

        i4b_rt = next(r for r in param_db.chemical.residues if r.name == "I4B")
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
        for a, b, _ in i4b_rt.bonds:
            assert a in atom_name_set and b in atom_name_set

    def test_hem_large_ligand(self, cif_155c_with_hem, param_db):
        """Large ligand (HEM, 43 heavy atoms) in cytochrome c."""
        param_db, new_co = prepare_ligands(cif_155c_with_hem, param_db=param_db)

        assert "HEM" in {r.name for r in param_db.chemical.residues}
        hem_rt = next(r for r in param_db.chemical.residues if r.name == "HEM")
        assert len(hem_rt.atoms) > 30
        assert len(hem_rt.icoors) >= len(hem_rt.atoms) - 1

    def test_pse_partial_occupancy(self, cif_1a25_with_pse, param_db):
        """Ligand with partial occupancy (PSE, 0.56) still prepares."""
        param_db, new_co = prepare_ligands(cif_1a25_with_pse, param_db=param_db)

        assert "PSE" in {r.name for r in param_db.chemical.residues}

    def test_caching_prevents_duplicate_work(self, cif_184l_with_i4b, param_db):
        n_before = len(param_db.chemical.residues)
        param_db, _ = prepare_ligands(cif_184l_with_i4b, param_db=param_db)
        n_after = len(param_db.chemical.residues)
        assert n_after > n_before

        param_db, _ = prepare_ligands(cif_184l_with_i4b, param_db=param_db)
        assert len(param_db.chemical.residues) == n_after

    def test_cache_key_includes_ph(self, cif_184l_with_i4b, param_db):
        clear_cache()
        cache = get_default_cache()
        prepare_ligands(cif_184l_with_i4b, param_db=param_db, ph=7.4, cache=cache)
        prepare_ligands(cif_184l_with_i4b, param_db=param_db, ph=6.8, cache=cache)
        i4b_keys = [k for k in cache.ligands_by_key.keys() if k[0] == "I4B"]
        assert len(i4b_keys) >= 2

    def test_cache_accessors_return_defensive_copies(self, cif_184l_with_i4b, param_db):
        clear_cache()
        cache = get_default_cache()
        prepare_ligands(cif_184l_with_i4b, param_db=param_db, ph=7.4, cache=cache)
        i4b_keys = [k for k in cache.ligands_by_key.keys() if k[0] == "I4B"]
        assert i4b_keys
        key = i4b_keys[0]

        cached_restype = get_cached_ligand_for_key(key, cache=cache)
        assert cached_restype is not None
        reread_restype = get_cached_ligand_for_key(key, cache=cache)
        assert reread_restype is not None
        assert reread_restype is not cached_restype

        cached_charges = get_cached_charges_for_key(key, cache=cache)
        assert cached_charges is not None
        some_atom = next(iter(cached_charges))
        cached_charges[some_atom] = 999.0  # mutate caller copy
        reread_charges = get_cached_charges_for_key(key, cache=cache)
        assert reread_charges is not None
        assert reread_charges[some_atom] != 999.0

    def test_register_ligand_returns_inserted_status(self, cif_184l_with_i4b, param_db):
        ligands = detect_nonstandard_residues(
            cif_184l_with_i4b, canonical_ordering_for_biotite()
        )
        i4b = next(l for l in ligands if l.res_name == "I4B")
        restype, charges, atom_type_elements = prepare_single_ligand(i4b, ph=7.4)

        inserted = register_ligand(
            param_db,
            restype,
            partial_charges=charges,
            atom_type_elements=atom_type_elements,
        )
        assert inserted is True

        inserted_again = register_ligand(
            param_db,
            restype,
            partial_charges=charges,
            atom_type_elements=atom_type_elements,
        )
        assert inserted_again is False

    def test_ubq_passes_through_unchanged(self, biotite_1ubq, param_db):
        n_before = len(param_db.chemical.residues)
        param_db, _ = prepare_ligands(biotite_1ubq, param_db=param_db)
        assert len(param_db.chemical.residues) == n_before


class TestLigandScoringData:
    """Verify that scoring databases are correctly populated for ligands."""

    def test_elec_charges_populated(self, cif_184l_with_i4b):
        """Elec partial charges are injected into the ParameterDatabase."""
        from tmol.database import ParameterDatabase

        param_db = ParameterDatabase.get_fresh_default()
        param_db, _ = prepare_ligands(cif_184l_with_i4b, param_db=param_db)

        i4b_charges = [
            p for p in param_db.scoring.elec.atom_charge_parameters if p.res == "I4B"
        ]
        assert len(i4b_charges) > 0, "No elec charges for I4B"
        assert any(abs(p.charge) > 1e-6 for p in i4b_charges), "All I4B charges zero"

    def test_cartbonded_params_populated(self, cif_184l_with_i4b):
        """CartBonded params (lengths, angles, impropers) are in the DB."""
        from tmol.database import ParameterDatabase

        param_db = ParameterDatabase.get_fresh_default()
        param_db, _ = prepare_ligands(cif_184l_with_i4b, param_db=param_db)

        assert "I4B" in param_db.scoring.cartbonded.residue_params
        cart_res = param_db.scoring.cartbonded.residue_params["I4B"]
        assert len(cart_res.length_parameters) > 0, "No bond length params for I4B"
        assert len(cart_res.angle_parameters) > 0, "No bond angle params for I4B"

    def test_hbond_atom_types_annotated(self, cif_184l_with_i4b):
        """New atom types have correct donor/acceptor/hybridization flags."""
        from tmol.database import ParameterDatabase

        param_db = ParameterDatabase.get_fresh_default()
        param_db, _ = prepare_ligands(cif_184l_with_i4b, param_db=param_db)

        atom_types_by_name = {at.name: at for at in param_db.chemical.atom_types}
        if "Ohx" in atom_types_by_name:
            ohx = atom_types_by_name["Ohx"]
            assert ohx.is_acceptor, "Ohx should be an acceptor"
            assert ohx.is_donor, "Ohx should be a donor"
        if "HN" in atom_types_by_name:
            hn = atom_types_by_name["HN"]
            assert hn.is_polarh, "HN should be polar hydrogen"

    def test_ljlk_halogen_params_exist(self):
        """All halogen types (aromatic and non-aromatic) have LJLK params."""
        from tmol.database import ParameterDatabase

        param_db = ParameterDatabase.get_fresh_default()
        ljlk_names = {p.name for p in param_db.scoring.ljlk.atom_type_parameters}
        for halogen in ["F", "Cl", "Br", "I", "FR", "ClR", "BrR", "IR"]:
            assert halogen in ljlk_names, f"Missing LJLK params for {halogen}"

    def test_remove_residue_scoring_params(self, cif_184l_with_i4b):
        """add/remove API works correctly."""
        from tmol.database import ParameterDatabase

        param_db = ParameterDatabase.get_fresh_default()
        param_db, _ = prepare_ligands(cif_184l_with_i4b, param_db=param_db)

        assert "I4B" in param_db.scoring.cartbonded.residue_params
        i4b_charges = [
            p for p in param_db.scoring.elec.atom_charge_parameters if p.res == "I4B"
        ]
        assert len(i4b_charges) > 0

        param_db.remove_residue_scoring_params("I4B")

        assert "I4B" not in param_db.scoring.cartbonded.residue_params
        i4b_charges_after = [
            p for p in param_db.scoring.elec.atom_charge_parameters if p.res == "I4B"
        ]
        assert len(i4b_charges_after) == 0

    def test_i4b_aliphatic_carbons_not_sp2_typed(self, cif_184l_with_i4b):
        """I4B aliphatic substituent carbons should stay saturated (CS*)."""
        from tmol.database import ParameterDatabase

        param_db = ParameterDatabase.get_fresh_default()
        param_db, _ = prepare_ligands(cif_184l_with_i4b, param_db=param_db)
        i4b_rt = next(r for r in param_db.chemical.residues if r.name == "I4B")
        atom_type_by_name = {a.name: a.atom_type for a in i4b_rt.atoms}

        # These are the clearly aliphatic substituent carbons in the I4B CIF.
        for name in ("C2'", "C3'", "C4'"):
            assert name in atom_type_by_name, f"Missing expected I4B atom {name}"
            assert atom_type_by_name[name].startswith("CS"), (
                f"{name} should be saturated carbon (CS*), got "
                f"{atom_type_by_name[name]}"
            )

    def test_i4b_impropers_exclude_aliphatic_substituent(self, cif_184l_with_i4b):
        """I4B impropers should not force planarity on the saturated substituent."""
        from tmol.database import ParameterDatabase

        param_db = ParameterDatabase.get_fresh_default()
        param_db, _ = prepare_ligands(cif_184l_with_i4b, param_db=param_db)
        cart_res = param_db.scoring.cartbonded.residue_params["I4B"]

        centers = {imp.atm3 for imp in cart_res.improper_parameters}
        for name in ("C2'", "C3'", "C4'"):
            assert (
                name not in centers
            ), f"Improper center incorrectly includes saturated atom {name}"


def test_prepare_ligands_missing_sidechain_rebuild_skips_ligand_dunbrack(
    cif_184l_with_i4b, torch_device
):
    """Missing-sidechain rebuild should not invoke Dunbrack on ligand blocks."""
    import numpy
    import torch

    from tmol.database import ParameterDatabase
    from tmol.io.pose_stack_from_biotite import pose_stack_from_biotite

    bt = cif_184l_with_i4b.copy()
    protein_cb = (bt.res_name != "I4B") & (bt.atom_name == "CB")
    if not numpy.any(protein_cb):
        pytest.skip("Could not find a protein CB atom to remove")

    first_cb = numpy.nonzero(protein_cb)[0][0]
    keep_mask = numpy.ones(bt.array_length(), dtype=bool)
    keep_mask[first_cb] = False
    bt_missing = bt[keep_mask]

    pose_stack = pose_stack_from_biotite(
        bt_missing,
        torch_device,
        prepare_ligands=True,
        param_db=ParameterDatabase.get_fresh_default(),
    )

    assert pose_stack.coords.shape[0] >= 1
    nonzero_coords = pose_stack.coords[pose_stack.coords != 0]
    assert not torch.any(torch.isnan(nonzero_coords))
    assert not torch.any(torch.isinf(nonzero_coords))


def test_prepare_ligands_ligand_only_missing_does_not_invoke_dunbrack(
    cif_184l_with_i4b, torch_device
):
    """Ligand-only missing atoms should bypass Dunbrack rebuilding cleanly."""
    import numpy
    import torch

    from tmol.database import ParameterDatabase
    from tmol.io.pose_stack_from_biotite import pose_stack_from_biotite

    bt = cif_184l_with_i4b.copy()
    ligand_atoms = numpy.nonzero(bt.res_name == "I4B")[0]
    if ligand_atoms.shape[0] == 0:
        pytest.skip("Could not find I4B ligand atoms to remove")

    keep_mask = numpy.ones(bt.array_length(), dtype=bool)
    keep_mask[ligand_atoms[0]] = False
    bt_ligand_missing = bt[keep_mask]

    pose_stack = pose_stack_from_biotite(
        bt_ligand_missing,
        torch_device,
        prepare_ligands=True,
        param_db=ParameterDatabase.get_fresh_default(),
    )

    assert pose_stack.coords.shape[0] >= 1
    nonzero_coords = pose_stack.coords[pose_stack.coords != 0]
    assert not torch.any(torch.isnan(nonzero_coords))
    assert not torch.any(torch.isinf(nonzero_coords))


@pytest.mark.skip(
    reason="Rotamer/Dunbrack energy term does not yet support ligands; "
    "crashes with SIGFPE before pytest can catch the error",
)
class TestPoseStackWithLigand:
    """Build a PoseStack with ligands and verify scoring runs."""

    @staticmethod
    def _score_cif_with_ligand(atom_array, torch_device):
        import torch

        from tmol.database import ParameterDatabase
        from tmol.io.pose_stack_from_biotite import pose_stack_from_biotite
        from tmol.score import beta2016_score_function

        param_db = ParameterDatabase.get_fresh_default()
        pose_stack = pose_stack_from_biotite(
            atom_array, torch_device, prepare_ligands=True, param_db=param_db
        )
        assert pose_stack.coords.shape[0] >= 1
        nonzero_coords = pose_stack.coords[pose_stack.coords != 0]
        assert not torch.any(torch.isnan(nonzero_coords))

        sfxn = beta2016_score_function(torch_device, param_db=param_db)
        scorer = sfxn.render_whole_pose_scoring_module(pose_stack)
        scores = scorer.unweighted_scores(pose_stack.coords)

        assert not torch.any(torch.isnan(scores))
        assert not torch.any(torch.isinf(scores))

    def test_i4b_posestack_scores(self, cif_184l_with_i4b, torch_device):
        """Small drug-like ligand (I4B, 10 atoms) in lysozyme."""
        self._score_cif_with_ligand(cif_184l_with_i4b, torch_device)

    def test_hem_posestack_builds(self, cif_155c_with_hem, torch_device):
        """Large ligand (HEM, 43 atoms) in cytochrome c.

        HEM contains Fe which is dropped during preparation (unsupported
        element). Verify the PoseStack builds but skip scoring — the
        remaining atoms don't have meaningful LJ/LK parameters.
        """
        import torch

        from tmol.database import ParameterDatabase
        from tmol.io.pose_stack_from_biotite import pose_stack_from_biotite

        param_db = ParameterDatabase.get_fresh_default()
        pose_stack = pose_stack_from_biotite(
            cif_155c_with_hem, torch_device, prepare_ligands=True, param_db=param_db
        )
        assert pose_stack.coords.shape[0] >= 1
        nonzero_coords = pose_stack.coords[pose_stack.coords != 0]
        assert not torch.any(torch.isnan(nonzero_coords))

    def test_pse_posestack_scores(self, cif_1a25_with_pse, torch_device):
        """Partial occupancy ligand (PSE, 0.56) scores without errors."""
        self._score_cif_with_ligand(cif_1a25_with_pse, torch_device)

    def test_atp_posestack_scores(self, torch_device, tmp_path):
        """ATP ligand (31 heavy atoms + H = >32 total) triggers tile edge case.

        ATP has >32 total atoms but <=32 heavy atoms, which exercises the
        second tile iteration in the LJLK scoring kernel. This was a known
        crash (see uw-ipd/tmol jflat06/atp_ligand_load branch).
        """
        import biotite.structure
        from biotite.database import rcsb

        path = rcsb.fetch("3QWQ", "cif", target_path=tmp_path)
        bt = biotite.structure.io.load_structure(
            path, extra_fields=["occupancy", "b_factor"]
        )
        if isinstance(bt, biotite.structure.AtomArrayStack):
            bt = bt[0]
        self._score_cif_with_ligand(bt, torch_device)

    def test_i4b_minimize_and_cif_roundtrip(
        self, cif_184l_with_i4b, torch_device, tmp_path
    ):
        """Build with ligands, minimize briefly, and write back to CIF."""
        import biotite.structure
        import torch
        from biotite.structure.io.pdbx import CIFFile, set_structure

        from tmol.database import ParameterDatabase
        from tmol.io.pose_stack_from_biotite import (
            biotite_from_pose_stack,
            pose_stack_from_biotite,
        )
        from tmol.optimization.lbfgs_armijo import LBFGS_Armijo
        from tmol.optimization.sfxn_modules import CartesianSfxnNetwork
        from tmol.score import beta2016_score_function

        if torch_device != torch.device("cpu"):
            pytest.skip("Roundtrip minimization test runs on CPU only")

        pose_stack, context = pose_stack_from_biotite(
            cif_184l_with_i4b,
            torch_device,
            prepare_ligands=True,
            param_db=ParameterDatabase.get_fresh_default(),
            return_context=True,
        )
        sfxn = beta2016_score_function(
            torch_device, param_db=context.parameter_database
        )
        network = CartesianSfxnNetwork(sfxn, pose_stack)
        optimizer = LBFGS_Armijo(network.parameters(), lr=0.05, max_iter=5)

        def closure():
            optimizer.zero_grad()
            e = network().sum()
            e.backward()
            return e

        optimizer.step(closure)

        out = biotite_from_pose_stack(pose_stack, co=context.canonical_ordering)
        assert isinstance(
            out, (biotite.structure.AtomArray, biotite.structure.AtomArrayStack)
        )
        cif = CIFFile()
        set_structure(
            cif, out if isinstance(out, biotite.structure.AtomArray) else out[0]
        )
        out_path = tmp_path / "i4b_minimized_roundtrip.cif"
        cif.write(out_path)
        assert out_path.exists()


class TestParamsRoundtrip:
    """Write a prepared ligand to .params and read it back."""

    def test_i4b_params_roundtrip(self, tmp_path, cif_184l_with_i4b):
        co = canonical_ordering_for_biotite()
        ligands = detect_nonstandard_residues(cif_184l_with_i4b, co)
        i4b = next(lig for lig in ligands if lig.res_name == "I4B")
        restype, charges, atom_type_elements = prepare_single_ligand(i4b)

        path = tmp_path / "I4B.params"
        write_params_file(restype, path, partial_charges=charges)
        loaded = read_params_file(path)

        assert loaded.name == "I4B"
        assert len(loaded.atoms) == len(restype.atoms)
        assert len(loaded.bonds) == len(restype.bonds)
        assert len(loaded.icoors) == len(restype.icoors)
        assert len(atom_type_elements) > 0

    def test_params_roundtrip_preserves_bond_types(self, tmp_path, cif_184l_with_i4b):
        co = canonical_ordering_for_biotite()
        ligands = detect_nonstandard_residues(cif_184l_with_i4b, co)
        i4b = next(lig for lig in ligands if lig.res_name == "I4B")
        restype, charges, _ = prepare_single_ligand(i4b)

        path = tmp_path / "I4B_bondtypes.params"
        write_params_file(restype, path, partial_charges=charges)
        loaded = read_params_file(path)

        assert all(len(b) == 3 for b in loaded.bonds)
        assert {(a, b, t) for a, b, t in loaded.bonds} == {
            (a, b, t) for a, b, t in restype.bonds
        }


def test_collect_new_atom_types_strict_mode_errors(default_database):
    from tmol.ligand.registry import _collect_new_atom_types

    residue = SimpleNamespace(
        name="UNK",
        atoms=(SimpleNamespace(name="X1", atom_type="ZZZ"),),
    )
    with pytest.raises(ValueError, match="Unknown element mapping"):
        _collect_new_atom_types(
            default_database.chemical,
            residue,
            atom_type_elements={},
            strict_atom_types=True,
        )


def test_protonate_mol_variants_produces_valid_mol():
    from rdkit import Chem

    input_smiles = "CC(=O)ON"
    mol = Chem.MolFromSmiles(input_smiles)
    assert mol is not None
    mol_variants = protonate_mol_variants(
        mol,
        min_ph=7.4,
        max_ph=7.4,
        pka_precision=0.1,
        max_variants=128,
        silent=True,
    )
    assert mol_variants
    result_smi = Chem.MolToSmiles(mol_variants[0], isomericSmiles=True)
    assert Chem.MolFromSmiles(result_smi) is not None


def test_prepare_single_ligand_uses_index_mapping_before_graph(
    cif_184l_with_i4b, monkeypatch
):
    ligands = detect_nonstandard_residues(
        cif_184l_with_i4b, canonical_ordering_for_biotite()
    )
    i4b = next(l for l in ligands if l.res_name == "I4B")

    def _fail_graph_match(*_args, **_kwargs):
        raise AssertionError("Graph matching should not be called in direct Mol path")

    monkeypatch.setattr(ligand_module, "_rename_atoms_to_cif", _fail_graph_match)
    restype, charges, _ = prepare_single_ligand(i4b, ph=7.4)
    atom_names = {a.name for a in restype.atoms}
    assert "C3'" in atom_names
    assert "C4'" in atom_names
    assert "C2'" in atom_names
    assert "C3'" in charges


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
    """Load a SMILES for a given molecule name from a .smi file.

    Accepts both tab-separated and whitespace-separated formats:
      <SMILES><ws><name>
    """
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[-1] == name:
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
        from rdkit import Chem

        from tmol.ligand.atom_typing import assign_tmol_atom_types
        from tmol.ligand.mol3d import compute_mmff94_charges, rdkit_mol_to_obmol
        from tmol.ligand.residue_builder import build_residue_type
        from tmol.ligand.smiles import protonate_ligand_mol

        name = request.param
        gt = self.GROUND_TRUTH_DIR

        input_smi = _load_smi_file(gt / "designs.smi", name)
        expected_prot_smi = _load_smi_file(gt / "designs.prot.smi", name)
        ref = _parse_reference_params(gt / f"{name}.params")

        rdkit_mol = Chem.MolFromSmiles(input_smi)
        protonated = protonate_ligand_mol(rdkit_mol, ph=7.4)
        protonated = Chem.AddHs(protonated, addCoords=False)
        prot_smi = Chem.MolToSmiles(
            Chem.RemoveHs(protonated), isomericSmiles=True
        )
        charges_by_idx = compute_mmff94_charges(protonated)
        obmol = rdkit_mol_to_obmol(protonated)
        atom_types = assign_tmol_atom_types(obmol.OBMol)
        charges = {at.atom_name: charges_by_idx[at.index] for at in atom_types
                   if at.index in charges_by_idx}
        restype = build_residue_type(obmol.OBMol, name, atom_types)

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
        for a, b, _ in ref_data["restype"].bonds:
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
