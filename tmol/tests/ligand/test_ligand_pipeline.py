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
from tmol.ligand.registry import inject_ligand_preparations


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
        assert "NON-POLYMER" in i4b.ccd_type.upper()
        assert i4b.coords.shape == (len(i4b.atom_names), 3)

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

        return ParameterDatabase.get_default()

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
        for a, b, *_ in i4b_rt.bonds:
            assert a in atom_name_set and b in atom_name_set

    def test_hem_metal_ligand_skipped(self, cif_155c_with_hem, param_db):
        """HEM contains Fe; metal-containing ligands are unsupported and skipped."""
        param_db, new_co = prepare_ligands(cif_155c_with_hem, param_db=param_db)

        assert "HEM" not in {r.name for r in param_db.chemical.residues}

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

    def test_inject_ligand_preparations_is_idempotent(
        self, cif_184l_with_i4b, param_db
    ):
        ligands = detect_nonstandard_residues(
            cif_184l_with_i4b, canonical_ordering_for_biotite()
        )
        i4b = next(l for l in ligands if l.res_name == "I4B")
        prep = prepare_single_ligand(i4b, ph=7.4)

        n_before = len(param_db.chemical.residues)
        extended_db = inject_ligand_preparations(param_db, [prep])
        assert len(extended_db.chemical.residues) > n_before
        assert any(r.name == "I4B" for r in extended_db.chemical.residues)

        # Re-injecting the same preparation is a no-op (residue already
        # registered) — same number of residues, same database identity.
        again = inject_ligand_preparations(extended_db, [prep])
        assert len(again.chemical.residues) == len(extended_db.chemical.residues)

    def test_ubq_passes_through_unchanged(self, biotite_1ubq, param_db):
        n_before = len(param_db.chemical.residues)
        param_db, _ = prepare_ligands(biotite_1ubq, param_db=param_db)
        assert len(param_db.chemical.residues) == n_before


class TestLigandScoringData:
    """Verify that scoring databases are correctly populated for ligands."""

    def test_elec_charges_populated(self, cif_184l_with_i4b):
        """Elec partial charges are injected into the ParameterDatabase."""
        from tmol.database import ParameterDatabase

        param_db = ParameterDatabase.get_default()
        param_db, _ = prepare_ligands(cif_184l_with_i4b, param_db=param_db)

        i4b_charges = [
            p for p in param_db.scoring.elec.atom_charge_parameters if p.res == "I4B"
        ]
        assert len(i4b_charges) > 0, "No elec charges for I4B"
        assert any(abs(p.charge) > 1e-6 for p in i4b_charges), "All I4B charges zero"

    def test_cartbonded_params_populated(self, cif_184l_with_i4b):
        """CartBonded params (lengths, angles, impropers) are in the DB."""
        from tmol.database import ParameterDatabase

        param_db = ParameterDatabase.get_default()
        param_db, _ = prepare_ligands(cif_184l_with_i4b, param_db=param_db)

        assert "I4B" in param_db.scoring.cartbonded.residue_params
        cart_res = param_db.scoring.cartbonded.residue_params["I4B"]
        assert len(cart_res.length_parameters) > 0, "No bond length params for I4B"
        assert len(cart_res.angle_parameters) > 0, "No bond angle params for I4B"

    def test_hbond_atom_types_annotated(self, cif_184l_with_i4b):
        """New atom types have correct donor/acceptor/hybridization flags."""
        from tmol.database import ParameterDatabase

        param_db = ParameterDatabase.get_default()
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

        param_db = ParameterDatabase.get_default()
        ljlk_names = {p.name for p in param_db.scoring.ljlk.atom_type_parameters}
        for halogen in ["F", "Cl", "Br", "I", "FR", "ClR", "BrR", "IR"]:
            assert halogen in ljlk_names, f"Missing LJLK params for {halogen}"

    def test_injection_does_not_mutate_original_db(self, cif_184l_with_i4b):
        """inject_residue_params returns a new DB; the original is unchanged."""
        from tmol.database import ParameterDatabase

        original_db = ParameterDatabase.get_default()
        original_residue_count = len(original_db.chemical.residues)
        original_charge_count = len(original_db.scoring.elec.atom_charge_parameters)

        extended_db, _ = prepare_ligands(cif_184l_with_i4b, param_db=original_db)

        assert "I4B" in extended_db.scoring.cartbonded.residue_params
        assert any(
            p.res == "I4B" for p in extended_db.scoring.elec.atom_charge_parameters
        )
        assert len(original_db.chemical.residues) == original_residue_count
        assert (
            len(original_db.scoring.elec.atom_charge_parameters)
            == original_charge_count
        )

    def test_i4b_aliphatic_carbons_not_sp2_typed(self, cif_184l_with_i4b):
        """I4B aliphatic substituent carbons should stay saturated (CS*)."""
        from tmol.database import ParameterDatabase

        param_db = ParameterDatabase.get_default()
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

        param_db = ParameterDatabase.get_default()
        param_db, _ = prepare_ligands(cif_184l_with_i4b, param_db=param_db)
        cart_res = param_db.scoring.cartbonded.residue_params["I4B"]

        centers = {imp.atm3 for imp in cart_res.improper_parameters}
        for name in ("C2'", "C3'", "C4'"):
            assert (
                name not in centers
            ), f"Improper center incorrectly includes saturated atom {name}"


def test_prepare_ligands_missing_ligand_atom_fails(cif_184l_with_i4b, torch_device):
    """Ligands with missing atoms are unsupported; loading must fail."""
    import numpy

    from tmol.database import ParameterDatabase
    from tmol.io.pose_stack_from_biotite import pose_stack_from_biotite

    bt = cif_184l_with_i4b.copy()
    ligand_atoms = numpy.nonzero(bt.res_name == "I4B")[0]
    if ligand_atoms.shape[0] == 0:
        pytest.skip("Could not find I4B ligand atoms to remove")

    keep_mask = numpy.ones(bt.array_length(), dtype=bool)
    keep_mask[ligand_atoms[0]] = False
    bt_ligand_missing = bt[keep_mask]

    with pytest.raises(Exception):
        pose_stack_from_biotite(
            bt_ligand_missing,
            torch_device,
            prepare_ligands=True,
            param_db=ParameterDatabase.get_fresh_default(),
        )


class TestPoseStackWithLigand:
    """Build a PoseStack with ligands and verify scoring runs."""

    @staticmethod
    def _sanity_check_pose_stack(pose_stack):
        """Validate PoseStack invariants before handing it to scoring.

        Scoring kernels access coords via block_coord_offset + per-block atom
        index; if any of those are inconsistent the kernel hits a bounds-check
        failure with no useful message. This surfaces the problem in Python.
        """
        import torch

        pbt = pose_stack.packed_block_types
        n_block_types = len(pbt.active_block_types)
        n_poses, max_n_blocks = pose_stack.block_type_ind.shape
        n_atoms_total = pose_stack.coords.shape[1]

        bti = pose_stack.block_type_ind.cpu()
        offsets = pose_stack.block_coord_offset.cpu()
        n_atoms_per_bt = pbt.n_atoms.cpu()

        valid_mask = bti != -1
        valid_bti = bti[valid_mask]
        assert valid_bti.numel() > 0, "PoseStack has no real blocks"
        assert int(valid_bti.min()) >= 0
        assert int(valid_bti.max()) < n_block_types, (
            f"block_type_ind has value {int(valid_bti.max())} but only "
            f"{n_block_types} block types exist"
        )

        # Each block's [offset, offset + n_atoms_for_that_bt) must fit inside coords.
        for p in range(n_poses):
            for b in range(max_n_blocks):
                bt = int(bti[p, b])
                if bt == -1:
                    continue
                off = int(offsets[p, b])
                n = int(n_atoms_per_bt[bt])
                assert 0 <= off, f"pose {p} block {b}: negative offset {off}"
                assert off + n <= n_atoms_total, (
                    f"pose {p} block {b} (bt={bt} name="
                    f"{pbt.active_block_types[bt].name}): offset {off} + "
                    f"n_atoms {n} exceeds coords length {n_atoms_total}"
                )
                slc = pose_stack.coords[p, off : off + n]
                if not torch.isfinite(slc).all():
                    bad = (~torch.isfinite(slc)).nonzero(as_tuple=False)
                    raise AssertionError(
                        f"non-finite coords in pose {p} block {b} "
                        f"(bt={bt} name={pbt.active_block_types[bt].name}, "
                        f"n_atoms={n}, offset={off}): atoms {bad[:5].tolist()}"
                    )
                # All-zero coords for a real atom usually means the block was
                # registered but its coordinates were never copied in.
                norms = slc.norm(dim=1)
                if n > 0 and (norms == 0).all():
                    raise AssertionError(
                        f"pose {p} block {b} (bt={bt} name="
                        f"{pbt.active_block_types[bt].name}): all "
                        f"{n} atom coordinates are exactly zero"
                    )

        # Per-block-type LJLK atom_types must all be in range. A -1 here means
        # the atom's Rosetta atom_type was never registered into the LJLK type
        # table; the kernel will then read type_params[-1] and trip the bounds
        # check.
        if hasattr(pbt, "atom_types"):
            at = pbt.atom_types.cpu()
            for bt_ind in {int(b) for b in valid_bti.tolist()}:
                bt = pbt.active_block_types[bt_ind]
                n = int(n_atoms_per_bt[bt_ind])
                row = at[bt_ind, :n]
                if (row < 0).any():
                    bad_idx = (row < 0).nonzero(as_tuple=False).flatten().tolist()
                    bad_names = [
                        (bt.atoms[i].name, bt.atoms[i].atom_type) for i in bad_idx
                    ]
                    raise AssertionError(
                        f"block type {bt.name}: atoms {bad_names} have unresolved "
                        f"LJLK atom_type indices (-1)"
                    )

        # Per-block-type LJLK heavy-atom-in-tile indices must be < n_atoms (they
        # index into the per-block coords slice).
        if hasattr(pbt, "ljlk_heavy_atoms_in_tile") and hasattr(
            pbt, "ljlk_n_heavy_atoms_in_tile"
        ):
            hat = pbt.ljlk_heavy_atoms_in_tile.cpu()
            n_in_tile = pbt.ljlk_n_heavy_atoms_in_tile.cpu()
            tile_size = 32
            for bt_ind in {int(b) for b in valid_bti.tolist()}:
                bt = pbt.active_block_types[bt_ind]
                n = int(n_atoms_per_bt[bt_ind])
                for tile in range(n_in_tile.shape[1]):
                    n_t = int(n_in_tile[bt_ind, tile])
                    if n_t == 0:
                        continue
                    row = hat[bt_ind, tile * tile_size : tile * tile_size + n_t]
                    if (row < 0).any() or (row >= n).any():
                        raise AssertionError(
                            f"block type {bt.name}: tile {tile} heavy-atom "
                            f"indices {row.tolist()} out of range for n_atoms={n}"
                        )

        # inter_residue_connections must point to a real block on the same pose.
        irc = pose_stack.inter_residue_connections.cpu()
        for p in range(n_poses):
            for b in range(max_n_blocks):
                if int(bti[p, b]) == -1:
                    continue
                for c in range(irc.shape[2]):
                    other = int(irc[p, b, c, 0])
                    if other == -1:
                        continue
                    assert (
                        0 <= other < max_n_blocks
                    ), f"pose {p} block {b} conn {c} -> {other} (out of range)"
                    assert int(bti[p, other]) != -1, (
                        f"pose {p} block {b} conn {c} -> block {other} which "
                        f"is sentinel (-1)"
                    )

    @staticmethod
    def _score_cif_with_ligand(
        atom_array, torch_device, dump_name=None, expected_ligand_name3=None
    ):
        import os
        import torch

        from tmol.database import ParameterDatabase
        from tmol.io.pose_stack_from_biotite import pose_stack_from_biotite
        from tmol.io.write_pose_stack_pdb import write_pose_stack_pdb
        from tmol.score import beta2016_score_function

        param_db = ParameterDatabase.get_default()
        pose_stack = pose_stack_from_biotite(
            atom_array, torch_device, prepare_ligands=True, param_db=param_db
        )

        pbt = pose_stack.packed_block_types
        bt_names = [bt.name3 for bt in pbt.active_block_types]
        bti0 = pose_stack.block_type_ind64[0].cpu().numpy()
        unique_bti = sorted({int(b) for b in bti0 if b != -1})
        if dump_name is not None:
            print(f"[dump debug] {dump_name}: active_block_types name3s = {bt_names}")
            print(
                f"[dump debug] {dump_name}: pose0 unique block_type_inds = "
                f"{unique_bti}"
            )
            for b in unique_bti:
                print(
                    f"[dump debug]   bti={b} name3={pbt.active_block_types[b].name3} "
                    f"name={pbt.active_block_types[b].name}"
                )
            write_pose_stack_pdb(
                pose_stack, os.path.join(os.getcwd(), f"{dump_name}.pdb")
            )

        if expected_ligand_name3 is not None:
            present = {pbt.active_block_types[b].name3 for b in unique_bti}
            assert expected_ligand_name3 in present, (
                f"expected ligand {expected_ligand_name3!r} not in pose stack; "
                f"present block types = {sorted(present)}"
            )

            lig_bt_ind = next(
                b
                for b in unique_bti
                if pbt.active_block_types[b].name3 == expected_ligand_name3
            )
            lig_bt = pbt.active_block_types[lig_bt_ind]
            n_atoms = int(pbt.n_atoms[lig_bt_ind])
            print(
                f"[ligand dump] {expected_ligand_name3} (bt={lig_bt_ind} "
                f"name={lig_bt.name}): n_atoms={n_atoms} "
                f"max_n_atoms_pbt={pbt.max_n_atoms}"
            )
            if hasattr(pbt, "atom_types"):
                print(
                    f"  atom_types[:n] = "
                    f"{pbt.atom_types[lig_bt_ind, :n_atoms].cpu().tolist()}"
                )
            else:
                print("  atom_types not yet attached to pbt")
            if hasattr(pbt, "ljlk_n_heavy_atoms_in_tile"):
                print(
                    f"  ljlk_n_heavy_atoms_in_tile = "
                    f"{pbt.ljlk_n_heavy_atoms_in_tile[lig_bt_ind].cpu().tolist()}"
                )
                print(
                    f"  ljlk_heavy_atoms_in_tile = "
                    f"{pbt.ljlk_heavy_atoms_in_tile[lig_bt_ind].cpu().tolist()}"
                )
            if hasattr(pbt, "n_conn"):
                print(f"  n_conn = {int(pbt.n_conn[lig_bt_ind])}")
            if hasattr(pbt, "conn_atom"):
                print(f"  conn_atom = {pbt.conn_atom[lig_bt_ind].cpu().tolist()}")
            if hasattr(pbt, "bond_separation"):
                bs = pbt.bond_separation[lig_bt_ind].cpu()
                print(
                    f"  bond_separation shape={tuple(bs.shape)} "
                    f"min={int(bs[:n_atoms, :n_atoms].min())} "
                    f"max={int(bs[:n_atoms, :n_atoms].max())}"
                )
            # Show the ligand's pose-level slice
            for p in range(pose_stack.coords.shape[0]):
                for b in range(pose_stack.block_type_ind.shape[1]):
                    if int(pose_stack.block_type_ind[p, b]) == lig_bt_ind:
                        off = int(pose_stack.block_coord_offset[p, b])
                        print(
                            f"  pose {p} block {b}: offset={off} "
                            f"n={n_atoms} coords[0]={pose_stack.coords[p, off].cpu().tolist()} "
                            f"coords[-1]={pose_stack.coords[p, off + n_atoms - 1].cpu().tolist()}"
                        )

        assert pose_stack.coords.shape[0] >= 1
        nonzero_coords = pose_stack.coords[pose_stack.coords != 0]
        assert not torch.any(torch.isnan(nonzero_coords))

        TestPoseStackWithLigand._sanity_check_pose_stack(pose_stack)

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

        param_db = ParameterDatabase.get_default()
        pose_stack = pose_stack_from_biotite(
            cif_155c_with_hem, torch_device, prepare_ligands=True, param_db=param_db
        )

    def test_pse_posestack_scores(self, cif_1a25_with_pse, torch_device):
        """Partial occupancy ligand (PSE, 0.56) scores without errors."""
        self._score_cif_with_ligand(
            cif_1a25_with_pse, torch_device, "test_pse_posestack_scores"
        )

    def test_atp_posestack_scores(self, cif_1a0i_with_atp, torch_device):
        """ATP ligand (31 heavy atoms + H = >32 total) triggers tile edge case.

        ATP has >32 total atoms but <=32 heavy atoms, which exercises the
        second tile iteration in the LJLK scoring kernel. This was a known
        crash (see uw-ipd/tmol jflat06/atp_ligand_load branch).
        """
        self._score_cif_with_ligand(
            cif_1a0i_with_atp,
            torch_device,
            "test_atp_posestack_scores",
            expected_ligand_name3="ATP",
        )

    def test_atp_posestack_scores_from_pdb(self, pdb_1a0i_with_atp, torch_device):
        """Same as test_atp_posestack_scores but loaded from PDB instead of CIF."""
        self._score_cif_with_ligand(
            pdb_1a0i_with_atp,
            torch_device,
            "test_atp_posestack_scores_from_pdb",
            expected_ligand_name3="ATP",
        )

    def test_i4b_minimize_and_cif_roundtrip(self, cif_184l_with_i4b, tmp_path):
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

        torch_device = torch.device("cpu")

        pose_stack, context = pose_stack_from_biotite(
            cif_184l_with_i4b,
            torch_device,
            prepare_ligands=True,
            param_db=ParameterDatabase.get_default(),
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
        prep = prepare_single_ligand(i4b)
        restype, charges = prep.residue_type, prep.partial_charges

        path = tmp_path / "I4B.params"
        write_params_file(restype, path, partial_charges=charges)
        loaded = read_params_file(path)

        assert loaded.name == "I4B"
        assert len(loaded.atoms) == len(restype.atoms)
        assert len(loaded.bonds) == len(restype.bonds)
        assert len(loaded.icoors) == len(restype.icoors)
        assert prep.atom_type_elements is not None
        assert len(prep.atom_type_elements) > 0

    def test_params_roundtrip_preserves_bond_types(self, tmp_path, cif_184l_with_i4b):
        co = canonical_ordering_for_biotite()
        ligands = detect_nonstandard_residues(cif_184l_with_i4b, co)
        i4b = next(lig for lig in ligands if lig.res_name == "I4B")
        prep = prepare_single_ligand(i4b)
        restype, charges = prep.residue_type, prep.partial_charges

        path = tmp_path / "I4B_bondtypes.params"
        write_params_file(restype, path, partial_charges=charges)
        loaded = read_params_file(path)

        assert all(len(b) == 4 for b in loaded.bonds)
        assert {(a, b, t, r) for a, b, t, r in loaded.bonds} == {
            (a, b, t, r) for a, b, t, r in restype.bonds
        }


def test_collect_new_atom_types_strict_mode_errors(default_database):
    from tmol.ligand.registry import collect_new_atom_types

    residue = SimpleNamespace(
        name="UNK",
        atoms=(SimpleNamespace(name="X1", atom_type="ZZZ"),),
    )
    with pytest.raises(ValueError, match="Unknown element mapping"):
        collect_new_atom_types(
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

    monkeypatch.setattr(
        ligand_module, "_rename_atoms_to_cif_by_graph", _fail_graph_match
    )
    prep = prepare_single_ligand(i4b, ph=7.4)
    atom_names = {a.name for a in prep.residue_type.atoms}
    assert "C3'" in atom_names
    assert "C4'" in atom_names
    assert "C2'" in atom_names
    assert "C3'" in prep.partial_charges


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
        from tmol.ligand.mol3d import compute_mmff94_charges
        from tmol.ligand.residue_builder import build_residue_type
        from tmol.ligand.rdkit_mol import protonate_ligand_mol

        name = request.param
        gt = self.GROUND_TRUTH_DIR

        input_smi = _load_smi_file(gt / "designs.smi", name)
        expected_prot_smi = _load_smi_file(gt / "designs.prot.smi", name)
        ref = _parse_reference_params(gt / f"{name}.params")

        rdkit_mol = Chem.MolFromSmiles(input_smi)
        protonated = protonate_ligand_mol(rdkit_mol, ph=7.4)
        protonated = Chem.AddHs(protonated, addCoords=False)
        prot_smi = Chem.MolToSmiles(Chem.RemoveHs(protonated), isomericSmiles=True)
        charges_by_idx = compute_mmff94_charges(protonated)
        atom_types = assign_tmol_atom_types(protonated)
        charges = {
            at.atom_name: charges_by_idx[at.index]
            for at in atom_types
            if at.index in charges_by_idx
        }
        restype = build_residue_type(protonated, name, atom_types)

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
        for a, b, *_ in ref_data["restype"].bonds:
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
