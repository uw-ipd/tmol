from pathlib import Path

import numpy as np

from tmol_jax.io import (
    canonical_form_from_pdb,
    default_packed_block_types,
    pose_stack_from_canonical_form,
)


PDB = Path(__file__).parents[2] / "tmol/tests/data/pdb/1ubq.pdb"


def _three_residues():
    canonical = canonical_form_from_pdb(PDB)
    return {
        key: value[:, :3].copy() if isinstance(value, np.ndarray) else value
        for key, value in canonical.items()
    }


def _variant_names(pose):
    return [pose.block_type(0, index).name for index in range(pose.max_n_blocks)]


def test_explicit_polymer_break_selects_terminal_variants_and_disconnects():
    canonical = _three_residues()
    canonical["res_not_connected"] = np.zeros((1, 3, 2), dtype=bool)
    canonical["res_not_connected"][0, 0, 1] = True
    pose = pose_stack_from_canonical_form(canonical, default_packed_block_types())
    names = _variant_names(pose)
    assert "nterm" in names[0] and "cterm" in names[0]
    assert "nterm" in names[1] and "cterm" not in names[1]
    assert np.all(np.asarray(pose.inter_residue_connections)[0, 0] == -1)
    assert 0 not in np.asarray(pose.inter_residue_connections)[0, 1, :, 0]
    assert np.isfinite(np.asarray(pose.coords)).all()


def test_internal_padding_creates_two_chain_segments():
    canonical = _three_residues()
    canonical["res_name"][0, 1] = ""
    canonical["res_types"][0, 1] = -1
    canonical["chain_id"][0, 1] = -1
    canonical["coords"][0, 1] = np.nan
    canonical["atom_is_present"][0, 1] = False
    pose = pose_stack_from_canonical_form(canonical, default_packed_block_types())
    assert pose.max_n_blocks == 2
    assert all(
        "nterm" in name and "cterm" in name for name in _variant_names(pose)
    )
    assert np.all(np.asarray(pose.inter_residue_connections) == -1)
    assert np.isfinite(np.asarray(pose.coords)).all()


def test_keyword_metadata_is_used_when_canonical_values_are_none():
    canonical = _three_residues()
    canonical["chain_id"] = None
    canonical["res_num"] = None
    pose = pose_stack_from_canonical_form(
        canonical,
        default_packed_block_types(),
        chain_id=[[0, 0, 1]],
        res_num=[[10, 11, 20]],
    )
    assert np.asarray(pose.pdb_info.res_num)[0].tolist() == [10, 11, 20]
    assert 2 not in np.asarray(pose.inter_residue_connections)[0, 1, :, 0]
    assert 1 not in np.asarray(pose.inter_residue_connections)[0, 2, :, 0]
