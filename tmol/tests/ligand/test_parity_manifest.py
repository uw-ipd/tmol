"""Tests for the dataset-driven parity manifest loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tmol.ligand.parity_manifest import (
    LigandParityEntry,
    default_dataset_manifest,
    load_parity_manifest,
)

_GROUND_TRUTH = Path(__file__).parent.parent / "data" / "ligand_ground_truth"


def _write_manifest(tmp_path, records: list, *, wrap: bool = True) -> Path:
    """Write a temporary manifest JSON (optionally wrapped) and return its path."""
    # a minimal params file each record can point at
    params = tmp_path / "m.params"
    params.write_text("NAME m\nATOM C1 CT X 0.0\n")
    for rec in records:
        rec.setdefault("params", "m.params")
    payload = {"molecules": records} if wrap else records
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload))
    return manifest


def test_seed_entries_are_smiles_only() -> None:
    """Built-in seed entries are SMILES-only and carry expected metadata."""
    entries = load_parity_manifest()
    assert len(entries) >= 2
    names = {e.name for e in entries}
    assert {"ref1", "ref2"} <= names
    for entry in entries:
        assert entry.mol2 is None  # orphan ref mol2s are not used as pairs
        assert not entry.has_mol2
        assert entry.input_smiles
        assert entry.expected_prot_smiles
        assert entry.params.exists()
        assert entry.charge_mode == "auto"
        # Seed references carry PROTON_CHI, so seed entries request sampling.
        assert entry.sample_proton_chi is True


def test_manifest_load_resolves_relative_paths(tmp_path) -> None:
    """Relative manifest paths resolve against the manifest directory."""
    (tmp_path / "lig.mol2").write_text("@<TRIPOS>MOLECULE\nx\n")
    manifest = _write_manifest(
        tmp_path,
        [
            {
                "name": "x",
                "input_smiles": "C",
                "expected_prot_smiles": "C",
                "mol2": "lig.mol2",
                "charge_mode": "auto",
                "sample_proton_chi": False,
                "expected_unsupported_fields": ["CUT_BOND"],
            }
        ],
    )
    entries = load_parity_manifest(manifest)
    assert len(entries) == 1
    entry = entries[0]
    assert isinstance(entry, LigandParityEntry)
    assert entry.has_mol2
    assert entry.mol2 == tmp_path / "lig.mol2"
    assert entry.params == tmp_path / "m.params"
    assert entry.expected_unsupported_fields == ("CUT_BOND",)


def test_manifest_accepts_bare_list(tmp_path) -> None:
    """A bare top-level list of molecules is accepted."""
    manifest = _write_manifest(
        tmp_path,
        [{"name": "x", "input_smiles": "C", "expected_prot_smiles": "C"}],
        wrap=False,
    )
    assert len(load_parity_manifest(manifest)) == 1


def test_manifest_rejects_missing_expected_prot_smiles(tmp_path) -> None:
    """A record without expected_prot_smiles raises ValueError."""
    manifest = _write_manifest(tmp_path, [{"name": "x", "input_smiles": "C"}])
    with pytest.raises(ValueError, match="expected_prot_smiles"):
        load_parity_manifest(manifest)


def test_manifest_rejects_missing_params_file(tmp_path) -> None:
    """A record pointing at a missing params file raises FileNotFoundError."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "molecules": [
                    {
                        "name": "x",
                        "input_smiles": "C",
                        "expected_prot_smiles": "C",
                        "params": "does_not_exist.params",
                    }
                ]
            }
        )
    )
    with pytest.raises(FileNotFoundError):
        load_parity_manifest(manifest)


def test_manifest_rejects_missing_mol2_file(tmp_path) -> None:
    """A record pointing at a missing mol2 file raises FileNotFoundError."""
    manifest = _write_manifest(
        tmp_path,
        [
            {
                "name": "x",
                "input_smiles": "C",
                "expected_prot_smiles": "C",
                "mol2": "ghost.mol2",
            }
        ],
    )
    with pytest.raises(FileNotFoundError):
        load_parity_manifest(manifest)


def test_missing_manifest_path_raises() -> None:
    """Loading a non-existent manifest path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_parity_manifest(_GROUND_TRUTH / "no_such_manifest.json")


def test_entry_count_grows_with_manifest(tmp_path) -> None:
    """The number of loaded entries tracks the manifest record count."""
    one = _write_manifest(
        tmp_path, [{"name": "a", "input_smiles": "C", "expected_prot_smiles": "C"}]
    )
    assert len(load_parity_manifest(one)) == 1
    two_dir = tmp_path / "two"
    two_dir.mkdir()
    two = _write_manifest(
        two_dir,
        [
            {"name": "a", "input_smiles": "C", "expected_prot_smiles": "C"},
            {"name": "b", "input_smiles": "N", "expected_prot_smiles": "N"},
        ],
    )
    assert len(load_parity_manifest(two)) == 2


@pytest.mark.skipif(
    not default_dataset_manifest().exists(),
    reason="DUD80 dataset manifest not present (git-ignored)",
)
def test_dud80_manifest_loads_all_paired_entries() -> None:
    """The DUD80 dataset manifest loads all 80 paired entries."""
    entries = load_parity_manifest(default_dataset_manifest())
    assert len(entries) == 80
    for entry in entries:
        assert entry.has_mol2
        assert entry.mol2.exists()
        assert entry.params.exists()
        assert entry.charge_mode == "auto"
