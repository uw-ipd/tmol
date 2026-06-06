"""Tests for non-destructive mol2 passthrough preparation."""

from pathlib import Path

import pytest

from tmol.ligand.mol2_io import (
    parse_mol2_atom_names,
    parse_mol2_sybyl_types,
    prepare_ligand_from_mol2_passthrough,
    read_mol2,
)
from tmol.ligand.params_io import read_params_file

GROUND_TRUTH_DIR = Path(__file__).parent.parent / "data" / "ligand_ground_truth"
REF1_MOL2 = GROUND_TRUTH_DIR / "ref1.mol2"
REF2_MOL2 = GROUND_TRUTH_DIR / "ref2.mol2"
REF1_PARAMS = GROUND_TRUTH_DIR / "ref1.params"
REF2_PARAMS = GROUND_TRUTH_DIR / "ref2.params"


def _parse_mol2_atom_block(path: Path) -> list[dict]:
    rows: list[dict] = []
    in_atom = False
    with open(path) as fh:
        for line in fh:
            stripped = line.strip()
            if stripped.startswith("@<TRIPOS>"):
                in_atom = stripped == "@<TRIPOS>ATOM"
                continue
            if not in_atom or not stripped:
                continue
            cols = stripped.split()
            if len(cols) < 9:
                continue
            rows.append(
                {
                    "name": cols[1],
                    "sybyl": cols[5],
                    "charge": float(cols[8]),
                }
            )
    return rows


def test_read_mol2_returns_first_molecule():
    mol = read_mol2(REF1_MOL2)
    assert mol.OBMol.NumAtoms() == 51


def test_parse_mol2_atom_names_matches_atom_block():
    parsed = parse_mol2_atom_names(REF1_MOL2)
    expected = [row["name"] for row in _parse_mol2_atom_block(REF1_MOL2)]
    assert parsed == expected


def test_parse_mol2_sybyl_types():
    sybyl = parse_mol2_sybyl_types(REF1_MOL2)
    expected = [row["sybyl"] for row in _parse_mol2_atom_block(REF1_MOL2)]
    assert sybyl == expected


def test_atom_names_preserved_by_default():
    prep = prepare_ligand_from_mol2_passthrough(REF1_MOL2, res_name="LG1")
    mol2_names = [row["name"] for row in _parse_mol2_atom_block(REF1_MOL2)]
    assert [a.name for a in prep.residue_type.atoms] == mol2_names


@pytest.mark.parametrize(
    ("mol2_path", "params_path"),
    [(REF1_MOL2, REF1_PARAMS), (REF2_MOL2, REF2_PARAMS)],
)
def test_passthrough_atom_types_match_rosetta_params(mol2_path, params_path):
    if not params_path.is_file():
        pytest.skip(f"missing reference params: {params_path}")
    prep = prepare_ligand_from_mol2_passthrough(mol2_path, res_name="LG1")
    ref = read_params_file(params_path)
    ref_types = {a.name: a.atom_type for a in ref.atoms}
    got_types = {a.name: a.atom_type for a in prep.residue_type.atoms}
    mismatches = [
        (name, got_types[name], ref_types[name])
        for name in sorted(ref_types)
        if name in got_types and got_types[name] != ref_types[name]
    ]
    assert not mismatches, f"atom type mismatches: {mismatches[:10]}"
