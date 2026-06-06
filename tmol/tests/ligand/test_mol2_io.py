"""Tests for mol2 passthrough preparation (Rosetta mol2genparams-aligned).

The PLI mol2 equivalence path intentionally does **not** run Dimorphite. Crystallographic
mol2 files already carry explicit hydrogens at the desired protonation state; passthrough
preserves that input protonation, then recomputes MMFF94 partial charges via OpenBabel.

Dimorphite + MMFF94 on the RDKit rebuild path is covered separately (e.g.
``TestGroundTruthRegression`` in ``test_ligand_pipeline.py``, DUD CIF golden tests).
"""

from pathlib import Path

import pytest

from tmol.ligand.mol2_io import (
    parse_mol2_atom_names,
    parse_mol2_elements,
    parse_mol2_sybyl_types,
    prepare_ligand_from_mol2_passthrough,
    read_mol2,
)
from tmol.ligand.params_io import read_params_file

GROUND_TRUTH_DIR = Path(__file__).parent.parent / "data" / "ligand_ground_truth"
PLI_DIR = Path(__file__).parent.parent / "data" / "protein_ligand_test"
REF1_MOL2 = GROUND_TRUTH_DIR / "ref1.mol2"
REF2_MOL2 = GROUND_TRUTH_DIR / "ref2.mol2"
REF1_PARAMS = GROUND_TRUTH_DIR / "ref1.params"
REF2_PARAMS = GROUND_TRUTH_DIR / "ref2.params"
PLI_ACE_MOL2 = PLI_DIR / "ace.lig.mol2"


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


def _hydrogen_count_from_mol2(path: Path) -> int:
    return sum(1 for elem in parse_mol2_elements(path) if elem == "H")


def _input_charges_by_name(path: Path) -> dict[str, float]:
    return {row["name"]: row["charge"] for row in _parse_mol2_atom_block(path)}


@pytest.mark.parametrize("mol2_path", [REF1_MOL2, PLI_ACE_MOL2])
def test_passthrough_preserves_input_protonation(mol2_path: Path):
    """Explicit H from mol2 are kept — Dimorphite is not applied on this path."""
    prep = prepare_ligand_from_mol2_passthrough(
        mol2_path, res_name="LG1", charge_mode="mmff94"
    )
    mol2_h = _hydrogen_count_from_mol2(mol2_path)
    out_h = sum(1 for a in prep.residue_type.atoms if a.element == "H")
    assert mol2_h > 0, f"{mol2_path.name} should include explicit hydrogens"
    assert out_h == mol2_h
    assert len(prep.partial_charges) == len(prep.residue_type.atoms)


@pytest.mark.parametrize("mol2_path", [REF1_MOL2, PLI_ACE_MOL2])
def test_passthrough_mmff94_recomputes_charges(mol2_path: Path):
    """charge_mode=mmff94 must recompute charges, not pass through mol2 columns."""
    prep_mmff = prepare_ligand_from_mol2_passthrough(
        mol2_path, res_name="LG1", charge_mode="mmff94"
    )
    prep_input = prepare_ligand_from_mol2_passthrough(
        mol2_path, res_name="LG1", charge_mode="input"
    )
    input_by_name = _input_charges_by_name(mol2_path)

    assert prep_mmff.partial_charges != prep_input.partial_charges
    assert any(
        abs(prep_mmff.partial_charges[name] - input_by_name[name]) > 1e-4
        for name in input_by_name
        if name in prep_mmff.partial_charges
    )

    net = sum(prep_mmff.partial_charges.values())
    assert (
        abs(net - round(net)) < 0.05
    ), f"MMFF94 net charge should be near-integer: {net}"


def test_prepare_single_ligand_mol2_passthrough_skips_dimorphite(monkeypatch):
    """prepare_mode=passthrough on mol2 must not call Dimorphite protonation."""
    from tmol.ligand import prepare_single_ligand
    from tmol.ligand.detect import nonstandard_residue_info_from_mol2
    import tmol.ligand.rdkit_mol as rdkit_mol

    def _fail_dimorphite(*_args, **_kwargs):
        raise AssertionError(
            "Dimorphite protonation should not run for mol2 passthrough"
        )

    monkeypatch.setattr(rdkit_mol, "protonate_ligand_mol", _fail_dimorphite)

    info = nonstandard_residue_info_from_mol2(PLI_ACE_MOL2, res_name="LG1")
    prep = prepare_single_ligand(
        info,
        ph=7.4,
        charge_mode="mmff94",
        prepare_mode="passthrough",
    )
    assert _hydrogen_count_from_mol2(PLI_ACE_MOL2) == sum(
        1 for a in prep.residue_type.atoms if a.element == "H"
    )


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
