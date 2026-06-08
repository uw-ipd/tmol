"""Cross-check mol2 passthrough atom types against checked-in .tmol references.

The PLI ``*.xtal-lig.mmff94.tmol`` files are regenerated from the same
OpenBabel-native mol2 passthrough pipeline (Rosetta mol2genparams-aligned).
This guards the live prep path against reference drift.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PLI_DIR = Path(__file__).parent.parent / "data" / "protein_ligand_test"
TMOL_SUFFIX = ".xtal-lig.mmff94.tmol"
_PARITY_MOL2 = sorted(p.name for p in PLI_DIR.glob("*.lig.mol2"))


def _tmol_types_by_name(tmol_path: Path) -> dict[str, str]:
    from tmol.ligand.params_file import load_params_file

    prep = load_params_file(tmol_path)[0]
    return {
        a.name: a.atom_type
        for a in prep.residue_type.atoms
        if not str(a.name).startswith("H")
    }


def _passthrough_types_by_name(mol2_path: Path, res_name: str) -> dict[str, str]:
    from tmol.ligand.mol2_io import prepare_ligand_from_mol2_passthrough

    prep = prepare_ligand_from_mol2_passthrough(mol2_path, res_name=res_name)
    return {
        a.name: a.atom_type
        for a in prep.residue_type.atoms
        if not str(a.name).startswith("H")
    }


@pytest.mark.parametrize("mol2_file", _PARITY_MOL2)
def test_atom_types_match_tmol_reference(mol2_file: str):
    """Passthrough mol2 typing must match the checked-in .tmol reference."""
    mol2_path = PLI_DIR / mol2_file
    target = mol2_file[: -len(".lig.mol2")]
    tmol_path = PLI_DIR / f"{target}{TMOL_SUFFIX}"
    if not mol2_path.is_file():
        pytest.skip(f"missing fixture {mol2_path}")
    if not tmol_path.is_file():
        pytest.skip(f"missing reference {tmol_path}")

    from tmol.ligand.params_file import load_params_file

    ref_res_name = load_params_file(tmol_path)[0].residue_type.name
    reference = _tmol_types_by_name(tmol_path)
    generated = _passthrough_types_by_name(mol2_path, res_name=ref_res_name)

    shared = sorted(set(reference) & set(generated))
    assert shared, "no shared atom names between reference and generated prep"

    mismatches = [
        (name, generated[name], reference[name])
        for name in shared
        if generated[name] != reference[name]
    ]
    assert not mismatches, (
        f"{mol2_file}: {len(mismatches)} type mismatches vs .tmol reference "
        f"(first 10): {mismatches[:10]}"
    )
