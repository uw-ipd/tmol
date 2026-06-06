"""Cross-check tmol atom typing against Rosetta mol2genparams (AtomTypeClassifier).

Rosetta reference implementation:
``rosetta/source/scripts/python/public/generic_potential/AtomTypeClassifier.py``

Atoms are matched by disambiguated Tripos names (``_TriposAtomName`` on the
RDKit mol) to Rosetta ``MoleculeClass.atms[].name``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PLI_DIR = Path(__file__).parent.parent / "data" / "protein_ligand_test"
ROSETTA_GENPOT = (
    Path(__file__).resolve().parents[4]
    / "rosetta"
    / "source"
    / "scripts"
    / "python"
    / "public"
    / "generic_potential"
)

_PARITY_MOL2 = sorted(p.name for p in PLI_DIR.glob("*.lig.mol2"))


def _rosetta_types_by_name(mol2_path: Path) -> dict[str, str]:
    """Run Rosetta MoleculeClass + AtomTypeClassifier on a mol2 file."""
    if not ROSETTA_GENPOT.is_dir():
        pytest.skip(f"Rosetta generic_potential not found at {ROSETTA_GENPOT}")
    sys.path.insert(0, str(ROSETTA_GENPOT))
    from BasicClasses import OptionClass  # noqa: WPS433
    from Molecule import MoleculeClass  # noqa: WPS433
    from Types import ACLASS_ID  # noqa: WPS433

    argv = ["test_rosetta_mol2gen_parity", "--inputs", str(mol2_path)]
    opt = OptionClass(argv)
    mol = MoleculeClass(str(mol2_path), opt)
    return {a.name: ACLASS_ID[a.aclass] for a in mol.atms if not a.is_H}


def _tmol_types_by_tripos_name(mol2_path: Path) -> dict[str, str]:
    from tmol.ligand.mol2_io import prepare_ligand_from_mol2_passthrough

    prep = prepare_ligand_from_mol2_passthrough(mol2_path, res_name="LG1")
    return {
        a.name: a.atom_type
        for a in prep.residue_type.atoms
        if not str(a.name).startswith("H")
    }


@pytest.mark.parametrize("mol2_file", _PARITY_MOL2)
def test_atom_types_match_rosetta_mol2gen(mol2_file: str):
    """tmol typing must match Rosetta AtomTypeClassifier on the same mol2."""
    mol2_path = PLI_DIR / mol2_file
    if not mol2_path.is_file():
        pytest.skip(f"missing fixture {mol2_path}")

    rosetta = _rosetta_types_by_name(mol2_path)
    tmol = _tmol_types_by_tripos_name(mol2_path)

    shared = sorted(set(rosetta) & set(tmol))
    assert shared, "no shared atom names between Rosetta and tmol loaders"

    mismatches = [
        (name, tmol[name], rosetta[name])
        for name in shared
        if tmol[name] != rosetta[name]
    ]
    assert not mismatches, (
        f"{mol2_file}: {len(mismatches)} type mismatches vs Rosetta "
        f"(first 10): {mismatches[:10]}"
    )
