"""Reference-integrity guardrails for the protein_ligand_test inputs.

The PLI set is used for dG scoring against golden references (see
``test_pli_scoring.py``). These tests only verify that the checked-in
CIF / MOL2 / ``.tmol`` inputs are mutually consistent and that the loader
behaves deterministically.

Generated-param equivalence (CIF/MOL2 -> ``.tmol``) is intentionally **not**
asserted here. Param-generation parity against Rosetta is covered by the
DUD-80 suite (``test_ligand_pipeline.py::TestGroundTruthRegression`` and
``scripts/ligand_prep/validate_dud80.py``); scoring correctness on this set is
covered by ``test_pli_scoring.py`` via the golden ``.tmol`` params.
"""

from pathlib import Path

import pytest

from tmol.ligand.cif_normalization import (
    audit_cif_bonds_vs_mol2,
    repaired_cif_path_from_mol2,
)

PLI_DIR = Path(__file__).parent.parent / "data" / "protein_ligand_test"
PLI_CIF_DIR = PLI_DIR / "cif_inputs"
TMOL_SUFFIX = ".xtal-lig.mmff94.tmol"

TMOL_TARGETS = {
    p.name[: -len(TMOL_SUFFIX)] for p in PLI_DIR.glob(f"*{TMOL_SUFFIX}") if p.is_file()
}
CIF_TARGETS = {
    p.name[: -len(".ligand.cif")]
    for p in PLI_CIF_DIR.glob("*.ligand.cif")
    if p.is_file()
}
MOL2_TARGETS = {
    p.name[: -len(".lig.mol2")] for p in PLI_DIR.glob("*.lig.mol2") if p.is_file()
}


def test_pli_reference_inputs_are_complete():
    """Guardrail: every reference .tmol target should have both CIF and MOL2 inputs."""
    missing_cif = sorted(TMOL_TARGETS - CIF_TARGETS)
    missing_mol2 = sorted(TMOL_TARGETS - MOL2_TARGETS)
    assert not missing_cif, f"Missing PLI CIF inputs for targets: {missing_cif}"
    assert not missing_mol2, f"Missing PLI MOL2 inputs for targets: {missing_mol2}"


def test_pli_cif_bond_tables_audit_and_regeneration():
    """Every paired CIF/MOL2 must be auditable and repairable."""
    shared = sorted(TMOL_TARGETS & CIF_TARGETS & MOL2_TARGETS)
    assert shared, "No shared PLI CIF/MOL2 targets found"
    for target in shared:
        cif_path = PLI_CIF_DIR / f"{target}.ligand.cif"
        mol2_path = PLI_DIR / f"{target}.lig.mol2"
        audit = audit_cif_bonds_vs_mol2(cif_path, mol2_path)
        if audit.consistent:
            continue
        repaired_path, _, regenerated = repaired_cif_path_from_mol2(
            cif_path,
            mol2_path,
            res_name="LG1",
        )
        try:
            repaired_audit = audit_cif_bonds_vs_mol2(repaired_path, mol2_path)
            assert repaired_audit.consistent, (
                f"Regenerated CIF for {target} still mismatches paired MOL2; "
                f"missing={repaired_audit.missing_in_cif} extra={repaired_audit.extra_in_cif}"
            )
        finally:
            if regenerated:
                repaired_path.unlink(missing_ok=True)


def test_prepare_single_ligand_rejects_topology_only_single_bonds():
    import biotite.structure as struc
    import biotite.structure.io

    from tmol.io.pose_stack_from_biotite import canonical_ordering_for_biotite
    from tmol.ligand import prepare_single_ligand
    from tmol.ligand.detect import detect_nonstandard_residues

    pdb_path = PLI_DIR / "ace_complex_nometals.pdb"
    bt_struct = biotite.structure.io.load_structure(str(pdb_path), include_bonds=True)
    if isinstance(bt_struct, struc.AtomArrayStack):
        bt_struct = bt_struct[0]

    ligands = detect_nonstandard_residues(bt_struct, canonical_ordering_for_biotite())
    lg1 = next(l for l in ligands if l.res_name == "LG1")

    with pytest.raises(
        ValueError,
        match=(
            "topology-only SINGLE bonds|"
            "PDB ligand chemistry inference is unsupported|"
            "unsupported bond type codes"
        ),
    ):
        prepare_single_ligand(lg1, ph=7.4)
