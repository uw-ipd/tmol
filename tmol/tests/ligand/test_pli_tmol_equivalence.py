"""Regression tests for protein_ligand_test file-to-.tmol equivalence.

For each PLI target, this suite checks both ligand-source formats:

1. ``cif_inputs/<target>.ligand.cif`` -> ``prepare_single_ligand``
2. ``<target>.lig.mol2``             -> ``prepare_single_ligand``

Each generated ``LigandPreparation`` must match the checked-in reference
``<target>.xtal-lig.mmff94.tmol`` under the same normalization semantics used
for DUD regression tests.
"""

from pathlib import Path

import attr
import pytest

from tmol.ligand.cif_normalization import (
    audit_cif_bonds_vs_mol2,
    repaired_cif_path_from_mol2,
)
from tmol.ligand.equivalence import compare_ligand_preparations

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

PLI_FORMAT_CASES = sorted(
    [("cif", target) for target in (TMOL_TARGETS & CIF_TARGETS)]
    + [("mol2", target) for target in (TMOL_TARGETS & MOL2_TARGETS)],
    key=lambda x: (x[0], x[1]),
)

CASE_IDS = [f"{source}_{target}" for source, target in PLI_FORMAT_CASES]


def _source_path(source: str, target: str) -> Path:
    if source == "cif":
        return PLI_CIF_DIR / f"{target}.ligand.cif"
    if source == "mol2":
        return PLI_DIR / f"{target}.lig.mol2"
    raise ValueError(f"Unsupported source format: {source}")


def _format_check_error(prep_pair: dict, check: str) -> str:
    details = prep_pair["equivalence"].details.get(check)
    return (
        f"{check} mismatch for {prep_pair['target']} ({prep_pair['source']} -> .tmol): "
        f"{details}"
    )


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


@pytest.fixture(scope="class", params=PLI_FORMAT_CASES, ids=CASE_IDS)
def prep_pair(request):
    from tmol.ligand import prepare_single_ligand
    from tmol.ligand.detect import (
        nonstandard_residue_info_from_cif,
        nonstandard_residue_info_from_mol2,
    )
    from tmol.ligand.params_file import load_params_file

    source, target = request.param
    source_path = _source_path(source, target)
    tmol_path = PLI_DIR / f"{target}{TMOL_SUFFIX}"

    preps_tmol = load_params_file(tmol_path)
    assert (
        len(preps_tmol) == 1
    ), f"{tmol_path.name}: expected one residue, got {len(preps_tmol)}"
    prep_tmol = preps_tmol[0]
    ref_res_name = prep_tmol.residue_type.name

    if source == "cif":
        source_info = nonstandard_residue_info_from_cif(
            source_path,
            res_name=ref_res_name,
            paired_mol2_path=PLI_DIR / f"{target}.lig.mol2",
            repair_invalid_bonds=True,
        )
    else:
        source_info = nonstandard_residue_info_from_mol2(
            source_path, res_name=ref_res_name
        )

    prep_source = prepare_single_ligand(source_info, ph=7.4, charge_mode="mmff94")
    equivalence = compare_ligand_preparations(
        prep_source, prep_tmol, charge_tolerance=0.05
    )

    return {
        "source": source,
        "target": target,
        "source_path": source_path,
        "tmol_path": tmol_path,
        "equivalence": equivalence,
    }


class TestPLIFileToTmolEquivalence:
    """File-derived ligand prep must match checked-in `.tmol` references."""

    def test_atom_set(self, prep_pair):
        assert prep_pair["equivalence"].checks["atom_set"], _format_check_error(
            prep_pair, "atom_set"
        )

    def test_atom_types(self, prep_pair):
        assert prep_pair["equivalence"].checks["atom_types"], _format_check_error(
            prep_pair, "atom_types"
        )

    def test_bonds(self, prep_pair):
        assert prep_pair["equivalence"].checks["bonds"], _format_check_error(
            prep_pair, "bonds"
        )

    def test_partial_charges(self, prep_pair):
        assert prep_pair["equivalence"].checks["partial_charges"], _format_check_error(
            prep_pair, "partial_charges"
        )

    def test_cartbonded_params(self, prep_pair):
        assert prep_pair["equivalence"].checks[
            "cartbonded_params"
        ], _format_check_error(prep_pair, "cartbonded_params")


def test_pli_prepare_is_deterministic_when_ccd_smiles_missing():
    """CIF setup should remain stable when optional CCD SMILES is unavailable."""
    from tmol.ligand import prepare_single_ligand
    from tmol.ligand.detect import nonstandard_residue_info_from_cif

    target = "ace"
    cif_path = PLI_CIF_DIR / f"{target}.ligand.cif"
    mol2_path = PLI_DIR / f"{target}.lig.mol2"
    info = nonstandard_residue_info_from_cif(
        cif_path,
        res_name="LG1",
        paired_mol2_path=mol2_path,
        repair_invalid_bonds=True,
    )
    info_without_smiles = attr.evolve(info, ccd_smiles=None)
    prep_with_smiles = prepare_single_ligand(info, ph=7.4, charge_mode="mmff94")
    prep_without_smiles = prepare_single_ligand(
        info_without_smiles, ph=7.4, charge_mode="mmff94"
    )
    comp = compare_ligand_preparations(prep_with_smiles, prep_without_smiles)
    assert comp.is_equivalent, f"Prep changed when ccd_smiles missing: {comp.details}"


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
