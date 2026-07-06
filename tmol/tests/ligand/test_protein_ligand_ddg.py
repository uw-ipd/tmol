"""Deterministic protein–ligand ddG regression from checked-in CIF + params fixtures.

Each case loads a full complex from ``{target}.tmol.nomin.cif`` (protein + aligned
ligand coordinates) and injects the matching golden ``{target}.xtal-lig.mmff94.tmol``
params file. OpenBabel is **not** run at test time — chemistry and icoor geometry
are frozen in the committed ``.tmol`` fixture (same pattern as the guanfeng
SMILES→params golden tests, which compare topology/charges rather than
regenerating conformers per run).

This isolates the production scoring path: **structure file + pre-made params → ddG**.

On-the-fly CIF→SMILES→OpenBabel prep (stochastic geometry) is covered separately
by finiteness tests in ``test_ligand_pipeline`` and ``test_cif_to_dg``.

``cox2`` and ``trypsin`` complex CIFs are not included: they carry explicit
termini atoms (OXT, H1–H3) that standard tmol residue types do not model.
"""

from __future__ import annotations

from pathlib import Path

import numpy
import pytest
import torch

from tmol.ligand.registry import clear_cache

PLI_DATA_DIR = Path(__file__).parent.parent / "data" / "protein_ligand_test"
LIGAND_RES_NAME = "LG1"

# Weighted block-pair ddG (beta2016, minimize=False, pack=False, no_optH=True)
# captured on CPU from fixture CIF + fixture .tmol params.
_GOLDEN_DDG: dict[str, float] = {
    "ace": -20.621658,
    "ache": -33.298431,
    "ada": 12.211016,
    "ampc": -16.555737,
    "ar": -30.130075,
    "cdk2": -21.332310,
    "cox1": -28.610884,
    "egfr": -13.663261,
    "er_agonist": -21.690508,
    "er_antagonist": -39.624435,
    "fgfr1": -25.388765,
    "fxa": -29.457779,
    "gr": -30.795242,
    "hivrt": -33.640625,
    "hmga": -6.930063,
    "hsp90": -31.734354,
    "mr": -24.517738,
    "na": -17.817350,
    "p38": -32.038181,
    "parp": -14.593683,
    "pde5": -40.632866,
    "pdgfrb": 37.943211,
    "pr": -36.805813,
    "rxr": -51.300358,
    "src": -33.015259,
    "tk": -29.968708,
    "vegfr2": 48.421783,
}


@pytest.fixture(autouse=True)
def _clear_ligand_cache() -> None:
    clear_cache()


def _load_complex_cif(target: str):
    import biotite.structure as struc
    import biotite.structure.io

    cif_path = PLI_DATA_DIR / f"{target}.tmol.nomin.cif"
    structure = biotite.structure.io.load_structure(
        str(cif_path), model=1, include_bonds=True
    )
    if isinstance(structure, struc.AtomArrayStack):
        structure = structure[0]
    return structure


def _ligand_block_mask(pose_stack, context, torch_device: torch.device) -> torch.Tensor:
    co = context.canonical_ordering
    res_types = context.canonical_form.res_types[0]
    n_blocks = pose_stack.max_n_blocks
    ligand_mask = torch.zeros((1, n_blocks), dtype=torch.bool, device=torch_device)
    found = False
    for block_idx in range(min(n_blocks, res_types.shape[0])):
        res_type_id = int(res_types[block_idx])
        if res_type_id < 0:
            continue
        if co.restype_io_equiv_classes[res_type_id] == LIGAND_RES_NAME:
            ligand_mask[0, block_idx] = True
            found = True
    assert found, f"{LIGAND_RES_NAME} ligand block not found in pose"
    return ligand_mask


def _weighted_ddg_from_fixtures(target: str, torch_device: torch.device) -> float:
    from tmol.database import ParameterDatabase
    from tmol.io.pose_stack_from_biotite import pose_stack_from_biotite
    from tmol.score import beta2016_score_function
    from tmol.score.score_utils import calculate_block_pair_ddg

    tmol_path = PLI_DATA_DIR / f"{target}.xtal-lig.mmff94.tmol"
    structure = _load_complex_cif(target)

    pose_stack, context = pose_stack_from_biotite(
        structure,
        torch_device,
        prepare_ligands=True,
        ligand_params_files=[str(tmol_path)],
        no_optH=True,
        sample_proton_chi=False,
        param_db=ParameterDatabase.get_default(),
        return_context=True,
    )

    ligand_mask = _ligand_block_mask(pose_stack, context, torch_device)
    sfxn = beta2016_score_function(torch_device, param_db=context.parameter_database)
    ddg = calculate_block_pair_ddg(
        pose_stack,
        ligand_mask,
        sfxn=sfxn,
        minimize=False,
        pack=False,
        database=context.parameter_database,
    )
    assert torch.isfinite(ddg).all(), f"non-finite ddG for {target}: {ddg}"
    return float(ddg.detach()[0])


@pytest.mark.parametrize("target", sorted(_GOLDEN_DDG))
def test_protein_ligand_cif_to_ddg_golden(target: str, torch_device) -> None:
    """CIF complex + golden .tmol params reproduces pinned weighted ddG."""
    actual = _weighted_ddg_from_fixtures(target, torch_device)
    expected = _GOLDEN_DDG[target]
    numpy.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)
