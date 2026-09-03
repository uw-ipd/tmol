"""Correctness of the ``context=`` reuse path in ``pose_stack_from_biotite``.

Scoring many structures that share one ligand can build the expensive,
structure-independent ``PoseBuildContext`` once and reuse it, recomputing
only the per-structure canonical form. These tests check that reusing a context
reproduces the ``prepare_ligands=True`` result and is stable across repeats.
"""

from pathlib import Path

import biotite.structure as struc
import biotite.structure.io
import pytest
import torch

from tmol.database import ParameterDatabase
from tmol.io import (
    build_context_from_biotite,
    pose_stack_from_biotite,
)
from tmol.pack import build_missing_sidechains
from tmol.score import ScoreType

PLI_DATA_DIR = Path(__file__).parent.parent / "data" / "protein_ligand_test"
TARGET = "ace"
N_REPEATS = 3


def _load_complex_cif(target: str) -> struc.AtomArray:
    cif_path = PLI_DATA_DIR / f"{target}.tmol.nomin.cif"
    structure = biotite.structure.io.load_structure(
        str(cif_path), model=1, include_bonds=True
    )
    if isinstance(structure, struc.AtomArrayStack):
        structure = structure[0]
    return structure


def _params_files(target: str = TARGET) -> list[str]:
    return [str(PLI_DATA_DIR / f"{target}.xtal-lig.mmff94.tmol")]


def _build_context(structure, torch_device):
    return build_context_from_biotite(
        structure,
        torch_device,
        prepare_ligands=True,
        ligand_params_files=_params_files(),
        param_db=ParameterDatabase.get_default(),
    )


def _assert_pose_stacks_equal(a, b) -> None:
    assert a.coords.shape == b.coords.shape
    assert torch.equal(a.block_type_ind, b.block_type_ind)
    assert torch.equal(a.real_atoms, b.real_atoms)
    real = a.real_atoms
    assert torch.allclose(a.coords[real], b.coords[real], atol=1e-5, equal_nan=True)


def test_reused_context_matches_prepare_ligands(torch_device):
    """A pose built from a reused context equals one built with prepare_ligands."""
    structure = _load_complex_cif(TARGET)

    reference, _ = pose_stack_from_biotite(
        structure,
        torch_device,
        prepare_ligands=True,
        ligand_params_files=_params_files(),
        param_db=ParameterDatabase.get_default(),
        no_optH=True,
        return_context=True,
    )

    context = _build_context(structure, torch_device)
    pose_stack = pose_stack_from_biotite(
        structure, torch_device, context=context, no_optH=True
    )
    _assert_pose_stacks_equal(reference, pose_stack)


def test_repeated_calls_with_same_context_stable(torch_device):
    """Reusing one context across repeated calls is stable."""
    structure = _load_complex_cif(TARGET)
    context = _build_context(structure, torch_device)

    first = pose_stack_from_biotite(
        structure, torch_device, context=context, no_optH=True
    )
    for _ in range(N_REPEATS):
        pose_stack = pose_stack_from_biotite(
            structure, torch_device, context=context, no_optH=True
        )
        _assert_pose_stacks_equal(first, pose_stack)


def test_context_opth_score_function_omits_invariant_terms(biotite_1ubq, torch_device):
    """The lighter OptH scorer preserves its rotamer assignment exactly."""
    context = build_context_from_biotite(biotite_1ubq, torch_device)
    score_function = context._opth_score_function
    for score_type in (
        ScoreType.disulfide,
        ScoreType.omega,
        ScoreType.rama,
        ScoreType.ref,
        ScoreType.na_torsion,
        ScoreType.na_torsion_well,
    ):
        assert score_function.get_weight(score_type) == 0
    assert score_function.get_weight(ScoreType.hbond) != 0

    pose = pose_stack_from_biotite(
        biotite_1ubq, torch_device, context=context, no_optH=True
    )
    missing = torch.zeros_like(pose.block_type_ind, dtype=torch.bool)
    expected = build_missing_sidechains(
        pose, context._packing_score_function, context._dunbrack_sampler, missing
    )
    actual = build_missing_sidechains(
        pose, score_function, context._dunbrack_sampler, missing
    )
    torch.testing.assert_close(actual.block_type_ind, expected.block_type_ind)
    torch.testing.assert_close(actual.coords, expected.coords)


def test_context_and_param_db_mutually_exclusive(torch_device):
    """Supplying both context= and param_db= is ambiguous and should raise."""
    structure = _load_complex_cif(TARGET)
    context = _build_context(structure, torch_device)

    with pytest.raises(ValueError):
        pose_stack_from_biotite(
            structure,
            torch_device,
            context=context,
            param_db=ParameterDatabase.get_default(),
        )


def test_context_and_prepare_ligands_mutually_exclusive(torch_device):
    """Supplying both context= and prepare_ligands=True should raise."""
    structure = _load_complex_cif(TARGET)
    context = _build_context(structure, torch_device)

    with pytest.raises(ValueError):
        pose_stack_from_biotite(
            structure,
            torch_device,
            context=context,
            prepare_ligands=True,
        )
